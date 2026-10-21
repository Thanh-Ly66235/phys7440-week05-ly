"""MCP server that exposes SIMBAD and arXiv queries as LLM-callable tools.

PHYS 7440 (Generative AI in Physics Research), Week 5 homework - Thanh Ly

The server speaks the Model Context Protocol (JSON-RPC 2.0 messages over
stdin/stdout) through the official ``mcp`` Python SDK and exposes two tools:

* ``query_simbad(object_name)`` - object type, spectral type, parallax and
  radial velocity of one object, from the SIMBAD TAP service (astroquery.simbad);
* ``search_arxiv(keyword, max_results)`` - the most recently submitted arXiv
  papers whose abstracts contain a keyword (arxiv package).

The JSON Schema that MCP advertises for each tool is generated from the type
hints below. Every argument has a type, a description, examples and range
constraints, and each return type is a ``TypedDict`` (a plain ``dict`` at run
time) with a documented field list, so an output schema is advertised as well.
Arguments that violate the schema are rejected by the SDK before the tool
function runs, and the client receives a JSON-RPC result with ``isError: true``.

Usage
-----
    python mcp_server_week05_ly.py               # serve MCP over stdio (the notebook does this)
    python mcp_server_week05_ly.py --list-tools  # print the advertised JSON Schemas and exit
    python mcp_server_week05_ly.py --check       # call both tools once (network test) and exit

SIMBAD and arXiv are public services, so the server itself needs no API key.
Its optional settings, ``SIMBAD_MIRROR``, ``SIMBAD_TIMEOUT_S`` and
``ARXIV_TIMEOUT_S``, are read from the environment or else from the ``.env``
file next to this module (see ``.env.example``). Only the lines that define
these three settings are parsed from ``.env``; every other line, including the
Anthropic API key, is skipped without being parsed, stored or exported.

In stdio mode stdout carries the JSON-RPC stream, so this module never prints
there while serving; diagnostics go to stderr through ``logging``.
"""

import argparse
import asyncio
import io
import json
import logging
import math
import os
import re
import sys
import threading
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict, TypeVar

import arxiv
import numpy as np
from astroquery import log as astroquery_log
from astroquery.simbad import SimbadClass
from dotenv import dotenv_values
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SERVER_SETTINGS = ("SIMBAD_MIRROR", "SIMBAD_TIMEOUT_S", "ARXIV_TIMEOUT_S")


def _read_server_settings(path: Path) -> dict[str, str | None]:
    """Parse only the lines of a .env file that define one of SERVER_SETTINGS (the API key line is skipped)."""
    if not path.is_file():
        return {}
    pattern = re.compile(rf"^\s*(?:export\s+)?(?:{'|'.join(SERVER_SETTINGS)})\s*=")
    wanted = [line for line in path.read_text(encoding="utf-8").splitlines() if pattern.match(line)]
    return dict(dotenv_values(stream=io.StringIO("\n".join(wanted))))


_DOTENV = _read_server_settings(Path(__file__).resolve().with_name(".env"))


def _setting(name: str, default: str) -> str:
    """Server setting: environment variable first, then the .env file, then the default."""
    return os.environ.get(name) or _DOTENV.get(name) or default


SIMBAD_MIRROR = _setting("SIMBAD_MIRROR", "simbad.cds.unistra.fr")    # or "simbad.harvard.edu"
SIMBAD_TIMEOUT_S = float(_setting("SIMBAD_TIMEOUT_S", "30"))
ARXIV_TIMEOUT_S = float(_setting("ARXIV_TIMEOUT_S", "60"))
ARXIV_MAX_RESULTS = 25            # hard upper bound advertised in the search_arxiv schema
SIMBAD_WEB_URL = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident={}"

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [mcp-server] %(levelname)s %(message)s")
logger = logging.getLogger("mcp_server_week05_ly")
astroquery_log.setLevel("WARNING")   # astropy-style loggers print INFO messages to stdout

server = MCPServer(
    name="physics-tools-week05-ly",
    version="1.0.0",
    instructions=("Astronomy research tools. Use query_simbad for catalogue data (object type, "
                  "spectral type, parallax, radial velocity) of one named object, and search_arxiv "
                  "for the most recent arXiv papers whose abstracts mention a keyword."),
)

_simbad = SimbadClass()
_simbad.server = SIMBAD_MIRROR          # validated against astroquery's list of official mirrors
_arxiv_client = arxiv.Client(page_size=ARXIV_MAX_RESULTS, delay_seconds=3.0, num_retries=2)

# One SIMBAD TAP (ADQL) query returns every field the tool needs. SIMBAD normalises the
# identifier match on ident.id, so 'Betelgeuse', 'betelgeuse' and 'alpha Ori' all resolve.
SIMBAD_ADQL = """
SELECT basic.main_id, basic.otype, otypedef.description AS otype_description,
       basic.sp_type, basic.sp_qual,
       basic.plx_value, basic.plx_err, basic.plx_qual, basic.plx_bibcode,
       basic.rvz_radvel, basic.rvz_err, basic.rvz_qual, basic.rvz_type,
       basic.rvz_redshift, basic.rvz_bibcode, basic.ra, basic.dec
FROM basic
JOIN ident ON ident.oidref = basic.oid
LEFT JOIN otypedef ON otypedef.otype = basic.otype
WHERE ident.id = '{identifier}'
"""

T = TypeVar("T")


# --------------------------------------------------------------------------------------
# Output schemas (TypedDicts are ordinary dicts at run time)
# --------------------------------------------------------------------------------------
class SimbadRecord(TypedDict):
    """Basic SIMBAD data for one astronomical object. Values SIMBAD does not have are null."""

    query: Annotated[str, Field(description="Identifier exactly as it was sent to SIMBAD.")]
    main_id: Annotated[str, Field(description="SIMBAD main identifier of the matched object, e.g. '* alf Ori'.")]
    object_type: Annotated[str | None, Field(
        description="SIMBAD object-type code, e.g. 's*r' (red supergiant) or 'PM*' (high proper-motion star).")]
    object_type_description: Annotated[str | None, Field(
        description="Plain-language meaning of object_type, e.g. 'Red Supergiant'.")]
    spectral_type: Annotated[str | None, Field(description="MK spectral type, e.g. 'M1-M2Ia-Iab'.")]
    spectral_type_quality: Annotated[str | None, Field(description="SIMBAD quality flag, A (best) to E (worst).")]
    parallax_mas: Annotated[float | None, Field(description="Trigonometric parallax in milliarcseconds (mas).")]
    parallax_error_mas: Annotated[float | None, Field(description="1-sigma parallax uncertainty in mas.")]
    parallax_quality: Annotated[str | None, Field(description="SIMBAD quality flag of the parallax, A to E.")]
    parallax_reference: Annotated[str | None, Field(description="ADS bibcode of the parallax measurement.")]
    radial_velocity_km_s: Annotated[float | None, Field(
        description="Heliocentric radial velocity in km/s (positive = receding).")]
    radial_velocity_error_km_s: Annotated[float | None, Field(description="1-sigma radial-velocity uncertainty in km/s.")]
    radial_velocity_quality: Annotated[str | None, Field(description="SIMBAD quality flag of the velocity, A to E.")]
    radial_velocity_reference: Annotated[str | None, Field(description="ADS bibcode of the velocity measurement.")]
    redshift: Annotated[float | None, Field(description="Redshift z corresponding to the radial velocity (dimensionless).")]
    ra_deg: Annotated[float | None, Field(description="Right ascension, ICRS J2000, in degrees.")]
    dec_deg: Annotated[float | None, Field(description="Declination, ICRS J2000, in degrees.")]
    simbad_url: Annotated[str, Field(description="SIMBAD web page of the object, for manual verification.")]
    notes: Annotated[list[str], Field(description="Caveats about missing or derived values; empty if none.")]


class ArxivPaper(TypedDict):
    """One arXiv preprint returned by search_arxiv."""

    arxiv_id: Annotated[str, Field(description="arXiv identifier including version, e.g. '2608.19339v2'.")]
    title: Annotated[str, Field(description="Paper title.")]
    authors: Annotated[list[str], Field(description="Author names in the order listed on arXiv.")]
    abstract: Annotated[str, Field(description="Full abstract text as posted on arXiv.")]
    published: Annotated[str, Field(description="Submission date of version 1, YYYY-MM-DD.")]
    updated: Annotated[str, Field(description="Date of the latest version, YYYY-MM-DD.")]
    primary_category: Annotated[str, Field(description="Primary arXiv category, e.g. 'astro-ph.SR'.")]
    url: Annotated[str, Field(description="URL of the arXiv abstract page.")]
    keyword_in_abstract: Annotated[bool, Field(
        description="True if the keyword occurs literally (case-insensitive) in the abstract; arXiv also "
                    "matches word stems, so False means the match came from a variant of the keyword.")]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _call_with_timeout(service: str, timeout_s: float, fn: Callable[[], T]) -> T:
    """Run a blocking network call in its own thread and give up after ``timeout_s`` seconds.

    Each call gets a fresh daemon thread, so a request that never returns can neither block
    later calls of either tool nor keep the server process alive when the client disconnects.
    """
    outcome: dict[str, Any] = {}

    def worker() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:   # re-raised in the calling thread below
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name=f"{service} request", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        logger.warning("%s did not answer within %g s", service, timeout_s)
        raise ToolError(f"{service} did not respond within {timeout_s:g} s; the service may be down "
                        "or overloaded. Try again later.")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _clean(value: Any) -> Any:
    """Convert one astropy table cell to a JSON-serialisable value (masked, NaN or empty -> None)."""
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.floating):
        # SIMBAD's error columns are 32-bit floats: float(np.float32(0.83)) would give 0.8299999833106995,
        # whereas str() gives the shortest decimal that round-trips, i.e. the catalogue value 0.83.
        value = float(str(value))
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _date(timestamp: Any) -> str:
    """Format a datetime as YYYY-MM-DD (empty string if missing; arxiv uses year 1 as its placeholder)."""
    return timestamp.date().isoformat() if timestamp and timestamp.year > 1 else ""


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
@server.tool()
def query_simbad(
    object_name: Annotated[str, Field(
        min_length=1,
        max_length=120,
        description=("Name or catalogue identifier of ONE astronomical object as SIMBAD knows it, e.g. "
                     "'Betelgeuse', 'alf Ori', 'HD 39801', 'Proxima Centauri' or 'M31'. Matching is "
                     "case-insensitive but the spelling must be exact (no fuzzy matching)."),
        examples=["Betelgeuse", "Proxima Centauri", "HD 39801"],
    )],
) -> SimbadRecord:
    """Look up one astronomical object in the SIMBAD database (CDS, Strasbourg).

    Returns the object's SIMBAD main identifier, object type, MK spectral type,
    trigonometric parallax (mas) and heliocentric radial velocity (km/s), each with
    its uncertainty, quality flag and literature reference where available, plus ICRS
    coordinates and a link to the SIMBAD web page. Fields SIMBAD does not have for
    this object are null: report them as unavailable rather than filling them in from
    memory. Fails with an error message if no object matches the identifier.
    """
    identifier = " ".join(object_name.split())
    adql = SIMBAD_ADQL.format(identifier=identifier.replace("'", "''"))   # ADQL escapes ' as ''
    logger.info("query_simbad(%r)", identifier)
    try:
        table = _call_with_timeout("SIMBAD", SIMBAD_TIMEOUT_S, lambda: _simbad.query_tap(adql, maxrec=10))
    except ToolError:
        raise
    except Exception as exc:  # network errors, TAP service errors, malformed responses
        logger.warning("SIMBAD query failed: %r", exc)
        raise ToolError(f"SIMBAD TAP query failed ({type(exc).__name__}: {exc}). "
                        f"The service at {SIMBAD_MIRROR} may be unreachable.") from exc

    if len(table) == 0:
        raise ToolError(f"No SIMBAD object matches the identifier '{identifier}'. Check the spelling, or try "
                        "another designation (common name, Bayer/Flamsteed name, HD or HIP number).")

    row = table[0]
    cell = lambda column: _clean(row[column])   # noqa: E731 - short local accessor
    notes = []
    if len(table) > 1:
        notes.append(f"The identifier matched {len(table)} SIMBAD objects; only the first is reported.")
    if cell("plx_value") is None:
        notes.append("SIMBAD has no trigonometric parallax for this object (normal for extragalactic "
                     "sources), so no parallax distance can be derived.")
    if cell("sp_type") is None:
        notes.append("SIMBAD has no MK spectral type for this object.")
    if cell("rvz_type") in ("z", "c"):
        notes.append("The velocity was derived from a measured redshift (SIMBAD rvz_type "
                     f"'{cell('rvz_type')}'), not measured directly as a radial velocity.")

    return {
        "query": identifier,
        "main_id": cell("main_id") or identifier,
        "object_type": cell("otype"),
        "object_type_description": cell("otype_description"),
        "spectral_type": cell("sp_type"),
        "spectral_type_quality": cell("sp_qual"),
        "parallax_mas": cell("plx_value"),
        "parallax_error_mas": cell("plx_err"),
        "parallax_quality": cell("plx_qual"),
        "parallax_reference": cell("plx_bibcode"),
        "radial_velocity_km_s": cell("rvz_radvel"),
        "radial_velocity_error_km_s": cell("rvz_err"),
        "radial_velocity_quality": cell("rvz_qual"),
        "radial_velocity_reference": cell("rvz_bibcode"),
        "redshift": cell("rvz_redshift"),
        "ra_deg": cell("ra"),
        "dec_deg": cell("dec"),
        "simbad_url": SIMBAD_WEB_URL.format(urllib.parse.quote_plus(identifier)),
        "notes": notes,
    }


@server.tool()
def search_arxiv(
    keyword: Annotated[str, Field(
        min_length=2,
        max_length=100,
        description=("Word or short phrase that must appear in the abstract, e.g. 'Betelgeuse' or "
                     "'red supergiant'. It is searched as one exact phrase in arXiv abstracts; do not "
                     "add boolean operators or arXiv field prefixes."),
        examples=["Betelgeuse", "Proxima Centauri"],
    )],
    max_results: Annotated[int, Field(
        ge=1,
        le=ARXIV_MAX_RESULTS,
        description=f"How many of the most recent matching papers to return (1-{ARXIV_MAX_RESULTS}).",
        examples=[3],
    )],
) -> list[ArxivPaper]:
    """Find the most recently submitted arXiv papers whose abstracts mention a keyword.

    Searches arXiv abstracts for the keyword as an exact phrase and returns up to
    max_results papers, newest submission first. Each paper has its arXiv ID, title,
    authors, full abstract, dates, primary category and URL. Results are ordered by
    submission date, not by relevance, so a paper may mention the keyword only in
    passing; read the abstracts before drawing conclusions. An empty list means that
    no abstract matched.
    """
    phrase = " ".join(keyword.replace('"', " ").split())
    if len(phrase) < 2:
        raise ToolError("The keyword must contain at least two non-quote characters.")
    logger.info("search_arxiv(%r, max_results=%d)", phrase, max_results)
    search = arxiv.Search(
        query=f'abs:"{phrase}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    try:
        results = _call_with_timeout("The arXiv API", ARXIV_TIMEOUT_S,
                                     lambda: list(_arxiv_client.results(search)))
    except ToolError:
        raise
    except Exception as exc:  # HTTP errors, connection errors, malformed feeds
        logger.warning("arXiv query failed: %r", exc)
        raise ToolError(f"arXiv API request failed ({type(exc).__name__}: {exc}).") from exc

    papers: list[ArxivPaper] = []
    for result in results:
        abstract = " ".join(result.summary.split())
        papers.append({
            "arxiv_id": result.get_short_id(),
            "title": " ".join(result.title.split()),
            "authors": [author.name for author in result.authors],
            "abstract": abstract,
            "published": _date(result.published),
            "updated": _date(result.updated),
            "primary_category": result.primary_category,
            "url": result.entry_id,
            "keyword_in_abstract": phrase.lower() in abstract.lower(),
        })
    return papers


# --------------------------------------------------------------------------------------
# Command-line entry point
# --------------------------------------------------------------------------------------
def _print_tool_schemas() -> None:
    """Print the tool list exactly as a client receives it from tools/list."""
    tools = asyncio.run(server.list_tools())
    print(json.dumps([tool.model_dump(by_alias=True, exclude_none=True) for tool in tools], indent=2))


def _self_check() -> None:
    """Call both tools once without an LLM to confirm that SIMBAD and arXiv are reachable."""
    record = query_simbad("Betelgeuse")
    print(f"SIMBAD ok: {record['main_id']} | {record['spectral_type']} | "
          f"parallax {record['parallax_mas']} +/- {record['parallax_error_mas']} mas | "
          f"RV {record['radial_velocity_km_s']} km/s")
    papers = search_arxiv("Betelgeuse", 1)
    print(f"arXiv ok: {papers[0]['arxiv_id']} - {papers[0]['title']}" if papers else "arXiv ok: no papers")


def main(argv: list[str] | None = None) -> None:
    """Serve MCP over stdio, or run one of the diagnostic modes."""
    parser = argparse.ArgumentParser(description="MCP server with SIMBAD and arXiv tools (Week 5, Thanh Ly).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list-tools", action="store_true", help="print the advertised tool schemas and exit")
    mode.add_argument("--check", action="store_true", help="call both tools once and exit")
    args = parser.parse_args(argv)

    if args.list_tools:
        _print_tool_schemas()
    elif args.check:
        _self_check()
    else:
        logger.info("Serving MCP over stdio (SIMBAD mirror %s, timeouts: SIMBAD %g s, arXiv %g s)",
                    SIMBAD_MIRROR, SIMBAD_TIMEOUT_S, ARXIV_TIMEOUT_S)
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
