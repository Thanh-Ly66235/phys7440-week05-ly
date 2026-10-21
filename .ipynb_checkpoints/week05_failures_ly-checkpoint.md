# Week 5 — Failure Modes of the MCP Physics Agent

**Author:** Thanh Ly
**Pipeline:** `mcp_server_week05_ly.py` (MCP server with `query_simbad` and `search_arxiv`) + `week05_ly.ipynb` (Claude agent loop, MCP client)

I triggered four failure modes deliberately, one in each layer of the pipeline. For each case I record the triggering input, the error or incorrect output, whether the agentic loop recovered by itself or needed manual intervention, and how I would fix it in a production system. The **Recovery (observed)** blocks between the `<!-- BEGIN AUTO … -->` markers are written by Section 6 of the notebook from the logged run (`agent_runs/failure_*.json` and the main runs). They therefore always describe the run that is actually logged.

| Case | Failure mode | Layer | Visible error? |
|---|---|---|---|
| F1 | Misspelled object name | tool input (entity resolution) | yes, `isError` from `query_simbad` |
| F2 | SIMBAD unavailable | upstream service | yes, `isError` from `query_simbad` |
| F3 | Turn budget exhausted | agent control loop | yes, no final answer |
| F4 | "Most recent" papers are not relevant papers | retrieval semantics | **no**, silent |

---

## F1 — Misspelled object name

### Triggering input

Agent question (run `failure_F1_misspelled_name`):

```
Query SIMBAD for the object 'Betelguese' and report its spectral type and parallax.
```

At the protocol level this becomes `tools/call query_simbad {"object_name": "Betelguese"}`.

### Error / incorrect output

SIMBAD's identifier matching ignores case and spacing but does no fuzzy matching. The ADQL query `… WHERE ident.id = 'Betelguese'` returns zero rows (I checked this against the SIMBAD TAP service on 2026-10-13), so the tool raises a `ToolError` and the client receives:

```
isError: true
Error executing tool query_simbad: No SIMBAD object matches the identifier 'Betelguese'. Check the spelling,
or try another designation (common name, Bayer/Flamsteed name, HD or HIP number).
```

The server does not crash. The error text goes back to the model as a `tool_result` block with `is_error: true`, and whether the agent recovers is then up to the model.

### Recovery (observed)

<!-- BEGIN AUTO F1 -->
_Not yet observed: run Section 6 of `week05_ly.ipynb` to fill in this block._
<!-- END AUTO F1 -->

### Production fix

On a miss, the server should query a name resolver with fuzzy matching (e.g. CDS Sesame) and return the candidates ("did you mean `Betelgeuse` = `* alf Ori`?") inside the error, so that the corrected identifier is confirmed explicitly and reported to the user instead of being guessed silently.

---

## F2 — Upstream service unavailable (SIMBAD down)

### Triggering input

The full Betelgeuse question from the assignment (run `failure_F2_simbad_unreachable`), with the server subprocess started with the environment variable `SIMBAD_TIMEOUT_S=0.001`. Every SIMBAD request is then abandoned after 1 ms, so from the agent's point of view SIMBAD behaves like a service that is down or overloaded. arXiv is unaffected.

### Error / incorrect output

Every `query_simbad` call returns:

```
isError: true
Error executing tool query_simbad: SIMBAD did not respond within 0.001 s; the service may be down or overloaded.
Try again later.
```

Without the timeout wrapper in the server, a hung TAP request would have no upper bound on its duration: neither the HTTP request nor the MCP client call has a read timeout by default, so the whole agent would freeze. With the wrapper, the failure is fast and explicit. The real risk lies in what the model does next. The question still asks for a spectral type and a parallax, and the model knows Betelgeuse well enough to supply both from memory, which would make them look as if they had been retrieved.

### Recovery (observed)

<!-- BEGIN AUTO F2 -->
_Not yet observed: run Section 6 of `week05_ly.ipynb` to fill in this block._
<!-- END AUTO F2 -->

### Production fix

Retry with exponential backoff and fail over to the second official SIMBAD mirror (`simbad.harvard.edu`), and if the data are still unavailable, have the orchestrator reject any final answer that quotes catalogue numbers not traceable to a successful tool result.

---

## F3 — Agent turn budget exhausted

### Triggering input

The full Betelgeuse question with a budget of one model call, `get_run(..., max_turns=1)` (run `failure_F3_turn_budget`).

### Error / incorrect output

The first model reply is expected to request tools (`stop_reason == "tool_use"`), because the model has no data yet. The loop executes the requested calls, but the budget is used up before the model can read the results, so `run_agent` ends with

```
status: gave up: all 1 allowed model call(s) used without a final answer
final_answer: None
```

All the data needed for the answer has been fetched, yet the user receives nothing. The default budget of 8 model calls is enough for the single-object questions in the notebook. A long multi-object question, or a model that retries failing calls many times (as in F1/F2), could still exhaust it.

### Recovery (observed)

<!-- BEGIN AUTO F3 -->
_Not yet observed: run Section 6 of `week05_ly.ipynb` to fill in this block._
<!-- END AUTO F3 -->

### Production fix

When the budget runs out, make one last model call with tools disabled (`tool_choice = none`) that must answer from the results gathered so far and say what is missing, and scale the budget with the number of sub-questions instead of using a fixed constant.

---

## F4 — Silent retrieval failure: "most recent" is not "most relevant"

### Triggering input

The assignment's own question, "…what are the 3 most recent arXiv papers whose abstracts mention Betelgeuse? Summarize what the papers suggest about its current evolutionary status." (and the same question for Proxima Centauri). `search_arxiv` translates it faithfully into the arXiv query `abs:"Betelgeuse"` sorted by submission date.

### Error / incorrect output

No error is raised: the output is wrong in meaning, not in form. On 2026-10-13 the newest abstract that mentions Betelgeuse was arXiv:2609.27055, a physics-education paper about measuring the angular diameter of **Venus** with a two-hole masked telescope. It mentions Betelgeuse only as the target of Michelson & Pease's 1921 interferometric measurement. For Proxima Centauri, two of the three newest matches mention the star in a single sentence: arXiv:2609.05599 (a spectral-sampling method validated on synthetic spectra) and arXiv:2608.10212 (the GMagAO-X instrument design, which names Proxima b as a science case). Apart from arXiv:2608.19339 calling Betelgeuse a red supergiant, none of the six abstracts discusses evolutionary status. Anything more specific in an "evolutionary status" summary is therefore built on the model's background knowledge, presented as if it came from "the papers". Only the post-hoc comparison of the summary with the abstracts in Section 5.3 of the notebook exposes this.

### Recovery (observed)

<!-- BEGIN AUTO F4 -->
_Not yet observed: run Section 6 of `week05_ly.ipynb` to fill in this block._
<!-- END AUTO F4 -->

### Production fix

Rank by relevance as well as date: fetch the 25 newest matches, re-rank them by embedding similarity to the question (as in the Week 4 retriever) or require the object in the title or several mentions in the abstract, and return a relevance score per paper so that the model can state that no recent paper addresses the question.
