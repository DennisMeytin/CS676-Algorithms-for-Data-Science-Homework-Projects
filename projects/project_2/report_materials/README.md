# report_materials: the evidence behind the report

Every live session in the report was run by piping a script of typed requests into the real
application (`uv run main.py --model <model> --persona-dir <folder> < script.txt`). The file
names below say which script and model produced which output.

| Files | What they are |
|---|---|
| `live_run1_*` | The first live script on Haiku 5.5 (script: `live_run1_script.txt`), as re-run after the Windows encoding fix; the very first attempt crashed on that bug and its output was not kept. This run produced the malformed persona (`live_run1_malformed_persona.md`) described in Section 5. |
| `live_run2_*` | Edit, undo and a refused rename on Haiku 5.5 (script: `live_run2_script.txt`). |
| `compare_*` | The same nine-line script (`live_script.txt`) on Haiku 5.5, Sonnet 5.5 and Opus 5.5: the `_session.txt` file is the full terminal output and the `_conversation.md` file is the saved transcript. Source of the cost table. |
| `holdprompt_*` | Two Haiku 5.5 runs of `live_script.txt` with the "hold your position" rule that was later removed. |
| `final_*` | The submitted session on Opus 5.5 (script: `final_script.txt`): the full output, the conversation transcript, the patient, and the doctor before and after the edit. |
| `regression_*` | A last Haiku 5.5 session on the finished code (script: `regression_script.txt`): 22 calls, about $0.007. |

Notes:

- `live_run1_script.txt` and `live_run2_script.txt` were written down after the fact from the exact
  requests typed on those two runs, which were piped in directly and not saved as files at the time.
  The other scripts were saved before they were run.
- The agreement experiment is in `../evals/` (script `hold_position_eval.py`, results in `results/`),
  not here.
- Costs are estimates from the token counts the API returned and the published per-million-token
  rates in `llm.py`.
