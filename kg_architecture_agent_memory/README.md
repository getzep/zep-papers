# Zep: A Temporal Knowledge Graph Architecture for Agent Memory Experiments
For this paper, experiments are contained in two notebooks:
* One for the DMR experiment first presented in the MemGPT paper
* A second for the LongMemEval experiment

The notebooks will walk through downloading the datasets and building the necessary objects

OpenAI and Zep keys are required to run both experiments.

The notebook cells will first walk through ingesting data into Zep, and the evaluating Zep.
In the case of LongMemEval, results are saved in a JSON file.


Both notebooks also contain a cell to run the experiment against a baseline.
Models will default to gpt-4o-mini but can be updated in code.

## Connecting to Zep

Create a project and API key at https://app.getzep.com/ and set `ZEP_API_KEY` in the environment or a
`.env` file. Do not set `ZEP_API_URL` for Cloud. Use it only for self-hosted / BYOC, as the
deployment root (the SDK appends `/api/v2`).

The DMR notebook (`zep_memgpt_eval.ipynb`) needs `zep-cloud>=3.0.0`. It defaults to ingesting one
user (`msc_experiment_user_0`). After episodes finish processing, run `await dmr_eval_session(zep, 0)`.
Set `MSC_INGEST_COUNT=500` to reproduce the paper. Place `msc.jsonl` at `locomo_eval/data/msc.jsonl`.

LongMemEval and LoCoMo in this directory still call `zep.memory` / `graph.add(group_id=...)` and have
not been migrated to the v3 thread API.
