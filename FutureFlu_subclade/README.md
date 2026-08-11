# FutureFlu_subclade

[简体中文](README.zh.md)

Subclade-level extension of FutureFlu for A/H1N1pdm09, A/H3N2, and B/Victoria
in seasons 2025–2026 (north and south).

## Layout

- `scripts/` — analysis entry points
- `data/` — subclade definitions, counts, configs, positivity, EVEscape
- `outputs/predictions/` — prediction tables
- `experiments/` — sensitivity analyses
- `raw_inputs/` — place private inputs here for a full rerun

## Run

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py definitions
python scripts/run_pipeline.py counts
python scripts/run_pipeline.py aux
```

`definitions` / `counts` refresh subclade labels and counts.
`aux` scores subclade risk components against truth and builds E/G/D combination
tables under `outputs/predictions/`.
