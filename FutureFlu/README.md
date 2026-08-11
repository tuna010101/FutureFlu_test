# FutureFlu

[简体中文](README.zh.md)

HA1 fitness for A/H1N1pdm09, A/H3N2, and B/Victoria over seasons 2013–2024,
with HA/NA and H3 DMS summary experiments.

## Layout

- `scripts/` — analysis entry points
- `data/` — clade ranks, counts, and related inputs
- `outputs/predictions/` — prediction tables
- `outputs/ha1_components/` — component summaries
- `experiments/` — HA/NA and H3 DMS experiments
- `raw_inputs/` — place sequence/metadata here for a full rerun

## Run

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py aux
python scripts/analyze_components.py
```

`aux` runs clade-component accuracy and E/G/D combination on packaged inputs,
writing tables such as `EGD_temperatures_Twindow.csv` under
`outputs/predictions/risk_components/`. `analyze_components.py` summarizes
those results.
