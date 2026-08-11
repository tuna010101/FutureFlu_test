# flu_forecasting_FutureFlu

[简体中文](README.zh.md)

Issue-date forecasting for A/H1N1pdm09, A/H3N2, and B/Victoria.

## Dependency

Clone [blab/flu-forecasting](https://github.com/blab/flu-forecasting) and set
`FLU_FORECASTING_ROOT`:

```bash
git clone https://github.com/blab/flu-forecasting.git
export FLU_FORECASTING_ROOT="$PWD/flu-forecasting"
pip install -r ../requirements.txt
```

TreeTime, Augur, and IQ-TREE come from the flu-forecasting environment.

## Layout

- `config/futureflu/` — lineage and distance-map settings
- `data/run_inputs/` — issue-date schedules (`H1N1`, `H3N2`, `Victoria`)
- `data/sequences/` — restore lineage metadata/FASTA tables for a full rerun
- `data/sources/` — optional primary tables for `derive`
- `results/futureflu/runs/` — per-lineage forecasts
- `results/futureflu/recommended_clades/primary/` — recommended clade CSV

## Run

```bash
python scripts/run_futureflu_workflow.py all --lineage all
```

Examples:

```bash
python scripts/run_futureflu_workflow.py all --lineage h3n2
python scripts/run_futureflu_workflow.py model --lineage victoria
python scripts/run_futureflu_workflow.py export
```

Steps: `derive` → `prepare` → `timepoints` → `aggregate` → `model` → `top1` → `export`.
Lineage keys: `h1n1pdm`, `h3n2`, `victoria` (`config/futureflu/lineages.yaml`).
H1N1pdm uses Canton-style epitope maps under `config/futureflu/distance_maps/h1n1pdm/ha/`.
