# Beth1

[简体中文](README.zh.md)

HA-only reproduction of [Beth-1](https://github.com/mwanglab/beth-1)
(overall-fit-once, no climate covariates; not the full upstream HA+NA method).

```bash
pip install -r ../requirements.txt
```

For a full rerun, place HA FASTA and metadata under `data/dataset/fasta/` and
`data/dataset/metadata/` (filenames in `work/ha_overall_fit/run_beth1_ha.py`),
then:

```bash
python scripts/run_pipeline.py --processes 6
```

Packaged inputs: `data/truth.csv`, epitope files, and epidemic positivity tables.
Results: `work/ha_overall_fit/results/` (e.g. `beth1_ha_clade_accuracy.csv`).
