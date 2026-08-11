# FutureFlu-norovirus

[简体中文](README.zh.md)

Norovirus GII.4 fitness analysis using the FutureFlu framework.

## Layout

- `scripts/` — analysis entry points
- `data/` — clade ranks, counts, positivity rates, EVEscape scores
- `outputs/predictions/` — yearly prediction and risk-component tables
- `raw_inputs/` — for a full rerun, place `norovirus_vp1_sequence.csv`
  (columns: `Isolate_Name`, `Collection_Date`, `year`, `genotype`, `X1..Xn`)

## Run

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py all
```

Results are under `outputs/predictions/`.

## References

Yearly positivity rates in
`data/positivity/zhang2024_yearly_norovirus_positive_rates.csv` are from
Table 2 of Zhang, Pan, et al., *Frontiers in Public Health* 12 (2024): 1373322.
https://doi.org/10.3389/fpubh.2024.1373322
