# FutureFlu-norovirus

[English](README.md)

基于 FutureFlu 框架的诺如病毒 GII.4 适应性分析。

## 目录

- `scripts/` — 分析入口
- `data/` — clade 标签、计数、阳性率、EVEscape 打分
- `outputs/predictions/` — 逐年预测与风险组分表
- `raw_inputs/` — 完整重跑时放入 `norovirus_vp1_sequence.csv`
  （列：`Isolate_Name`、`Collection_Date`、`year`、`genotype`、`X1..Xn`）

## 运行

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py all
```

结果在 `outputs/predictions/`。

## 参考文献

`data/positivity/zhang2024_yearly_norovirus_positive_rates.csv` 中的逐年阳性率
取自 Zhang, Pan, et al., *Frontiers in Public Health* 12 (2024): 1373322 的
Table 2。https://doi.org/10.3389/fpubh.2024.1373322
