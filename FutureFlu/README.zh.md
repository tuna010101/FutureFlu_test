# FutureFlu

[English](README.md)

A/H1N1pdm09、A/H3N2、B/Victoria 的 HA1 适应性分析（2013–2024），含 HA/NA 与
H3 DMS 汇总实验。

## 目录

- `scripts/` — 分析入口
- `data/` — clade 排序、计数等输入
- `outputs/predictions/` — 预测结果表
- `outputs/ha1_components/` — 组分汇总
- `experiments/` — HA/NA 与 H3 DMS 实验
- `raw_inputs/` — 完整重跑时放置序列/元数据

## 运行

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py aux
python scripts/analyze_components.py
```

`aux`：基于包内输入计算 clade 组分准确率并做 E/G/D 组合，写出
`EGD_temperatures_Twindow.csv` 等表到 `outputs/predictions/risk_components/`。
`analyze_components.py` 对这些结果做汇总。
