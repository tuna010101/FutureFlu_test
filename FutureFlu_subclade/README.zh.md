# FutureFlu_subclade

[English](README.md)

FutureFlu 在亚分支（subclade）分辨率上的延伸：覆盖 A/H1N1pdm09、A/H3N2、
B/Victoria 的 2025–2026 南北半球季节。

## 目录

- `scripts/` — 分析入口
- `data/` — 亚分支定义、计数、配置、阳性率、EVEscape
- `outputs/predictions/` — 预测结果表
- `experiments/` — 敏感性分析
- `raw_inputs/` — 完整重跑时放置输入

## 运行

```bash
pip install -r ../requirements.txt
python scripts/run_pipeline.py definitions
python scripts/run_pipeline.py counts
python scripts/run_pipeline.py aux
```

`definitions` / `counts` 更新亚分支标签与计数。
`aux`：用真值评估亚分支风险组分，并生成 E/G/D 组合表，结果在 `outputs/predictions/`。
