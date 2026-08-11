# flu_forecasting_FutureFlu

[English](README.md)

A/H1N1pdm09、A/H3N2、B/Victoria 的 issue-date 预测流程。

## 依赖

请 clone [blab/flu-forecasting](https://github.com/blab/flu-forecasting) 并设置
`FLU_FORECASTING_ROOT`：

```bash
git clone https://github.com/blab/flu-forecasting.git
export FLU_FORECASTING_ROOT="$PWD/flu-forecasting"
pip install -r ../requirements.txt
```

TreeTime、Augur、IQ-TREE 使用 flu-forecasting 环境。

## 目录

- `config/futureflu/` — 谱系与距离图配置
- `data/run_inputs/` — issue-date 日程（`H1N1`、`H3N2`、`Victoria`）
- `data/sequences/` — 完整重跑时恢复谱系 metadata/FASTA 表
- `data/sources/` — 可选；`derive` 用的原始表
- `results/futureflu/runs/` — 各谱系 forecast
- `results/futureflu/recommended_clades/primary/` — 推荐 clade CSV

## 运行

```bash
python scripts/run_futureflu_workflow.py all --lineage all
```

示例：

```bash
python scripts/run_futureflu_workflow.py all --lineage h3n2
python scripts/run_futureflu_workflow.py model --lineage victoria
python scripts/run_futureflu_workflow.py export
```

步骤：`derive` → `prepare` → `timepoints` → `aggregate` → `model` → `top1` → `export`。
谱系键：`h1n1pdm`、`h3n2`、`victoria`（见 `config/futureflu/lineages.yaml`）。
H1N1pdm 使用 `config/futureflu/distance_maps/h1n1pdm/ha/` 下的 Canton 风格表位图。
