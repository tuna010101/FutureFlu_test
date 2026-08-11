# FutureFlu

[English](README.md)

季节性流感适应性与预测分析，并包含 Beth-1 复现与诺如病毒 GII.4 相关包。

## 目录

| 目录 | 说明 |
|------|------|
| `FutureFlu/` | A/H1N1pdm09、A/H3N2、B/Victoria 的 HA1 适应性分析（**2013–2024**） |
| `FutureFlu_subclade/` | 同上谱系的亚分支预测（**2025–2026**） |
| `flu_forecasting_FutureFlu/` | [blab/flu-forecasting](https://github.com/blab/flu-forecasting) 方法复现 |
| `Beth1/` | [Beth-1](https://github.com/mwanglab/beth-1) HA overall-fit 复现 |
| `FutureFlu-norovirus/` | FutureFlu 流程在诺如病毒 GII.4 上的应用 |

## 安装

```bash
pip install -r requirements.txt
```

`flu_forecasting_FutureFlu/` 另需
[blab/flu-forecasting](https://github.com/blab/flu-forecasting)：

```bash
git clone https://github.com/blab/flu-forecasting.git
export FLU_FORECASTING_ROOT="$PWD/flu-forecasting"
```

## 致谢

GISAID 贡献者表见
[`acknowledgements/`](acknowledgements/) 与
[`GISAID_ACKNOWLEDGEMENTS.md`](GISAID_ACKNOWLEDGEMENTS.md)。

流感分析所用序列来自 GISAID EpiFlu 数据库。致谢表中的 `isolate_id` 列
（`EPI_ISL_…`）标识实际使用的分离株。
