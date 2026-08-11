# Beth1

[English](README.md)

[Beth-1](https://github.com/mwanglab/beth-1) 的 HA-only 复现（overall-fit-once、无气候项；
非上游完整 HA+NA 流程）。

```bash
pip install -r ../requirements.txt
```

完整重跑时，将 HA FASTA 与 metadata 放入 `data/dataset/fasta/` 与
`data/dataset/metadata/`（文件名见 `work/ha_overall_fit/run_beth1_ha.py`），然后：

```bash
python scripts/run_pipeline.py --processes 6
```

包内输入：`data/truth.csv`、epitope 文件与疫情阳性率表。
结果：`work/ha_overall_fit/results/`（如 `beth1_ha_clade_accuracy.csv`）。
