# FutureFlu

[简体中文](README.zh.md)

Seasonal influenza fitness and forecasting analyses, with companion packages for
Beth-1 reproduction and norovirus GII.4.

## Packages

| Directory | Description |
|-----------|-------------|
| `FutureFlu/` | HA1 fitness for A/H1N1pdm09, A/H3N2, and B/Victoria, seasons **2013–2024** |
| `FutureFlu_subclade/` | Same lineages at **subclade** resolution, seasons **2025–2026** |
| `flu_forecasting_FutureFlu/` | Reproduction of the [blab/flu-forecasting](https://github.com/blab/flu-forecasting) method |
| `Beth1/` | [Beth-1](https://github.com/mwanglab/beth-1) HA overall-fit reproduction |
| `FutureFlu-norovirus/` | FutureFlu workflow applied to norovirus GII.4 |

## Setup

```bash
pip install -r requirements.txt
```

`flu_forecasting_FutureFlu/` also needs
[blab/flu-forecasting](https://github.com/blab/flu-forecasting):

```bash
git clone https://github.com/blab/flu-forecasting.git
export FLU_FORECASTING_ROOT="$PWD/flu-forecasting"
```

## Acknowledgements

GISAID contributor tables:
[`acknowledgements/`](acknowledgements/) and
[`GISAID_ACKNOWLEDGEMENTS.md`](GISAID_ACKNOWLEDGEMENTS.md).

Influenza analyses used sequences from the GISAID EpiFlu database. The
`isolate_id` column (`EPI_ISL_…`) in those tables identifies the isolates that
were used.
