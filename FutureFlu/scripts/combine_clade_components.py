"""Evaluate E/G/D component combinations with rolling temperatures.

English: Evaluate E/G/D component combinations, optimize rolling temperatures,
and write accuracy, frequency, ELPD, AIC, and provenance tables.
中文：评估 E/G/D 组件组合、优化滚动温度，并输出准确率、频率、ELPD、AIC 和来源表。

评估 total_escape (E)、predicted_prevalence (G)、mutual_information (D)
单指标、两两及三者组合对 TOP1 clade 的预测准确率，含滚动温度搜索。

额外输出（E+G+D 组合）：
  - EGD_combine_fit.csv  : subtype, hemisphere, year, clade, fit_E+G+D
  - EGD_temperatures.csv : subtype, hemisphere, year, tem_E, tem_G, tem_D,
                           平均L1损失, TOP1_clade, T/F

新增输出（所有组合，含单指标 E / G / D）：
  - *_ELPD.csv : subtype, hemisphere, year,
                 ELPD_E, ELPD_G, ELPD_D, ELPD_E+G, ELPD_E+D, ELPD_G+D, ELPD_E+G+D
                 （末尾各亚型汇总行：hemisphere='Summary', year='All'，值为该亚型所有季度 ELPD 之和）
  - *_AIC.csv  : subtype, AIC_E, AIC_G, AIC_D, AIC_E+G, AIC_E+D, AIC_G+D, AIC_E+G+D
                 AIC = 2k − 2 × Σ ELPD（k 为该组合的温度参数数量）
  - pre_act_freq.csv : subtype, year, hemisphere, clade, act_freq, freq_prev,
                       E_pre_fit, E_pre_freq, G_pre_fit, G_pre_freq,
                       D_pre_fit, D_pre_freq, E+G_pre_fit, E+G_pre_freq,
                       E+D_pre_fit, E+D_pre_freq, G+D_pre_fit, G+D_pre_freq,
                       E+G+D_pre_fit, E+G+D_pre_freq
    act_freq   : 当季各 clade 的归一化实际频率（基于 collection_count）
    freq_prev  : 上一季各 clade 的实际频率（基于 submission_count）
    *_pre_fit  : fit 方法 — softmax(fitness) 给出的预测频率分布
    *_pre_freq : freq 方法 — xi_prev × exp(fitness) 归一化后的预测频率分布
  - divergence_inform.csv / escape_inform.csv / growth_inform.csv

说明：
  - 各亚型 min_seq_count：H3N2=10，H1N1/Victoria=0（内置，无命令行参数）
  - 历史窗口：不限制，始终使用全部历史季度
  - 温度优化与预测计算（xi_prev、L1 损失）仍使用 submission_count
  - 输出表格中仅 act_freq 列改用 collection_count 计算，freq_prev 仍使用 submission_count

温度选取规则（跨半球全局滚动）：
  - 起始流感季（全局第一季）：所有指标 T = 1
  - 后续流感季：在全部历史流感季中搜索综合平均 L1 损失最小的 T 组合
  - 平均 L1 损失 = 历史窗口内各季 L1 损失之和 / 参与训练的流感季数量
  - 同一年内顺序：南半球 → 北半球

用法：
    python scripts/combine_clade_components.py \
        <acc_output> <egdfit_output> <egdtemp_output> \
        [--elpd_output FILENAME] [--aic_output FILENAME] \
        [--pre_act_output FILENAME] \
        [--divergence_inform FILENAME] [--escape_inform FILENAME] \
        [--growth_inform FILENAME]

示例：
    python scripts/combine_clade_components.py \
        clade_component_combine_acc_all.csv \
        EGD_combine_fit.csv \
        EGD_temperatures.csv \
        --divergence_inform divergence_inform.csv \
        --escape_inform escape_inform.csv \
        --growth_inform growth_inform.csv
"""

import pandas as pd
import numpy as np
import re
import argparse
import os
from pathlib import Path
from itertools import product

# ═════════════════════════════════════════════════════════════
# English: Parse output arguments; min_seq_count and max_history are built in.
# 中文：解析输出参数；min_seq_count 和 max_history 不再作为命令行参数。
# ═════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description='Clade combination accuracy (E/G/D) with temperature optimisation.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('acc_output',     type=str,
                    help='准确率输出文件名（例如：clade_component_combine_acc_all.csv）')
parser.add_argument('egdfit_output',  type=str,
                    help='EGD clade fitness 输出文件名（例如：EGD_combine_fit.csv）')
parser.add_argument('egdtemp_output', type=str,
                    help='EGD 温度记录输出文件名（例如：EGD_temperatures.csv）')
parser.add_argument('--elpd_output',  type=str, default=None,
                    help='ELPD 输出文件名（默认：acc_output 替换 .csv 为 _ELPD.csv）')
parser.add_argument('--aic_output',   type=str, default=None,
                    help='AIC  输出文件名（默认：acc_output 替换 .csv 为 _AIC.csv）')
parser.add_argument('--pre_act_output', type=str, default='pre_act_freq.csv',
                    help='各 clade 预测频率与实际频率输出文件名（默认：pre_act_freq.csv）')
parser.add_argument('--divergence_inform', type=str, default='divergence_inform.csv',
                    help='D 组件(mutual_information) argmax 来源行信息输出文件名')
parser.add_argument('--escape_inform',     type=str, default='escape_inform.csv',
                    help='E 组件(total_escape) argmax 来源行信息输出文件名')
parser.add_argument('--growth_inform',     type=str, default='growth_inform.csv',
                    help='G 组件(predicted_prevalence) argmax 来源行信息输出文件名')
args = parser.parse_args()

ACC_FILENAME     = args.acc_output
EGDFIT_FILENAME  = args.egdfit_output
EGDTEMP_FILENAME = args.egdtemp_output

ELPD_FILENAME = (args.elpd_output if args.elpd_output is not None
                 else ACC_FILENAME.replace('.csv', '_ELPD.csv'))
AIC_FILENAME  = (args.aic_output  if args.aic_output  is not None
                 else ACC_FILENAME.replace('.csv', '_AIC.csv'))
PRE_ACT_FILENAME    = args.pre_act_output
DIVERGENCE_FILENAME = args.divergence_inform
ESCAPE_FILENAME     = args.escape_inform
GROWTH_FILENAME     = args.growth_inform

# ═════════════════════════════════════════════════════════════
# English: Built-in subtype thresholds: H3N2=10, all others=0.
# 中文：亚型级 min_seq_count 内置配置：H3N2=10，其余亚型=0。
# ═════════════════════════════════════════════════════════════
_MIN_SEQ_COUNT_MAP: dict = {'H3N2': 10}


def get_min_seq_count(subtype: str) -> int:
    """Return the subtype threshold. / 根据亚型名称返回 min_seq_count 阈值。"""
    s = str(subtype).upper()
    for key, val in _MIN_SEQ_COUNT_MAP.items():
        if key.upper() in s:
            return val
    return 0


# ═════════════════════════════════════════════════════════════
# English: Resolve all inputs and outputs relative to this package root.
# 中文：所有输入输出均相对本发布包根目录解析。
# ═════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    PROJECT_ROOT
    / 'outputs'
    / 'predictions'
    / 'risk_components'
)
COMPONENT_PATH = OUT_DIR / 'risk_mutation_group_component.csv'
LABEL_PATH = PROJECT_ROOT / 'data' / 'futureflu_rank' / 'circulating_clade.csv'
COUNT_DIR = PROJECT_ROOT / 'data' / 'clade_counts'

OUT_ACC_PATH = OUT_DIR / ACC_FILENAME
OUT_EGD_FIT_PATH = OUT_DIR / EGDFIT_FILENAME
OUT_EGD_TEMP_PATH = OUT_DIR / EGDTEMP_FILENAME
OUT_ELPD_PATH = OUT_DIR / ELPD_FILENAME
OUT_AIC_PATH = OUT_DIR / AIC_FILENAME
OUT_PRE_ACT_PATH = OUT_DIR / PRE_ACT_FILENAME
OUT_DIVERGENCE_PATH = OUT_DIR / DIVERGENCE_FILENAME
OUT_ESCAPE_PATH = OUT_DIR / ESCAPE_FILENAME
OUT_GROWTH_PATH = OUT_DIR / GROWTH_FILENAME

print(f"[Config] H1N1/Victoria min_seq_count : 0")
print(f"[Config] H3N2          min_seq_count : 10")
print(f"[Config] accuracy output file        : {OUT_ACC_PATH}")
print(f"[Config] EGD fit output file         : {OUT_EGD_FIT_PATH}")
print(f"[Config] EGD temperature output      : {OUT_EGD_TEMP_PATH}")
print(f"[Config] ELPD output file            : {OUT_ELPD_PATH}")
print(f"[Config] AIC  output file            : {OUT_AIC_PATH}")
print(f"[Config] pre_act output file         : {OUT_PRE_ACT_PATH}")
print(f"[Config] divergence_inform output    : {OUT_DIVERGENCE_PATH}")
print(f"[Config] escape_inform output        : {OUT_ESCAPE_PATH}")
print(f"[Config] growth_inform output        : {OUT_GROWTH_PATH}")
print(f"[Config] max_history                 : unlimited")

# ═════════════════════════════════════════════════════════════
# 全局常量
# ═════════════════════════════════════════════════════════════
METRICS_3 = ['total_escape', 'predicted_prevalence', 'mutual_information']

COMBINATIONS = [
    ('E',     ['total_escape']),
    ('G',     ['predicted_prevalence']),
    ('D',     ['mutual_information']),
    ('E+G',   ['total_escape', 'predicted_prevalence']),
    ('E+D',   ['total_escape', 'mutual_information']),
    ('G+D',   ['predicted_prevalence', 'mutual_information']),
    ('E+G+D', ['total_escape', 'predicted_prevalence', 'mutual_information']),
]

N_PARAMS: dict = {combo: len(metrics) for combo, metrics in COMBINATIONS}

COMBO_NAMES_ORDERED: list = ['E', 'G', 'D', 'E+G', 'E+D', 'G+D', 'E+G+D']

PRE_ACT_COL_ORDER: list = (
    ['subtype', 'year', 'hemisphere', 'clade', 'act_freq', 'freq_prev']
    + [col
       for c in COMBO_NAMES_ORDERED
       for col in (f'{c}_pre_fit', f'{c}_pre_freq')]
)

_INFORM_BASE_COLS: list = [
    'subtype', 'hemisphere', 'year', 'clade',
    'risk_mutation_group', 'mutation_count', 'mutation_group_seq_count',
]
DIVERGENCE_COL_ORDER: list = _INFORM_BASE_COLS + ['mutual_information']
ESCAPE_COL_ORDER:     list = _INFORM_BASE_COLS + ['total_escape']
GROWTH_COL_ORDER:     list = _INFORM_BASE_COLS + ['predicted_prevalence']

T_VALUES: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ═════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════
_CLADE_RE = re.compile(r'([^\(,]+?)\s*\((\d+\.?\d*)%\)')


def parse_clade_string(clade_str) -> list:
    if pd.isna(clade_str) or str(clade_str).strip().lower() in ('unknown', ''):
        return []
    results = []
    for name, pct_str in _CLADE_RE.findall(str(clade_str)):
        name = name.strip()
        if name.lower() not in ('unassigned', 'unknown'):
            results.append((name, float(pct_str) / 100.0))
    return results


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def z_score_arr(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    # English: A one-value or all-missing group has no sample variance.
    # 中文：单值组或全缺失组没有样本方差，直接返回零向量。
    valid = np.isfinite(vals)
    if valid.sum() < 2:
        return np.zeros_like(vals)
    mu   = vals[valid].mean()
    std  = vals[valid].std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return np.zeros_like(vals)
    result = np.zeros_like(vals)
    result[valid] = (vals[valid] - mu) / std
    return result


# ═════════════════════════════════════════════════════════════
# 最大法（同步追踪各指标 argmax 的来源行信息，含指标原始值）
# ═════════════════════════════════════════════════════════════
def compute_max_method_with_info(df: pd.DataFrame) -> tuple:
    GROUP_KEYS = ['subtype', 'hemisphere', 'year', 'clade_single']

    rows = []
    for _, row in df.iterrows():
        parsed = parse_clade_string(row['clade'])
        if not parsed:
            continue
        dominant = max(parsed, key=lambda x: x[1])[0]
        rec = {
            'subtype':                  row['subtype'],
            'hemisphere':               row['hemisphere'],
            'year':                     row['year'],
            'clade_single':             dominant,
            'risk_mutation_group':      row.get('risk_mutation_group',      np.nan),
            'mutation_count':           row.get('mutation_count',           np.nan),
            'mutation_group_seq_count': row.get('mutation_group_seq_count', np.nan),
        }
        for m in METRICS_3:
            rec[f'fit_{m}'] = row.get(m, np.nan)
        rows.append(rec)

    empty_max = pd.DataFrame(columns=GROUP_KEYS + [f'fit_{m}' for m in METRICS_3])
    empty_div = pd.DataFrame(columns=DIVERGENCE_COL_ORDER)
    empty_esc = pd.DataFrame(columns=ESCAPE_COL_ORDER)
    empty_gro = pd.DataFrame(columns=GROWTH_COL_ORDER)

    if not rows:
        return empty_max, empty_div, empty_esc, empty_gro

    tmp = pd.DataFrame(rows)

    max_df = (tmp.groupby(GROUP_KEYS, as_index=False)
                 .agg({f'fit_{m}': 'max' for m in METRICS_3}))

    metric_info_map = [
        ('fit_mutual_information',   'mutual_information',   'divergence'),
        ('fit_total_escape',         'total_escape',         'escape'),
        ('fit_predicted_prevalence', 'predicted_prevalence', 'growth'),
    ]

    inform_dfs: dict = {}
    for fit_col, metric_col_name, out_key in metric_info_map:
        idx_max = (tmp.groupby(GROUP_KEYS, sort=False)[fit_col]
                      .idxmax()
                      .dropna()
                      .astype(int))

        select_cols = GROUP_KEYS + [
            'risk_mutation_group',
            'mutation_count',
            'mutation_group_seq_count',
            fit_col,
        ]
        info_df = (tmp.loc[idx_max.values, select_cols]
                      .copy()
                      .rename(columns={
                          'clade_single': 'clade',
                          fit_col:        metric_col_name,
                      })
                      .reset_index(drop=True))
        inform_dfs[out_key] = info_df

    return (
        max_df,
        inform_dfs['divergence'],
        inform_dfs['escape'],
        inform_dfs['growth'],
    )


# ═════════════════════════════════════════════════════════════
# 读取 xi(t) — 基于 submission_count（用于温度优化、预测计算及 freq_prev 输出）
# ═════════════════════════════════════════════════════════════
def load_clade_freq(subtype_list: list) -> pd.DataFrame:
    frames = []
    for subtype in subtype_list:
        key  = subtype.lower().replace('/', '').replace(' ', '')
        path = os.path.join(COUNT_DIR,
                            f'submission_collection_clade_count_{key}.csv')
        if not os.path.exists(path):
            print(f"  [WARN] count file not found: {path} "
                  f"→ freq columns for {subtype} will be NaN")
            continue
        cnt = pd.read_csv(path)
        cnt['hemisphere'] = cnt['hemisphere'].str.lower()
        total = (cnt.groupby(['year', 'hemisphere'])['submission_count']
                    .sum().rename('total_sc').reset_index())
        cnt = cnt.merge(total, on=['year', 'hemisphere'])
        cnt['xi_t'] = np.where(
            cnt['total_sc'] > 0,
            cnt['submission_count'] / cnt['total_sc'],
            np.nan
        )
        cnt['subtype'] = subtype
        frames.append(cnt[['subtype', 'year', 'hemisphere', 'clade', 'xi_t']])
        print(f"  [FreqData] {subtype}: {len(cnt)} clade-records loaded from {path}")

    if not frames:
        print("  [WARN] No clade count file loaded.")
        return pd.DataFrame(columns=['subtype', 'year', 'hemisphere', 'clade', 'xi_t'])
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════
# 读取 xi(t) — 基于 collection_count
#   仅用于输出表格中的 act_freq 列
# ═════════════════════════════════════════════════════════════
def load_clade_freq_collection(subtype_list: list) -> pd.DataFrame:
    """
    与 load_clade_freq 读取相同的源文件，
    但使用 collection_count 列计算频率，
    结果仅用于写入 pre_act_freq.csv 中的 act_freq 列。
    """
    frames = []
    for subtype in subtype_list:
        key  = subtype.lower().replace('/', '').replace(' ', '')
        path = os.path.join(COUNT_DIR,
                            f'submission_collection_clade_count_{key}.csv')
        if not os.path.exists(path):
            print(f"  [WARN] count file not found: {path} "
                  f"→ collection freq for {subtype} will be NaN")
            continue
        cnt = pd.read_csv(path)
        cnt['hemisphere'] = cnt['hemisphere'].str.lower()

        if 'collection_count' not in cnt.columns:
            print(f"  [WARN] 'collection_count' column not found in {path}; "
                  f"act_freq for {subtype} will be NaN")
            dummy = cnt[['year', 'hemisphere', 'clade']].copy()
            dummy['xi_t']    = np.nan
            dummy['subtype'] = subtype
            frames.append(dummy[['subtype', 'year', 'hemisphere', 'clade', 'xi_t']])
            continue

        total = (cnt.groupby(['year', 'hemisphere'])['collection_count']
                    .sum().rename('total_cc').reset_index())
        cnt = cnt.merge(total, on=['year', 'hemisphere'])
        cnt['xi_t'] = np.where(
            cnt['total_cc'] > 0,
            cnt['collection_count'] / cnt['total_cc'],
            np.nan
        )
        cnt['subtype'] = subtype
        frames.append(cnt[['subtype', 'year', 'hemisphere', 'clade', 'xi_t']])
        print(f"  [CollFreqData] {subtype}: {len(cnt)} clade-records loaded "
              f"(collection_count-based) from {path}")

    if not frames:
        print("  [WARN] No clade count file loaded for collection_count.")
        return pd.DataFrame(columns=['subtype', 'year', 'hemisphere', 'clade', 'xi_t'])
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════
# 温度网格搜索
# ═════════════════════════════════════════════════════════════
def find_best_temperatures(past_list: list, combo_metrics: list) -> tuple:
    n_m = len(combo_metrics)
    if not past_list:
        return {m: 1.0 for m in combo_metrics}, np.nan

    T_grid     = np.array(list(product(T_VALUES, repeat=n_m)), dtype=float)
    n_combo    = len(T_grid)
    total_loss = np.zeros(n_combo)
    valid_cnt  = 0

    for data in past_list:
        z_mat    = data['z_matrix']
        xi_prev  = data['xi_prev']
        act_freq = data['actual_freq']
        valid_cnt += 1

        log_sig = np.log(safe_sigmoid(
            z_mat[np.newaxis, :, :] / T_grid[:, :, np.newaxis]
        ))
        fitness   = log_sig.sum(axis=1)
        numerator = xi_prev[np.newaxis, :] * np.exp(fitness)
        Z         = numerator.sum(axis=1, keepdims=True)
        valid_mask = Z.flatten() > 0

        pred_freq = np.where(Z > 0, numerator / Z, 0.0)
        loss = np.abs(pred_freq - act_freq[np.newaxis, :]).sum(axis=1)
        loss[~valid_mask] = np.inf
        total_loss += loss

    if valid_cnt == 0:
        return {m: 1.0 for m in combo_metrics}, np.nan

    mean_loss      = total_loss / valid_cnt
    best_idx       = int(np.argmin(mean_loss))
    best_mean_loss = float(mean_loss[best_idx])
    if np.isinf(best_mean_loss):
        best_mean_loss = np.nan

    return (
        {combo_metrics[i]: float(T_grid[best_idx, i]) for i in range(n_m)},
        best_mean_loss,
    )


# ═════════════════════════════════════════════════════════════
# 主准确率计算
# act_freq 用 collection_count；freq_prev 用 submission_count。
# ═════════════════════════════════════════════════════════════
def compute_combination_accuracy(max_df:       pd.DataFrame,
                                 freq_df:      pd.DataFrame,
                                 label_df:     pd.DataFrame,
                                 freq_df_coll: pd.DataFrame) -> tuple:
    """
    Parameters
    ----------
    max_df       : 各指标最大法处理后的 DataFrame
    freq_df      : 基于 submission_count 的 clade 频率（用于温度优化、预测计算及 freq_prev 输出）
    label_df     : 真实 clade 标签
    freq_df_coll : 基于 collection_count 的 clade 频率（仅用于输出 act_freq 列）

    Returns
    -------
    (acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df)
    """
    acc_rows     = []
    egdfit_rows  = []
    egdtemp_rows = []
    lpd_rows     = []
    pre_act_dict: dict = {}

    for combo_name, combo_metrics in COMBINATIONS:
        n_m = len(combo_metrics)
        print(f"\n  ══ Combination: {combo_name}  metrics={combo_metrics} ══")

        for subtype in sorted(max_df['subtype'].unique()):
            sub_clade = max_df[max_df['subtype'] == subtype]
            sub_label = label_df[label_df['subtype'] == subtype]
            sub_freq  = freq_df[freq_df['subtype'] == subtype]
            sub_freq_coll = freq_df_coll[freq_df_coll['subtype'] == subtype]

            if sub_label.empty:
                print(f"    [WARN] {subtype}: no label records, skipping")
                continue

            label_hy    = (sub_label[['hemisphere', 'year']]
                           .drop_duplicates().reset_index(drop=True))
            total_count = len(label_hy)

            # ── Step 1 ──
            all_seasons: set = set()
            for hemi in sub_clade['hemisphere'].unique():
                for year in sub_clade[sub_clade['hemisphere'] == hemi]['year'].unique():
                    all_seasons.add((year, hemi))

            def season_sort_key(hy):
                year, hemi = hy
                hemi_order = 0 if hemi.lower() == 'south' else 1
                return (year, hemi_order)

            sorted_seasons = sorted(all_seasons, key=season_sort_key)
            if not sorted_seasons:
                continue

            # ── Step 2: 基于 submission_count 构建 per_season（温度优化专用）──
            per_season: dict = {}
            for (year, hemi) in sorted_seasons:
                hemi_clade = sub_clade[sub_clade['hemisphere'] == hemi]
                hemi_freq  = sub_freq[sub_freq['hemisphere'] == hemi]

                grp    = hemi_clade[hemi_clade['year'] == year].reset_index(drop=True)
                clades = grp['clade_single'].values.copy()

                z_mat = np.vstack([
                    z_score_arr(grp[f'fit_{m}'].fillna(0.0).values)
                    for m in combo_metrics
                ])

                freq_prev = hemi_freq[hemi_freq['year'] == year - 1]
                c2xi      = dict(zip(freq_prev['clade'], freq_prev['xi_t']))
                xi_prev   = np.array(
                    [np.nan_to_num(c2xi.get(c, 0.0), nan=0.0) for c in clades]
                )

                freq_cur     = hemi_freq[hemi_freq['year'] == year]
                present_set  = set(clades)
                freq_cur_sub = freq_cur[freq_cur['clade'].isin(present_set)]
                raw_sum      = freq_cur_sub['xi_t'].fillna(0.0).sum()

                if freq_cur_sub.empty or raw_sum <= 0:
                    act_freq = None
                else:
                    c2act    = dict(zip(freq_cur_sub['clade'],
                                        freq_cur_sub['xi_t'].fillna(0.0)))
                    act_arr  = np.array([c2act.get(c, 0.0) for c in clades])
                    act_freq = act_arr / act_arr.sum()

                per_season[(year, hemi)] = dict(
                    clades   = clades,
                    z_mat    = z_mat,
                    xi_prev  = xi_prev,
                    act_freq = act_freq,
                )

            # ── Step 3: 温度优化（无窗口限制，使用全部历史）──
            stored: dict = {}

            for idx, (year, hemi) in enumerate(sorted_seasons):
                if idx == 0:
                    T_best         = {m: 1.0 for m in combo_metrics}
                    best_mean_loss = np.nan
                else:
                    history_seasons = sorted_seasons[:max(0, idx - 1)]   # 全部历史，无窗口限制
                    past_data = [
                        dict(
                            z_matrix    = per_season[(py, ph)]['z_mat'],
                            xi_prev     = per_season[(py, ph)]['xi_prev'],
                            actual_freq = per_season[(py, ph)]['act_freq'],
                        )
                        for (py, ph) in history_seasons
                        if per_season[(py, ph)]['act_freq'] is not None
                    ]
                    T_best, best_mean_loss = find_best_temperatures(
                        past_data, combo_metrics
                    )

                stored[(hemi, year)] = dict(
                    clades         = per_season[(year, hemi)]['clades'],
                    z_mat          = per_season[(year, hemi)]['z_mat'],
                    xi_prev        = per_season[(year, hemi)]['xi_prev'],
                    T              = T_best,
                    best_mean_loss = best_mean_loss,
                )

            # ── Step 3b: E+G+D 专属输出 ──
            if combo_name == 'E+G+D':
                for (hemi, year), d in stored.items():
                    clades = d['clades']
                    z_mat  = d['z_mat']
                    T      = d['T']
                    T_col  = np.array([T[m] for m in combo_metrics],
                                       dtype=float)[:, np.newaxis]
                    log_sig = np.log(safe_sigmoid(z_mat / T_col))
                    fitness = log_sig.sum(axis=0)

                    for i, clade in enumerate(clades):
                        egdfit_rows.append({
                            'subtype':    subtype,
                            'hemisphere': hemi,
                            'year':       year,
                            'clade':      clade,
                            'fit_E+G+D':  round(float(fitness[i]), 6),
                        })

                    top1_clade = clades[np.argmax(fitness)] if len(clades) > 0 else None
                    true_rows  = sub_label[
                        (sub_label['hemisphere'] == hemi) &
                        (sub_label['year']       == year)
                    ]
                    if not true_rows.empty and top1_clade is not None:
                        true_clade = true_rows['clade'].iloc[0]
                        tf = 'T' if top1_clade == true_clade else 'F'
                    else:
                        true_clade = None
                        tf         = None

                    egdtemp_rows.append({
                        'subtype':    subtype,
                        'hemisphere': hemi,
                        'year':       year,
                        'tem_E':      T['total_escape'],
                        'tem_G':      T['predicted_prevalence'],
                        'tem_D':      T['mutual_information'],
                        '平均L1损失': d['best_mean_loss'],
                        'TOP1_clade': top1_clade,
                        'T/F':        tf,
                    })

            # ── Step 4a: 逐 clade 记录预测频率与实际频率 ──
            # act_freq 基于 collection_count（freq_df_coll）；
            # freq_prev 与预测列（*_pre_fit / *_pre_freq）基于 submission_count。
            for (hemi, year), d in stored.items():
                clades = d['clades']
                z_mat  = d['z_mat']
                xi_p   = d['xi_prev']   # submission_count based（用于预测计算与 freq_prev 输出）
                T      = d['T']

                if len(clades) == 0:
                    continue

                T_col   = np.array([T[m] for m in combo_metrics],
                                    dtype=float)[:, np.newaxis]
                log_sig = np.log(safe_sigmoid(z_mat / T_col))
                fitness = log_sig.sum(axis=0)

                fit_freq_arr = fitness.copy()

                numerator = xi_p * np.exp(fitness)
                Z         = numerator.sum()
                if Z > 0:
                    freq_freq_arr = numerator / Z
                else:
                    freq_freq_arr = np.full(len(clades), np.nan)

                # ── act_freq：基于 collection_count ──
                hemi_freq_coll = sub_freq_coll[sub_freq_coll['hemisphere'] == hemi]
                fc_coll        = hemi_freq_coll[hemi_freq_coll['year'] == year]
                fc_coll_sub    = fc_coll[fc_coll['clade'].isin(set(clades))]
                raw_sum_coll   = fc_coll_sub['xi_t'].fillna(0.0).sum()
                if fc_coll_sub.empty or raw_sum_coll <= 0:
                    act_freq_coll = None
                else:
                    c2act_coll    = dict(zip(fc_coll_sub['clade'],
                                             fc_coll_sub['xi_t'].fillna(0.0)))
                    act_arr_coll  = np.array([c2act_coll.get(c, 0.0) for c in clades])
                    act_freq_coll = act_arr_coll / act_arr_coll.sum()

                for i, clade in enumerate(clades):
                    pk = (subtype, hemi, year, clade)
                    if pk not in pre_act_dict:
                        # act_freq  : collection_count based
                        # freq_prev : submission_count based（直接取 xi_p[i]）
                        act_val  = (float(act_freq_coll[i])
                                    if act_freq_coll is not None else np.nan)
                        prev_val = float(xi_p[i])
                        pre_act_dict[pk] = {
                            'subtype':    subtype,
                            'year':       year,
                            'hemisphere': hemi,
                            'clade':      clade,
                            'act_freq':   (round(act_val,  6)
                                           if np.isfinite(act_val)  else np.nan),
                            'freq_prev':  (round(prev_val, 6)
                                           if np.isfinite(prev_val) else np.nan),
                        }

                    ffit  = float(fit_freq_arr[i])
                    ffreq = float(freq_freq_arr[i])
                    pre_act_dict[pk][f'{combo_name}_pre_fit']  = (
                        round(ffit,  6) if np.isfinite(ffit)  else np.nan)
                    pre_act_dict[pk][f'{combo_name}_pre_freq'] = (
                        round(ffreq, 6) if np.isfinite(ffreq) else np.nan)

            # ── Step 4: 准确率 + LPD 记录 ──
            hit_fit = hit_freq = 0

            for _, lrow in label_hy.iterrows():
                hemi      = lrow['hemisphere']
                year      = lrow['year']
                true_rows = sub_label[
                    (sub_label['hemisphere'] == hemi) &
                    (sub_label['year']       == year)
                ]
                if true_rows.empty:
                    continue
                true_clade = true_rows['clade'].iloc[0]

                key = (hemi, year)
                if key not in stored:
                    lpd_rows.append({
                        'subtype': subtype, 'hemisphere': hemi,
                        'year': year, 'combo': combo_name, 'lpd': np.nan,
                    })
                    continue

                d      = stored[key]
                clades = d['clades']
                z_mat  = d['z_mat']
                xi_p   = d['xi_prev']
                T      = d['T']

                if len(clades) == 0:
                    lpd_rows.append({
                        'subtype': subtype, 'hemisphere': hemi,
                        'year': year, 'combo': combo_name, 'lpd': np.nan,
                    })
                    continue

                T_col   = np.array([T[m] for m in combo_metrics],
                                    dtype=float)[:, np.newaxis]
                log_sig = np.log(safe_sigmoid(z_mat / T_col))
                fitness = log_sig.sum(axis=0)

                pred_fit_clade = clades[np.argmax(fitness)]
                if pred_fit_clade == true_clade:
                    hit_fit += 1

                numerator = xi_p * np.exp(fitness)
                Z         = numerator.sum()
                if Z > 0:
                    pred_freq_arr   = numerator / Z
                    pred_freq_clade = clades[np.argmax(numerator)]
                    if pred_freq_clade == true_clade:
                        hit_freq += 1
                    true_idx_arr = np.where(clades == true_clade)[0]
                    if len(true_idx_arr) > 0:
                        pred_prob = max(float(pred_freq_arr[true_idx_arr[0]]), 1e-10)
                    else:
                        pred_prob = 1e-10
                    lpd_value = float(np.log(pred_prob))
                else:
                    lpd_value = float(np.log(1e-10))

                lpd_rows.append({
                    'subtype':    subtype,
                    'hemisphere': hemi,
                    'year':       year,
                    'combo':      combo_name,
                    'lpd':        lpd_value,
                })

            print(f"    [{subtype}] combo={combo_name}  "
                  f"fit={hit_fit}/{total_count}  freq={hit_freq}/{total_count}")

            for method_name, hit in [('fit', hit_fit), ('freq', hit_freq)]:
                acc_rows.append({
                    'subtype':        subtype,
                    'metric_combine': combo_name,
                    'accuracy':       (round(hit / total_count, 4)
                                       if total_count > 0 else np.nan),
                    'hit_count':      hit,
                    'total_count':    total_count,
                    'methods':        method_name,
                })

    # ── 构建 pre_act_df ──
    if pre_act_dict:
        pre_act_df = pd.DataFrame(list(pre_act_dict.values()))
        for col in PRE_ACT_COL_ORDER:
            if col not in pre_act_df.columns:
                pre_act_df[col] = np.nan
        pre_act_df = pre_act_df[PRE_ACT_COL_ORDER].sort_values(
            ['subtype', 'year', 'hemisphere', 'clade']
        ).reset_index(drop=True)
    else:
        pre_act_df = pd.DataFrame(columns=PRE_ACT_COL_ORDER)

    acc_df     = pd.DataFrame(acc_rows)
    egdfit_df  = pd.DataFrame(egdfit_rows)
    egdtemp_df = pd.DataFrame(egdtemp_rows)
    lpd_df     = pd.DataFrame(lpd_rows)

    return acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df


# ═════════════════════════════════════════════════════════════
# ELPD / AIC 构建
# ═════════════════════════════════════════════════════════════
def build_elpd_aic(lpd_df: pd.DataFrame) -> tuple:
    if lpd_df.empty:
        print("  [WARN] lpd_df is empty; ELPD/AIC tables will be empty.")
        return pd.DataFrame(), pd.DataFrame()

    elpd_pivot = lpd_df.pivot_table(
        index=['subtype', 'hemisphere', 'year'],
        columns='combo',
        values='lpd',
        aggfunc='first'
    ).reset_index()
    elpd_pivot.columns.name = None

    rename_map = {c: f'ELPD_{c}' for c in COMBO_NAMES_ORDERED
                  if c in elpd_pivot.columns}
    elpd_pivot = elpd_pivot.rename(columns=rename_map)

    for combo in COMBO_NAMES_ORDERED:
        col = f'ELPD_{combo}'
        if col not in elpd_pivot.columns:
            elpd_pivot[col] = np.nan

    elpd_cols  = [f'ELPD_{c}' for c in COMBO_NAMES_ORDERED]
    elpd_pivot = elpd_pivot[['subtype', 'hemisphere', 'year'] + elpd_cols]

    summary_rows = []
    for subtype in sorted(elpd_pivot['subtype'].unique()):
        sub = elpd_pivot[elpd_pivot['subtype'] == subtype]
        row = {'subtype': subtype, 'hemisphere': 'Summary', 'year': 'All'}
        for col in elpd_cols:
            row[col] = round(float(np.nansum(sub[col])), 4)
        summary_rows.append(row)

    elpd_final = pd.concat(
        [elpd_pivot, pd.DataFrame(summary_rows)],
        ignore_index=True
    )

    aic_rows = []
    for subtype in sorted(elpd_pivot['subtype'].unique()):
        sub = elpd_pivot[elpd_pivot['subtype'] == subtype]
        row = {'subtype': subtype}
        for combo in COMBO_NAMES_ORDERED:
            col      = f'ELPD_{combo}'
            elpd_sum = float(np.nansum(sub[col]))
            k        = N_PARAMS[combo]
            row[f'AIC_{combo}'] = round(2 * k - 2 * elpd_sum, 4)
        aic_rows.append(row)

    aic_df = pd.DataFrame(aic_rows)
    return elpd_final, aic_df


# ═════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════
print("\n[1/4] Loading and filtering component data (per-subtype min_seq_count) ...")
df = pd.read_csv(COMPONENT_PATH)
print(f"  Total rows before filtering: {len(df)}")

# 按亚型分别过滤 min_seq_count
filtered_frames = []
for subtype, sub_df in df.groupby('subtype'):
    min_count = get_min_seq_count(subtype)
    kept = sub_df[sub_df['mutation_group_seq_count'] >= min_count]
    print(f"  [{subtype}] min_seq_count={min_count}  "
          f"kept {len(kept)}/{len(sub_df)} rows "
          f"(dropped {len(sub_df) - len(kept)})")
    filtered_frames.append(kept)

if filtered_frames:
    df = pd.concat(filtered_frames, ignore_index=True)
else:
    df = df.iloc[0:0].reset_index(drop=True)

print(f"  Total rows after filtering : {len(df)}")

missing_cols = [m for m in METRICS_3 if m not in df.columns]
if missing_cols:
    raise ValueError(f"输入文件缺少以下列: {missing_cols}\n"
                     f"实际列名: {df.columns.tolist()}")

print("[2/4] Loading label and clade frequency data ...")
label_df = pd.read_csv(LABEL_PATH)
if 'region' in label_df.columns and 'hemisphere' not in label_df.columns:
    label_df = label_df.rename(columns={'region': 'hemisphere'})

subtype_list = df['subtype'].unique().tolist()

# submission_count 版本（温度优化、预测计算及 freq_prev 输出）
freq_df = load_clade_freq(subtype_list)

# collection_count 版本（仅用于 act_freq 输出列）
print("[2b/4] Loading collection_count-based clade frequency data ...")
freq_df_coll = load_clade_freq_collection(subtype_list)

print(f"  Component      : {len(df)} rows | subtypes: {df['subtype'].unique().tolist()}")
print(f"  Labels         : {len(label_df)} rows | subtypes: {label_df['subtype'].unique().tolist()}")
print(f"  FreqData (sub) : {len(freq_df)} rows")
print(f"  FreqData (col) : {len(freq_df_coll)} rows")

print("[3/4] Computing max method (no min-max normalisation) ...")
max_df, divergence_inform_df, escape_inform_df, growth_inform_df = \
    compute_max_method_with_info(df)
print(f"  max_df             : {len(max_df)} rows")
print(f"  divergence_inform  : {len(divergence_inform_df)} rows")
print(f"  escape_inform      : {len(escape_inform_df)} rows")
print(f"  growth_inform      : {len(growth_inform_df)} rows")

print("[4/4] Computing combination accuracy with rolling temperature optimisation ...")
print(f"  T candidate pool : {T_VALUES}")
print(f"  Grid sizes       : 1-metric={len(T_VALUES)**1}, "
      f"2-metric={len(T_VALUES)**2}, 3-metric={len(T_VALUES)**3}")
print(f"  Max history (n)  : unlimited")

acc_df, egdfit_df, egdtemp_df, lpd_df, pre_act_df = compute_combination_accuracy(
    max_df, freq_df, label_df, freq_df_coll
)

# ── 保存准确率主文件 ──
acc_df = acc_df[['subtype', 'metric_combine', 'accuracy',
                 'hit_count', 'total_count', 'methods']]
acc_df.to_csv(OUT_ACC_PATH, index=False)
print(f"\n[Done] Saved {OUT_ACC_PATH}  ({len(acc_df)} rows)")

# ── 保存 EGD_combine_fit ──
if not egdfit_df.empty:
    egdfit_df = egdfit_df[['subtype', 'hemisphere', 'year', 'clade', 'fit_E+G+D']]
    egdfit_df.to_csv(OUT_EGD_FIT_PATH, index=False)
    print(f"[Done] Saved {OUT_EGD_FIT_PATH}  ({len(egdfit_df)} rows)")
else:
    print("[WARN] EGD fit data is empty; file not saved.")

# ── 保存 EGD_temperatures ──
if not egdtemp_df.empty:
    egdtemp_df = egdtemp_df[['subtype', 'hemisphere', 'year',
                              'tem_E', 'tem_G', 'tem_D', '平均L1损失',
                              'TOP1_clade', 'T/F']]
    egdtemp_df.to_csv(OUT_EGD_TEMP_PATH, index=False)
    print(f"[Done] Saved {OUT_EGD_TEMP_PATH}  ({len(egdtemp_df)} rows)")
else:
    print("[WARN] EGD temperature data is empty; file not saved.")

# ── 构建并保存 ELPD / AIC ──
print("\n[5/6] Building ELPD and AIC tables ...")
elpd_final, aic_df = build_elpd_aic(lpd_df)

if not elpd_final.empty:
    elpd_final.to_csv(OUT_ELPD_PATH, index=False)
    n_summary = elpd_final['hemisphere'].eq('Summary').sum()
    print(f"[Done] Saved {OUT_ELPD_PATH}  "
          f"({len(elpd_final)} rows, incl. {n_summary} summary row(s))")
else:
    print("[WARN] ELPD data is empty; file not saved.")

if not aic_df.empty:
    aic_df.to_csv(OUT_AIC_PATH, index=False)
    print(f"[Done] Saved {OUT_AIC_PATH}  ({len(aic_df)} rows)")
else:
    print("[WARN] AIC data is empty; file not saved.")

# ── 保存 pre_act_freq ──
print("\n[6/7] Saving per-clade predicted / actual frequency table ...")
if not pre_act_df.empty:
    pre_act_df.to_csv(OUT_PRE_ACT_PATH, index=False)
    print(f"[Done] Saved {OUT_PRE_ACT_PATH}  ({len(pre_act_df)} rows, "
          f"{len(pre_act_df.columns)} columns)")
else:
    print("[WARN] pre_act_freq data is empty; file not saved.")

# ── 保存 divergence_inform / escape_inform / growth_inform ──
print("\n[7/7] Saving component inform tables ...")
for path, name, df_info, col_order in [
    (OUT_DIVERGENCE_PATH, 'divergence_inform', divergence_inform_df, DIVERGENCE_COL_ORDER),
    (OUT_ESCAPE_PATH,     'escape_inform',     escape_inform_df,     ESCAPE_COL_ORDER),
    (OUT_GROWTH_PATH,     'growth_inform',     growth_inform_df,     GROWTH_COL_ORDER),
]:
    if not df_info.empty:
        for col in col_order:
            if col not in df_info.columns:
                df_info[col] = np.nan
        df_out = (df_info[col_order]
                  .sort_values(['subtype', 'hemisphere', 'year', 'clade'])
                  .reset_index(drop=True))
        df_out.to_csv(path, index=False)
        print(f"[Done] Saved {path}  ({len(df_out)} rows, cols={col_order})")
    else:
        print(f"[WARN] {name} data is empty; file not saved.")
