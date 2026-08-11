"""Compute clade-level component maxima and prediction accuracy.

English: Read packaged component and truth tables, apply subtype-specific
filters, and write normalized maximum-score and accuracy tables.
中文：读取发布包中的组件表和真值表，应用亚型级筛选，并输出归一化最大得分及准确率表。
"""

import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path

# ═════════════════════════════════════════════════════════════
# English: Parse output-file arguments (acc_output, max_output).
# 中文：解析输出文件参数（acc_output、max_output）。
# ═════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description='Compute clade-level metrics and accuracy.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('acc_output', type=str,
                    help='准确率输出文件名（例如：clade_component_acc.csv）')
parser.add_argument('max_output', type=str,
                    help='最大法输出文件名（例如：clade_component_max.csv）')
args = parser.parse_args()

ACC_FILENAME = args.acc_output
MAX_FILENAME = args.max_output

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
OUT_MAX_PATH = OUT_DIR / MAX_FILENAME
OUT_ACC_PATH = OUT_DIR / ACC_FILENAME

print(f"[Config] H1N1/Victoria min_seq_count : 0")
print(f"[Config] H3N2          min_seq_count : 10")
print(f"[Config] max output file             : {OUT_MAX_PATH}")
print(f"[Config] accuracy output file        : {OUT_ACC_PATH}")

# ═════════════════════════════════════════════════════════════
# English: Output-column constants; frequency columns are not included.
# 中文：输出列常量；移除 FREQ_COLS，OUT_COLS 不再包含 freq_ 列。
# ═════════════════════════════════════════════════════════════
METRICS = [
    'total_escape', 'predicted_prevalence', 'mutual_information',
    'dissimilarity_charge_hydro', 'accessibility_wcn', 'fitness_eve',
    'antigenic_novelty'
]

FIT_COLS = [f'fit_{m}' for m in METRICS]

OUT_COLS = (
    ['subtype', 'hemisphere', 'year', 'clade_single']
    + FIT_COLS
)

# ═════════════════════════════════════════════════════════════
# English: Load component and truth tables, then filter by subtype threshold.
# 中文：读取组件和标签数据，并按亚型 min_seq_count 分别筛选。
# ═════════════════════════════════════════════════════════════
df = pd.read_csv(COMPONENT_PATH)
print(f"[Pre-filter] Total rows before filtering: {len(df)}")

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

print(f"[Pre-filter] Total rows after filtering : {len(df)}")

missing = [m for m in METRICS if m not in df.columns]
if missing:
    raise ValueError(f"以下指标列在输入文件中缺失: {missing}\n"
                     f"实际列名: {df.columns.tolist()}")

label_df = pd.read_csv(LABEL_PATH)
if 'region' in label_df.columns and 'hemisphere' not in label_df.columns:
    label_df = label_df.rename(columns={'region': 'hemisphere'})

print(f"Component file : {len(df)} rows, subtypes: {df['subtype'].unique().tolist()}")
print(f"Label file     : {len(label_df)} rows, subtypes: {label_df['subtype'].unique().tolist()}")

# ═════════════════════════════════════════════════════════════
# English: Parse clade names and percentages from formatted strings.
# 中文：从格式化字符串中解析 clade 名称和百分比。
# ═════════════════════════════════════════════════════════════
_CLADE_RE = re.compile(r'([^\(,]+?)\s*\((\d+\.?\d*)%\)')


def parse_clade_string(clade_str):
    if pd.isna(clade_str) or str(clade_str).strip().lower() in ('unknown', ''):
        return []
    results = []
    for name, pct_str in _CLADE_RE.findall(str(clade_str)):
        name = name.strip()
        if name.lower() not in ('unassigned', 'unknown'):
            results.append((name, float(pct_str) / 100.0))
    return results


# ═════════════════════════════════════════════════════════════
# English: Select the dominant clade and aggregate maximum fit values by group.
# 中文：选择占比最高的 clade，并按组聚合 fit_ 最大值。
# ═════════════════════════════════════════════════════════════
def compute_max_method(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        parsed = parse_clade_string(row['clade'])
        if not parsed:
            continue
        dominant_clade = max(parsed, key=lambda x: x[1])[0]
        rec = {
            'subtype':      row['subtype'],
            'hemisphere':   row['hemisphere'],
            'year':         row['year'],
            'clade_single': dominant_clade,
        }
        for m in METRICS:
            rec[f'fit_{m}'] = row.get(m, np.nan)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(
            columns=['subtype', 'hemisphere', 'year', 'clade_single'] + FIT_COLS)

    tmp = pd.DataFrame(rows)
    agg = (
        tmp.groupby(['subtype', 'hemisphere', 'year', 'clade_single'], as_index=False)
           .agg({f'fit_{m}': 'max' for m in METRICS})
    )
    return agg


# ═════════════════════════════════════════════════════════════
# English: Min-max normalize fit columns within subtype/hemisphere/year groups;
# use 0.0 when all values in a group are equal.
# 中文：在 subtype/hemisphere/year 分组内对 fit_ 列做 Min-Max 归一化；
# 若组内值全部相同，则归一化结果设为 0.0。
# ═════════════════════════════════════════════════════════════
def normalize_fit_columns(clade_df: pd.DataFrame) -> pd.DataFrame:
    result = clade_df.copy()
    GROUP_KEYS = ['subtype', 'hemisphere', 'year']

    for _, grp in result.groupby(GROUP_KEYS):
        idx = grp.index
        for col in FIT_COLS:
            if col not in result.columns:
                continue
            vals  = result.loc[idx, col]
            v_min = vals.min()
            v_max = vals.max()
            if v_max > v_min:
                result.loc[idx, col] = (vals - v_min) / (v_max - v_min)
            else:
                result.loc[idx, col] = 0.0

    return result


# ═════════════════════════════════════════════════════════════
# English: Compare maximum-score predictions with the packaged truth labels.
# 中文：将最大得分预测与发布包真值标签比较并计算准确率。
# ═════════════════════════════════════════════════════════════
def compute_accuracy(clade_df: pd.DataFrame,
                     label_df: pd.DataFrame,
                     method_name: str,
                     col_prefix: str = 'fit_') -> pd.DataFrame:
    acc_rows = []

    for subtype in sorted(clade_df['subtype'].unique()):
        sub_clade = clade_df[clade_df['subtype'] == subtype]
        sub_label = label_df[label_df['subtype'] == subtype]

        if sub_label.empty:
            print(f"  [WARN] subtype={subtype} 在标签文件中无记录，跳过")
            continue

        label_hy    = (sub_label[['hemisphere', 'year']]
                       .drop_duplicates()
                       .reset_index(drop=True))
        total_count = len(label_hy)

        for metric in METRICS:
            col = f'{col_prefix}{metric}'
            if col not in sub_clade.columns:
                continue

            hit_count = 0
            for _, lrow in label_hy.iterrows():
                hemi = lrow['hemisphere']
                year = lrow['year']

                true_rows = sub_label.loc[
                    (sub_label['hemisphere'] == hemi) &
                    (sub_label['year']       == year),
                    'clade'
                ]
                if true_rows.empty:
                    continue
                true_clade = true_rows.iloc[0]

                grp = sub_clade[
                    (sub_clade['hemisphere'] == hemi) &
                    (sub_clade['year']       == year)
                ]
                if grp.empty or grp[col].isna().all():
                    continue

                pred_clade = grp.loc[grp[col].idxmax(), 'clade_single']
                if pred_clade == true_clade:
                    hit_count += 1

            acc_rows.append({
                'subtype':     subtype,
                'metric':      metric,
                'hit_count':   hit_count,
                'total_count': total_count,
                'accuracy':    round(hit_count / total_count, 4) if total_count > 0 else np.nan,
                'methods':     method_name,
            })

    return pd.DataFrame(acc_rows)


# ═════════════════════════════════════════════════════════════
# English: Execute the maximum-score, normalization, and accuracy stages.
# 中文：依次执行最大得分、归一化和准确率计算阶段。
# ═════════════════════════════════════════════════════════════
print("\n[1/3] Computing max method (fit_ columns) ...")
max_df = compute_max_method(df)
print(f"      {len(max_df)} rows")

print("[2/3] Normalizing fit_ columns within (subtype, hemisphere, year) ...")
max_df = normalize_fit_columns(max_df)
print("      Done.")


def save_with_cols(df_out, path, expected_cols):
    actual_cols = [c for c in expected_cols if c in df_out.columns]
    missing_out = [c for c in expected_cols if c not in df_out.columns]
    if missing_out:
        print(f"  [WARN] columns missing in output: {missing_out}")
    df_out[actual_cols].to_csv(path, index=False)
    print(f"      Saved {path}  ({len(df_out)} rows, {len(actual_cols)} cols)")


save_with_cols(max_df, OUT_MAX_PATH, OUT_COLS)

print("[3/3] Computing accuracy: fit_max ...")
acc_fit_max = compute_accuracy(max_df, label_df, 'fit_max', col_prefix='fit_')
acc_fit_max = acc_fit_max[['subtype', 'metric', 'accuracy', 'hit_count', 'total_count', 'methods']]
acc_fit_max.to_csv(OUT_ACC_PATH, index=False)
print(f"      Saved {OUT_ACC_PATH}  ({len(acc_fit_max)} rows)")
