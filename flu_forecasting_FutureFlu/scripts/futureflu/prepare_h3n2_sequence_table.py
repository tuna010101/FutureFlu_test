"""Build the H3N2 sequence table from FASTA and metadata exports.

English: Filters sequence records and writes tabular amino-acid sequence data.
中文：过滤序列记录，并写出表格化氨基酸序列数据。
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences" / ".staging"

# English: The final package keeps this path layout, but the raw inputs are omitted.
# 中文：final 包保留该路径布局，但原始输入已省略。
INPUT_FASTA = PROJECT_ROOT / "data" / "sources" / "msa-H3N2-all-20250131-submission.fasta"
INPUT_METADATA = PROJECT_ROOT / "data" / "sources" / "H3N2-all-20250131-submission.csv"

OUTPUT_TSV = STAGING_OUTPUT_DIR / "h3n2_sequence_table.tsv"
OUTPUT_PKL = STAGING_OUTPUT_DIR / "h3n2_sequence_table.pkl"
OUTPUT_REPORT = PROJECT_ROOT / "results" / "futureflu" / "artifacts" / "h3n2_sequence_table_artifacts.md"


@contextmanager
def fasta_input(path: Path):
    """Open FASTA input. / 打开 FASTA 输入。"""
    with path.open("r", encoding="utf-8") as handle:
        yield handle


def parallel_sequence_processing(
    chunk_data: Tuple[List[str], List[str], pd.DataFrame],
    cutoff_date: str,
    subtype: str = "H3N2",
) -> List[dict]:
    sequences, names, info_data = chunk_data
    valid_rows = []
    for seq, name in zip(sequences, names):
        parts = name.split("|")
        isolate_id = parts[0]
        victoria_segment_id = parts[1] if len(parts) > 1 else parts[0]

        if seq.count("-") > 3:
            continue

        if subtype == "Victoria":
            iso_id_col = (
                info_data["Isolate_Id"].fillna("")
                if "Isolate_Id" in info_data.columns
                else pd.Series([""] * len(info_data))
            )
            mask = iso_id_col.astype(str) == victoria_segment_id
        else:
            mask = info_data["Isolate_Id"] == isolate_id

        collection_info = info_data.loc[mask, "Collection_Date"].values
        submission_info = info_data.loc[mask, "Submission_Date"].values
        clade_info = info_data.loc[mask, "Clade"].values
        isolate_name_info = (
            info_data.loc[mask, "Isolate_Name"].values
            if "Isolate_Name" in info_data.columns
            else np.array([], dtype=object)
        )

        if len(collection_info) > 0 and pd.notna(collection_info[0]):
            collection_str = str(collection_info[0])
            submission_str = (
                str(submission_info[0])
                if len(submission_info) > 0 and pd.notna(submission_info[0])
                else ""
            )

            if submission_str:
                parts = submission_str.split("-")
                if len(parts) == 1:
                    submission_str = f"{submission_str}-01-01"
                elif len(parts) == 2:
                    submission_str = f"{submission_str}-01"
            try:
                submission_date = pd.to_datetime(submission_str) if submission_str else None
            except (ValueError, TypeError):
                submission_date = None

            parts = collection_str.split("-")
            if len(parts) == 1:
                collection_str = f"{collection_str}-01-01"
            elif len(parts) == 2:
                collection_str = f"{collection_str}-01"
            try:
                collection_date = pd.to_datetime(collection_str)
            except (ValueError, TypeError):
                continue

            if submission_date is None:
                submission_date = collection_date
            if submission_date >= pd.to_datetime(cutoff_date):
                continue
            if collection_date.year >= 2010:
                row_data = {
                    "accession_number": isolate_id,
                    "name": (
                        str(isolate_name_info[0])
                        if len(isolate_name_info) > 0 and pd.notna(isolate_name_info[0])
                        else ""
                    ),
                    "clade": (
                        clade_info[0]
                        if len(clade_info) > 0 and pd.notna(clade_info[0])
                        else ""
                    ),
                    "collection_date": collection_date.strftime("%Y-%m-%d"),
                    "submission_date": submission_date.strftime("%Y-%m-%d"),
                    "season": collection_date.year - 1 if collection_date.month < 2 else collection_date.year,
                }

                for j, aa in enumerate(seq):
                    row_data[f"X{j+1}"] = aa
                valid_rows.append(row_data)
    return valid_rows


def prepare_seq(
    input_fasta_seq: str | Path,
    info_file: str | Path,
    cutoff_date: str,
    subtype: str = "H3N2",
    processes: int | None = None,
) -> pd.DataFrame:
    sequences: List[str] = []
    names: List[str] = []
    with fasta_input(Path(input_fasta_seq)) as fasta_handle:
        for record in SeqIO.parse(fasta_handle, "fasta"):
            if record.id == "H3N2_reference":
                continue
            sequences.append(str(record.seq))
            names.append(str(record.id))

    info_data = pd.read_csv(info_file, dtype=str, low_memory=False)

    n_cores = processes if processes is not None else cpu_count()
    chunk_size = len(sequences) // n_cores + 1
    sequence_chunks = [sequences[i : i + chunk_size] for i in range(0, len(sequences), chunk_size)]
    name_chunks = [names[i : i + chunk_size] for i in range(0, len(names), chunk_size)]
    chunks = [(seq_chunk, name_chunk, info_data) for seq_chunk, name_chunk in zip(sequence_chunks, name_chunks)]

    process_func = partial(parallel_sequence_processing, cutoff_date=cutoff_date, subtype=subtype)

    if n_cores <= 1 or len(chunks) <= 1:
        results = [process_func(chunk) for chunk in chunks]
    else:
        with Pool(n_cores) as pool:
            results = pool.map(process_func, chunks)

    all_rows = list(itertools.chain(*results))
    return pd.DataFrame(all_rows)


def main() -> None:
    cutoff_date = "2025-02-01"
    df = prepare_seq(INPUT_FASTA, INPUT_METADATA, cutoff_date=cutoff_date, subtype="H3N2", processes=1)
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    df.to_pickle(OUTPUT_PKL)

    report = f"""# H3N2 Sequence Table Artifacts

## Command

```bash
python scripts/futureflu/prepare_h3n2_sequence_table.py
```

## Logic source

This step mirrors the filtering logic in:

- `futureflu/sequence.py`

## Outputs

- `data/sequences/.staging/h3n2_sequence_table.tsv`
- `data/sequences/.staging/h3n2_sequence_table.pkl`

## Summary

- rows written: {len(df)}
- seasons covered: {df['season'].min()} to {df['season'].max()}
- collection date min: {df['collection_date'].min()}
- collection date max: {df['collection_date'].max()}

## Important note

This table is the canonical intermediate file for repeated probe/subset work in
this project.
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_TSV}")
    print(f"Wrote {OUTPUT_PKL}")
    print(f"Wrote {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
