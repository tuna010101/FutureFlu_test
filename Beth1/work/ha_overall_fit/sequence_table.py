"""Build beth-1 sequence tables from package-local FASTA and metadata files.

English: Parse FASTA headers against metadata and emit the filtered amino-acid
sequence table used by the HA-only runner.
中文：将 FASTA header 与 metadata 对齐，生成 HA-only runner 使用的过滤后氨基酸
序列表。
"""

from __future__ import annotations

import itertools
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO


def _process_sequence_chunk(
    chunk_data: Tuple[List[str], List[str], pd.DataFrame],
    cutoff_date: str,
    subtype: str = "H3N2",
) -> List[dict]:
    """Process one sequence chunk. / 处理一个序列分块。"""
    sequences, names, info_data = chunk_data
    valid_rows = []
    cutoff = pd.to_datetime(cutoff_date)

    for seq, name in zip(sequences, names):
        parts = name.split("|")
        isolate_id = parts[0]
        victoria_segment_id = parts[1] if len(parts) > 1 else parts[0]

        if seq.count("-") > 3:
            continue

        if subtype == "Victoria":
            isolate_column = (
                info_data["Isolate_Id"].fillna("")
                if "Isolate_Id" in info_data.columns
                else pd.Series([""] * len(info_data))
            )
            mask = isolate_column.astype(str) == victoria_segment_id
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

        if len(collection_info) == 0 or pd.isna(collection_info[0]):
            continue

        collection_text = str(collection_info[0])
        submission_text = (
            str(submission_info[0])
            if len(submission_info) > 0 and pd.notna(submission_info[0])
            else ""
        )

        if submission_text:
            submission_parts = submission_text.split("-")
            if len(submission_parts) == 1:
                submission_text = f"{submission_text}-01-01"
            elif len(submission_parts) == 2:
                submission_text = f"{submission_text}-01"
        try:
            submission_date = pd.to_datetime(submission_text) if submission_text else None
        except (ValueError, TypeError):
            submission_date = None

        collection_parts = collection_text.split("-")
        if len(collection_parts) == 1:
            collection_text = f"{collection_text}-01-01"
        elif len(collection_parts) == 2:
            collection_text = f"{collection_text}-01"
        try:
            collection_date = pd.to_datetime(collection_text)
        except (ValueError, TypeError):
            continue

        if submission_date is None:
            submission_date = collection_date
        if submission_date >= cutoff or collection_date.year < 2010:
            continue

        row_data = {
            "accession number": isolate_id,
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
            "season": (
                collection_date.year - 1
                if collection_date.month < 2
                else collection_date.year
            ),
        }
        for index, amino_acid in enumerate(seq):
            row_data[f"X{index + 1}"] = amino_acid
        valid_rows.append(row_data)

    return valid_rows


def generate_sequence_table(
    subtype: str,
    fasta_path: Path,
    info_path: Path,
    cutoff_date: str,
    processes: int | None = None,
) -> pd.DataFrame:
    """Create the filtered amino-acid sequence table. / 创建过滤后的氨基酸序列表。"""
    fasta_path = Path(fasta_path)
    info_path = Path(info_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(fasta_path)
    if not info_path.is_file():
        raise FileNotFoundError(info_path)

    sequences: List[str] = []
    names: List[str] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        sequences.append(str(record.seq))
        names.append(str(record.id))

    if info_path.suffix.lower() == ".csv":
        info_data = pd.read_csv(info_path)
    else:
        info_data = pd.read_excel(info_path)

    required_columns = {"Isolate_Id", "Collection_Date", "Submission_Date", "Clade"}
    missing = required_columns.difference(info_data.columns)
    if missing:
        raise ValueError(f"{info_path} missing required columns: {sorted(missing)}")

    worker_count = processes if processes is not None else cpu_count()
    worker_count = max(1, int(worker_count))
    chunk_size = len(sequences) // worker_count + 1
    sequence_chunks = [
        sequences[index : index + chunk_size]
        for index in range(0, len(sequences), chunk_size)
    ]
    name_chunks = [
        names[index : index + chunk_size]
        for index in range(0, len(names), chunk_size)
    ]
    chunks = [
        (sequence_chunk, name_chunk, info_data)
        for sequence_chunk, name_chunk in zip(sequence_chunks, name_chunks)
    ]

    process_chunk = partial(
        _process_sequence_chunk,
        cutoff_date=cutoff_date,
        subtype=subtype,
    )
    if worker_count <= 1 or len(chunks) <= 1:
        results = [process_chunk(chunk) for chunk in chunks]
    else:
        with Pool(worker_count) as pool:
            results = pool.map(process_chunk, chunks)

    return pd.DataFrame(list(itertools.chain.from_iterable(results)))


__all__ = ["generate_sequence_table"]
