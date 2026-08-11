#!/usr/bin/env python3
"""Fetch NA references and align local NA sequences to them with MAFFT.

English: Prepare subtype-specific NA reference alignments for the HA/NA controls.
中文：为 HA/NA 对照实验准备各亚型的 NA 参考比对结果。

The source NA FASTA files are amino-acid sequences. The requested accessions are
nucleotide records, so this script extracts the neuraminidase CDS translation
from each GenBank record before running MAFFT.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Bio import SeqIO


EXP_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = EXP_ROOT.parents[1]


def configured_workspace() -> Path:
    """Return only an explicitly configured source workspace.

    English: Do not discover a sibling workspace implicitly.
    中文：不自动发现同级工作区。
    """
    configured = os.environ.get("FUTUREFLU_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return PACKAGE_ROOT / "external" / "workspace"


WORKSPACE = configured_workspace()
NA_FASTA_ROOT = Path(
    os.environ.get(
        "FUTUREFLU_NA_FASTA_ROOT",
        WORKSPACE / "na_fasta",
    )
)
REF_DIR = EXP_ROOT / "data" / "references"
INPUT_DIR = EXP_ROOT / "data" / "na_alignment_inputs"
ALIGNED_DIR = EXP_ROOT / "data" / "na_aligned"
LOG_DIR = EXP_ROOT / "logs"


@dataclass(frozen=True)
class SubtypeSpec:
    subtype: str
    stem: str
    accession: str
    source_fasta: Path


SUBTYPES = {
    "H1N1": SubtypeSpec(
        subtype="H1N1",
        stem="h1n1",
        accession="NC_026434.1",
        source_fasta=NA_FASTA_ROOT / "H1N1_all_sequences.fasta",
    ),
    "H3N2": SubtypeSpec(
        subtype="H3N2",
        stem="h3n2",
        accession="NC_007368.1",
        source_fasta=NA_FASTA_ROOT / "H3N2_all_sequences.fasta",
    ),
    "Victoria": SubtypeSpec(
        subtype="Victoria",
        stem="victoria",
        accession="FJ766839.1",
        source_fasta=NA_FASTA_ROOT / "Vic_all_sequences.fasta",
    ),
}


def ensure_dirs() -> None:
    for path in (REF_DIR, INPUT_DIR, ALIGNED_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fetch_genbank(accession: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"[ref] reuse {output_path}")
        return

    params = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": accession,
            "rettype": "gb",
            "retmode": "text",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    print(f"[ref] fetching {accession}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to fetch {accession} from NCBI: {exc}") from exc
    if "LOCUS" not in text:
        raise RuntimeError(f"NCBI response for {accession} does not look like GenBank")
    output_path.write_text(text, encoding="utf-8")


def _feature_text(feature, key: str) -> str:
    return " ".join(feature.qualifiers.get(key, [])).strip()


def extract_na_translation(genbank_path: Path) -> tuple[str, dict[str, str]]:
    record = SeqIO.read(str(genbank_path), "genbank")
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for feature in record.features:
        if feature.type != "CDS":
            continue
        translation = _feature_text(feature, "translation").replace(" ", "")
        if not translation:
            continue
        product = _feature_text(feature, "product")
        gene = _feature_text(feature, "gene")
        note = _feature_text(feature, "note")
        text = f"{product} {gene} {note}".lower()
        score = 0
        if "neuraminidase" in text:
            score += 100
        if gene.upper() == "NA":
            score += 50
        if "NB" in gene.upper().split():
            score -= 50
        score += len(translation)
        candidates.append(
            (
                score,
                translation,
                {
                    "record_id": record.id,
                    "record_name": record.name,
                    "description": record.description,
                    "product": product,
                    "gene": gene,
                    "protein_id": _feature_text(feature, "protein_id"),
                    "translation_length": str(len(translation)),
                },
            )
        )
    if not candidates:
        raise RuntimeError(f"no translated CDS found in {genbank_path}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def write_reference_files(spec: SubtypeSpec) -> dict[str, str]:
    gb_path = REF_DIR / f"{spec.stem}_{spec.accession}.gb"
    fetch_genbank(spec.accession, gb_path)
    sequence, info = extract_na_translation(gb_path)

    fasta_path = REF_DIR / f"{spec.stem}_na_reference_{spec.accession}.fasta"
    header = (
        f"{spec.stem.upper()}_NA_REF|{spec.accession}|{info.get('protein_id', '')}|"
        f"{info.get('product', 'neuraminidase')}"
    )
    with fasta_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f">{header}\n")
        for i in range(0, len(sequence), 80):
            handle.write(sequence[i : i + 80] + "\n")

    info = dict(info)
    info.update(
        {
            "subtype": spec.subtype,
            "accession": spec.accession,
            "genbank_path": str(gb_path),
            "reference_fasta": str(fasta_path),
        }
    )
    print(
        f"[ref] {spec.subtype} {spec.accession} "
        f"translation_length={len(sequence)} product={info.get('product', '')!r}"
    )
    return info


def prepare_query_fasta(spec: SubtypeSpec) -> dict[str, object]:
    if not spec.source_fasta.exists():
        raise FileNotFoundError(spec.source_fasta)
    output_path = INPUT_DIR / f"{spec.stem}_na_queries_nonempty.fasta"
    counts: Counter[int] = Counter()
    source_records = 0
    written_records = 0
    skipped_empty = 0

    with output_path.open("w", encoding="utf-8", newline="") as out:
        for record in SeqIO.parse(str(spec.source_fasta), "fasta"):
            source_records += 1
            seq = str(record.seq).replace(" ", "").replace("\n", "").upper()
            counts[len(seq)] += 1
            if not seq:
                skipped_empty += 1
                continue
            record.seq = type(record.seq)(seq)
            record.description = record.id
            SeqIO.write(record, out, "fasta")
            written_records += 1

    top_lengths = ";".join(f"{length}:{count}" for length, count in counts.most_common(10))
    print(
        f"[input] {spec.subtype} source={source_records} nonempty={written_records} "
        f"empty={skipped_empty} top_lengths={top_lengths}"
    )
    return {
        "subtype": spec.subtype,
        "source_fasta": str(spec.source_fasta),
        "query_fasta": str(output_path),
        "source_records": source_records,
        "query_records": written_records,
        "skipped_empty_records": skipped_empty,
        "source_length_distribution_top10": top_lengths,
    }


def count_fasta_records(path: Path) -> tuple[int, Counter[int]]:
    counts: Counter[int] = Counter()
    total = 0
    for record in SeqIO.parse(str(path), "fasta"):
        total += 1
        counts[len(str(record.seq))] += 1
    return total, counts


def run_mafft(spec: SubtypeSpec, threads: int, force: bool) -> dict[str, object]:
    ref_fasta = REF_DIR / f"{spec.stem}_na_reference_{spec.accession}.fasta"
    query_fasta = INPUT_DIR / f"{spec.stem}_na_queries_nonempty.fasta"
    output_fasta = ALIGNED_DIR / f"{spec.stem}_na_aligned_to_{spec.accession}.fasta"
    log_path = LOG_DIR / f"{spec.stem}_mafft.log"
    if output_fasta.exists() and output_fasta.stat().st_size > 0 and not force:
        print(f"[mafft] reuse {output_fasta}")
    else:
        cmd = [
            "mafft",
            "--thread",
            str(threads),
            "--keeplength",
            "--addfragments",
            str(query_fasta),
            str(ref_fasta),
        ]
        print(f"[mafft] {' '.join(cmd)}")
        start = time.time()
        with output_fasta.open("w", encoding="utf-8", newline="") as stdout, log_path.open(
            "w", encoding="utf-8", newline=""
        ) as stderr:
            subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)
        print(f"[mafft] wrote {output_fasta} elapsed={time.time() - start:.1f}s")

    total, length_counts = count_fasta_records(output_fasta)
    top_lengths = ";".join(f"{length}:{count}" for length, count in length_counts.most_common(10))
    return {
        "subtype": spec.subtype,
        "aligned_fasta": str(output_fasta),
        "mafft_log": str(log_path),
        "aligned_records": total,
        "aligned_length_distribution_top10": top_lengths,
    }


def selected_specs(names: Iterable[str]) -> list[SubtypeSpec]:
    specs = []
    for name in names:
        if name not in SUBTYPES:
            choices = ", ".join(SUBTYPES)
            raise ValueError(f"unknown subtype {name!r}; choose from {choices}")
        specs.append(SUBTYPES[name])
    return specs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subtypes",
        nargs="+",
        default=list(SUBTYPES),
        choices=list(SUBTYPES),
    )
    parser.add_argument(
        "--stage",
        choices=["references", "inputs", "align", "all"],
        default="all",
    )
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate MAFFT outputs even if aligned FASTA files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    specs = selected_specs(args.subtypes)

    ref_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    align_rows: list[dict[str, object]] = []

    if args.stage in {"references", "all"}:
        for spec in specs:
            ref_rows.append(write_reference_files(spec))
        write_csv(EXP_ROOT / "data" / "na_reference_summary.csv", ref_rows)

    if args.stage in {"inputs", "all"}:
        for spec in specs:
            input_rows.append(prepare_query_fasta(spec))
        write_csv(EXP_ROOT / "data" / "na_alignment_input_summary.csv", input_rows)

    if args.stage in {"align", "all"}:
        for spec in specs:
            for path in (
                REF_DIR / f"{spec.stem}_na_reference_{spec.accession}.fasta",
                INPUT_DIR / f"{spec.stem}_na_queries_nonempty.fasta",
            ):
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing; run --stage all or prior stages first")
            align_rows.append(run_mafft(spec, threads=args.threads, force=args.force))
        write_csv(EXP_ROOT / "data" / "na_alignment_summary.csv", align_rows)


if __name__ == "__main__":
    main()
