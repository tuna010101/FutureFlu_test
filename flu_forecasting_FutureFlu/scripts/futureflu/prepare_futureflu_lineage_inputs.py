"""Prepare lineage-level FutureFlu derived inputs.

English: Builds standardized metadata, amino-acid FASTA, sequence tables, and distance-map scaffolds.
中文：构建标准化 metadata、氨基酸 FASTA、sequence table 和 distance-map 结构。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUTUREFLU_DATA_ROOT = PROJECT_ROOT / "data" / "futureflu"
FUTUREFLU_CONFIG_ROOT = PROJECT_ROOT / "config" / "futureflu"
FUTUREFLU_RESULTS_ROOT = PROJECT_ROOT / "results" / "futureflu"


def public_path(path: Path) -> str:
    """Return a repository-relative path when possible.

    English: Derived-input summaries should be portable across machines.
    中文：derived 输入摘要使用可迁移的仓库相对路径。
    """
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(PROJECT_ROOT.parent))
        except ValueError:
            return str(path)


ASIA_BUCKETS = {
    "china": "china",
    "hong kong": "china",
    "hong kong (sar)": "china",
    "hong_kong": "china",
    "taiwan": "china",
    "japan": "japan_korea",
    "south korea": "japan_korea",
    "korea": "japan_korea",
    "india": "south_asia",
    "pakistan": "south_asia",
    "bangladesh": "south_asia",
    "sri lanka": "south_asia",
    "nepal": "south_asia",
    "bhutan": "south_asia",
    "maldives": "south_asia",
    "indonesia": "southeast_asia",
    "thailand": "southeast_asia",
    "vietnam": "southeast_asia",
    "singapore": "southeast_asia",
    "malaysia": "southeast_asia",
    "philippines": "southeast_asia",
    "cambodia": "southeast_asia",
    "laos": "southeast_asia",
    "lao, people's democratic republic": "southeast_asia",
    "myanmar": "southeast_asia",
    "brunei": "southeast_asia",
    "timor-leste": "southeast_asia",
    "united arab emirates": "west_asia",
    "saudi arabia": "west_asia",
    "qatar": "west_asia",
    "oman": "west_asia",
    "kuwait": "west_asia",
    "bahrain": "west_asia",
    "iraq": "west_asia",
    "iran": "west_asia",
    "jordan": "west_asia",
    "lebananon": "west_asia",
    "lebanon": "west_asia",
    "israel": "west_asia",
    "palestine": "west_asia",
    "yemen": "west_asia",
    "syria": "west_asia",
    "turkey": "west_asia",
    "georgia": "west_asia",
    "armenia": "west_asia",
    "azerbaijan": "west_asia",
}


def sanitize_token(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def normalize_lookup_key(value: object) -> str:
    return sanitize_token(str(value).lower()).lower()


def normalize_date(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    parts = text.split("-")
    if len(parts) == 1 and parts[0].isdigit():
        text = f"{parts[0]}-01-01"
    elif len(parts) == 2 and all(part.isdigit() for part in parts):
        text = f"{parts[0]}-{parts[1]}-01"

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed) or parsed.year < 1990 or parsed.year > 2100:
        return ""
    return parsed.strftime("%Y-%m-%d")


def normalize_host_age(age_value: object, unit_value: object) -> str:
    age_text = str(age_value).strip()
    if not age_text or age_text.lower() == "nan":
        return "?"

    unit_text = str(unit_value).strip().lower()
    if unit_text in {"", "nan"}:
        unit_text = "y"
    unit_suffix = {
        "y": "y",
        "year": "y",
        "years": "y",
        "m": "m",
        "month": "m",
        "months": "m",
    }.get(unit_text, unit_text)

    try:
        age_float = float(age_text)
        age_text = str(int(age_float)) if age_float.is_integer() else str(age_float)
    except ValueError:
        pass
    return f"{age_text}{unit_suffix}"


def normalize_host_gender(value: object) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "?"
    if text in {"f", "female"}:
        return "female"
    if text in {"m", "male"}:
        return "male"
    return text


def parse_location(location: object) -> dict[str, str]:
    parts = [part.strip() for part in str(location).split("/") if part.strip()]
    continent = parts[0] if len(parts) > 0 else "unknown"
    country = parts[1] if len(parts) > 1 else "unknown"
    division = parts[2] if len(parts) > 2 else "unknown"
    locality = parts[3] if len(parts) > 3 else division

    continent_key = continent.lower()
    country_key = country.lower()
    if continent_key == "europe":
        region = "europe"
    elif continent_key == "africa":
        region = "africa"
    elif continent_key == "north america":
        region = "north_america"
    elif continent_key == "south america":
        region = "south_america"
    elif continent_key == "oceania":
        region = "oceania"
    elif continent_key == "asia":
        region = ASIA_BUCKETS.get(country_key, "west_asia")
    else:
        region = sanitize_token(continent.lower())

    return {
        "region": region,
        "country": country,
        "division": division,
        "location": locality,
    }


def build_metadata_lookup(metadata: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for _, row in metadata.iterrows():
        row_dict = row.to_dict()
        isolate_id = str(row.get("Isolate_Id", "")).strip()
        if isolate_id and isolate_id.lower() != "nan":
            lookup[f"isolate::{normalize_lookup_key(isolate_id)}"] = row_dict

        ha_segment = str(row.get("HA Segment_Id", "")).strip()
        if ha_segment and ha_segment.lower() != "nan":
            lookup[f"ha::{normalize_lookup_key(ha_segment)}"] = row_dict
            ha_prefix = ha_segment.split("|")[0].strip()
            if ha_prefix:
                lookup[f"ha::{normalize_lookup_key(ha_prefix)}"] = row_dict

        isolate_name = str(row.get("Isolate_Name", "")).strip()
        if isolate_name and isolate_name.lower() != "nan":
            lookup[f"name::{normalize_lookup_key(isolate_name)}"] = row_dict

    return lookup


def find_metadata_for_record(record_id: str, lookup: dict[str, dict]) -> tuple[dict | None, str, str]:
    parts = [part.strip() for part in record_id.split("|")]
    candidates: list[tuple[str, str]] = []
    if len(parts) > 0:
        candidates.extend(
            [
                ("isolate", parts[0]),
                ("name", parts[0]),
                ("ha", parts[0]),
            ]
        )
    if len(parts) > 1:
        candidates.extend(
            [
                ("isolate", parts[1]),
                ("ha", parts[1]),
                ("name", parts[1]),
            ]
        )

    for kind, value in candidates:
        row = lookup.get(f"{kind}::{normalize_lookup_key(value)}")
        if row is not None:
            isolate_id = str(row.get("Isolate_Id", "")).strip()
            ha_segment = str(row.get("HA Segment_Id", "")).strip()
            return row, isolate_id, ha_segment

    return None, "", ""


def parse_epitope_sites(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8").strip()
    sites = sorted({int(token.strip()) for token in text.replace("\n", ",").split(",") if token.strip()})
    if not sites:
        raise ValueError(f"No epitope sites found in {path}")
    return sites


def write_distance_maps(epitope_sites: list[int], output_dir: Path, ha1_length: int) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    epitope_set = set(epitope_sites)
    nonepitope_sites = [site for site in range(1, ha1_length + 1) if site not in epitope_set]

    maps = {
        "luksza.json": {
            "default": 0,
            "map": {"HA1": {str(site): 1 for site in epitope_sites}},
            "name": "lineage_epitope",
        },
        "luksza_nonepitope.json": {
            "default": 0,
            "map": {"HA1": {str(site): 1 for site in nonepitope_sites}},
            "name": "lineage_nonepitope",
        },
        "wolf.json": {
            "default": 0,
            "map": {"HA1": {str(site): 1 for site in epitope_sites}},
            "name": "lineage_epitope",
        },
        "koel.json": {
            "default": 0,
            "map": {"HA1": {str(site): 1 for site in epitope_sites}},
            "name": "lineage_epitope",
        },
        "oracle.json": {
            "default": 0,
            "map": {"HA1": {str(site): 1 for site in epitope_sites}},
            "name": "lineage_epitope",
        },
    }

    outputs = {}
    for filename, payload in maps.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=4, sort_keys=True), encoding="utf-8")
        outputs[filename] = public_path(path)
    return outputs


def prepare_inputs(
    input_fasta: Path,
    input_metadata: Path,
    output_prefix: str,
    lineage: str,
    epitope_file: Path,
    ha1_length: int,
    max_gaps: int,
) -> dict:
    metadata = pd.read_csv(input_metadata, dtype=str, low_memory=False)
    lookup = build_metadata_lookup(metadata)

    output_records = []
    sequence_rows = []
    fasta_records = []
    seen_strains: dict[str, int] = {}
    missing_metadata = 0
    invalid_dates = 0
    too_many_gaps = 0

    for record in SeqIO.parse(str(input_fasta), "fasta"):
        sequence = str(record.seq)
        if sequence.count("-") > max_gaps:
            too_many_gaps += 1
            continue

        row, isolate_id, ha_segment_id = find_metadata_for_record(record.id, lookup)
        if row is None:
            missing_metadata += 1
            continue

        collection_date = normalize_date(row.get("Collection_Date", ""))
        submission_date = normalize_date(row.get("Submission_Date", ""))
        if not collection_date:
            invalid_dates += 1
            continue
        if not submission_date:
            submission_date = collection_date

        isolate_name = str(row.get("Isolate_Name", isolate_id or record.id)).strip()
        unique_id = isolate_id or ha_segment_id or record.id
        base_name = sanitize_token(f"{isolate_name}__{unique_id}")
        seen_strains[base_name] = seen_strains.get(base_name, 0) + 1
        strain = base_name if seen_strains[base_name] == 1 else f"{base_name}__dup{seen_strains[base_name]}"

        location_info = parse_location(row.get("Location", "unknown"))
        clade = str(row.get("Clade", "")).strip()

        output_records.append(
            {
                "strain": strain,
                "accession": isolate_id,
                "age": normalize_host_age(row.get("Host_Age", ""), row.get("Host_Age_Unit", "Y")),
                "collection_date": collection_date,
                "date": collection_date,
                "submission_date": submission_date,
                "gender": normalize_host_gender(row.get("Host_Gender", "")),
                "region": location_info["region"],
                "country": location_info["country"],
                "division": location_info["division"],
                "location": location_info["location"],
                "passage": str(row.get("Passage_History", "undetermined")).strip() or "undetermined",
                "submitting_lab": str(row.get("Submitting_Lab", "")).strip(),
                "clade": clade,
                "virus": "flu",
                "lineage": lineage,
                "segment": "ha",
                "source_isolate_id": isolate_id,
                "source_ha_segment_id": ha_segment_id,
                "source_fasta_id": record.id,
                "sequence_type": "amino_acid",
            }
        )

        sequence_row = {
            "accession_number": isolate_id,
            "name": isolate_name,
            "clade": clade,
            "collection_date": collection_date,
            "submission_date": submission_date,
            "season": pd.Timestamp(collection_date).year - 1
            if pd.Timestamp(collection_date).month < 2
            else pd.Timestamp(collection_date).year,
        }
        for index, aa in enumerate(sequence, start=1):
            sequence_row[f"X{index}"] = aa
        sequence_rows.append(sequence_row)

        record.id = strain
        record.name = strain
        record.description = strain
        fasta_records.append(record)

    output_df = pd.DataFrame(output_records).sort_values(["collection_date", "strain"]).reset_index(drop=True)
    sequence_df = pd.DataFrame(sequence_rows)

    derived_dir = FUTUREFLU_DATA_ROOT / "derived"
    map_dir = FUTUREFLU_CONFIG_ROOT / "distance_maps" / output_prefix / "ha"
    output_fasta = derived_dir / f"{output_prefix}_futureflu_aa_sequences.fasta"
    output_metadata = derived_dir / f"{output_prefix}_futureflu_metadata.tsv"
    output_summary = derived_dir / f"{output_prefix}_futureflu_summary.json"
    output_sequence_tsv = derived_dir / f"{output_prefix}_sequence_table.tsv"
    output_sequence_pkl = derived_dir / f"{output_prefix}_sequence_table.pkl"
    output_report = FUTUREFLU_RESULTS_ROOT / "artifacts" / f"{output_prefix}_input_preparation.md"

    derived_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_metadata, sep="\t", index=False)
    SeqIO.write(fasta_records, str(output_fasta), "fasta")
    sequence_df.to_csv(output_sequence_tsv, sep="\t", index=False)
    sequence_df.to_pickle(output_sequence_pkl)

    epitope_sites = parse_epitope_sites(epitope_file)
    distance_maps = write_distance_maps(epitope_sites, map_dir, ha1_length)

    summary = {
        "lineage": lineage,
        "output_prefix": output_prefix,
        "input_fasta": public_path(input_fasta),
        "input_metadata": public_path(input_metadata),
        "epitope_file": public_path(epitope_file),
        "ha1_length": ha1_length,
        "max_gaps": max_gaps,
        "rows": len(output_df),
        "sequence_rows": len(sequence_df),
        "collection_date_min": output_df["collection_date"].min() if len(output_df) else None,
        "collection_date_max": output_df["collection_date"].max() if len(output_df) else None,
        "missing_metadata": missing_metadata,
        "invalid_dates": invalid_dates,
        "too_many_gaps": too_many_gaps,
        "outputs": {
            "metadata": public_path(output_metadata),
            "fasta": public_path(output_fasta),
            "sequence_table_tsv": public_path(output_sequence_tsv),
            "sequence_table_pkl": public_path(output_sequence_pkl),
            "summary": public_path(output_summary),
            "distance_maps": distance_maps,
        },
    }
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_report.write_text(
        "\n".join(
            [
                f"# {output_prefix} FutureFlu Input Preparation",
                "",
                "## Outputs",
                "",
                f"- metadata: `{output_metadata.relative_to(PROJECT_ROOT)}`",
                f"- amino-acid FASTA: `{output_fasta.relative_to(PROJECT_ROOT)}`",
                f"- sequence table: `{output_sequence_tsv.relative_to(PROJECT_ROOT)}`",
                f"- distance map directory: `{map_dir.relative_to(PROJECT_ROOT)}`",
                "",
                "## Counts",
                "",
                f"- retained rows: {len(output_df)}",
                f"- missing metadata records: {missing_metadata}",
                f"- invalid date records: {invalid_dates}",
                f"- records with >{max_gaps} gaps: {too_many_gaps}",
                f"- collection date range: {summary['collection_date_min']} to {summary['collection_date_max']}",
                "",
                "## Epitope Map",
                "",
                f"- epitope sites: {','.join(map(str, epitope_sites))}",
                f"- HA1 length for nonepitope map: {ha1_length}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare generic FutureFlu lineage inputs for the FutureFlu issue-date pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--input-metadata", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--epitope-file", type=Path, required=True)
    parser.add_argument("--ha1-length", type=int, default=329)
    parser.add_argument("--max-gaps", type=int, default=3)
    args = parser.parse_args()

    summary = prepare_inputs(
        input_fasta=args.input_fasta,
        input_metadata=args.input_metadata,
        output_prefix=args.output_prefix,
        lineage=args.lineage,
        epitope_file=args.epitope_file,
        ha1_length=args.ha1_length,
        max_gaps=args.max_gaps,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
