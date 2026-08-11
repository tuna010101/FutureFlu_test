"""Convert H3N2 FASTA and metadata exports into derived workflow inputs.

English: Writes standardized metadata, amino-acid FASTA, and conversion summary files.
中文：写出标准化 metadata、氨基酸 FASTA 和转换汇总文件。
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from Bio import SeqIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGING_OUTPUT_DIR = PROJECT_ROOT / "data" / "sequences" / ".staging"

# English: Optional primary H3N2 submission files live under data/sources/.
# 中文：可选的 H3N2 原始提交文件放在 data/sources/。
INPUT_FASTA = PROJECT_ROOT / "data" / "sources" / "msa-H3N2-all-20250131-submission.fasta"
INPUT_METADATA = PROJECT_ROOT / "data" / "sources" / "H3N2-all-20250131-submission.csv"

OUTPUT_FASTA = STAGING_OUTPUT_DIR / "h3n2_futureflu_aa_sequences.fasta"
OUTPUT_METADATA = STAGING_OUTPUT_DIR / "h3n2_futureflu_metadata.tsv"
OUTPUT_SUMMARY = STAGING_OUTPUT_DIR / "h3n2_futureflu_summary.json"
OUTPUT_REPORT = PROJECT_ROOT / "results" / "futureflu" / "artifacts" / "h3n2_conversion_artifacts.md"


@contextmanager
def fasta_input(path: Path):
    """Open FASTA input. / 打开 FASTA 输入。"""
    with path.open("r", encoding="utf-8") as handle:
        yield handle


def public_path(path: Path) -> str:
    """Return a repository-relative path when possible.

    English: Summary JSON should be portable across machines.
    中文：summary JSON 使用可迁移的相对路径。
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
    text = re.sub(r"[^A-Za-z0-9._/-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def parse_location(location: str) -> dict[str, str]:
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


def normalize_date(value: str) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""

    parts = text.split("-")
    if len(parts) == 1 and parts[0].isdigit():
        text = f"{parts[0]}-01-01"
    elif len(parts) == 2 and all(part.isdigit() for part in parts):
        text = f"{parts[0]}-{parts[1]}-01"

    try:
        parsed = pd.to_datetime(text, errors="raise")
    except Exception:
        return ""

    if parsed.year < 1990 or parsed.year > 2100:
        return ""

    return parsed.strftime("%Y-%m-%d")


def normalize_host_age(age_value: str, unit_value: str) -> str:
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


def normalize_host_gender(value: str) -> str:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return "?"
    if text in {"f", "female"}:
        return "female"
    if text in {"m", "male"}:
        return "male"
    return text


def build_metadata_lookup(metadata: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for _, row in metadata.iterrows():
        isolate_id = str(row.get("Isolate_Id", "")).strip()
        if isolate_id:
            lookup[f"isolate::{isolate_id}"] = row.to_dict()

        ha_segment = str(row.get("HA Segment_Id", "")).strip()
        if ha_segment:
            ha_prefix = ha_segment.split("|")[0].strip()
            if ha_prefix:
                lookup[f"ha::{ha_prefix}"] = row.to_dict()

    return lookup


def main() -> None:
    metadata = pd.read_csv(INPUT_METADATA, dtype=str, low_memory=False)
    metadata_lookup = build_metadata_lookup(metadata)

    output_records = []
    seen_strains: dict[str, int] = {}
    fasta_records = []
    missing = []
    invalid_dates = 0

    with fasta_input(INPUT_FASTA) as fasta_handle:
        input_records = SeqIO.parse(fasta_handle, "fasta")
        for record in input_records:
            if record.id == "H3N2_reference":
                continue

            parts = record.id.split("|")
            isolate_id = parts[0].strip() if len(parts) > 0 else ""
            ha_segment_id = parts[1].strip() if len(parts) > 1 else ""

            row = metadata_lookup.get(f"isolate::{isolate_id}") or metadata_lookup.get(f"ha::{ha_segment_id}")
            if row is None:
                missing.append(record.id)
                continue

            isolate_name = str(row.get("Isolate_Name", isolate_id)).strip()
            base_name = sanitize_token(f"{isolate_name}__{ha_segment_id or isolate_id}")
            seen_strains[base_name] = seen_strains.get(base_name, 0) + 1
            strain = base_name if seen_strains[base_name] == 1 else f"{base_name}__dup{seen_strains[base_name]}"

            location_info = parse_location(row.get("Location", "unknown"))
            collection_date = normalize_date(row.get("Collection_Date", ""))
            submission_date = normalize_date(row.get("Submission_Date", ""))
            if not collection_date:
                invalid_dates += 1
                continue

            if not submission_date:
                submission_date = collection_date

            passage = str(row.get("Passage_History", "undetermined")).strip() or "undetermined"
            clade = str(row.get("Clade", "")).strip()
            age = normalize_host_age(row.get("Host_Age", ""), row.get("Host_Age_Unit", "Y"))
            gender = normalize_host_gender(row.get("Host_Gender", ""))

            output_records.append(
                {
                    "strain": strain,
                    "accession": isolate_id,
                    "age": age,
                    "collection_date": collection_date,
                    "date": collection_date,
                    "submission_date": submission_date,
                    "gender": gender,
                    "region": location_info["region"],
                    "country": location_info["country"],
                    "division": location_info["division"],
                    "location": location_info["location"],
                    "passage": passage,
                    "submitting_lab": str(row.get("Submitting_Lab", "")).strip(),
                    "clade": clade,
                    "virus": "flu",
                    "lineage": "h3n2",
                    "segment": "ha",
                    "source_isolate_id": isolate_id,
                    "source_ha_segment_id": ha_segment_id,
                    "source_fasta_id": record.id,
                    "sequence_type": "amino_acid",
                }
            )

            record.id = strain
            record.name = strain
            record.description = strain
            fasta_records.append(record)

    output_df = pd.DataFrame(output_records)
    output_df = output_df.sort_values(["collection_date", "strain"]).reset_index(drop=True)

    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)

    SeqIO.write(fasta_records, str(OUTPUT_FASTA), "fasta")
    output_df.to_csv(OUTPUT_METADATA, sep="\t", index=False)

    summary = {
        "input_fasta": public_path(INPUT_FASTA),
        "input_metadata": public_path(INPUT_METADATA),
        "output_fasta": public_path(OUTPUT_FASTA),
        "output_metadata": public_path(OUTPUT_METADATA),
        "input_sequence_type": "amino_acid",
        "records_written": int(output_df.shape[0]),
        "missing_metadata_records": int(len(missing)),
        "invalid_collection_dates_dropped": int(invalid_dates),
        "date_min": str(output_df["collection_date"].min()) if not output_df.empty else None,
        "date_max": str(output_df["collection_date"].max()) if not output_df.empty else None,
        "regions": sorted(output_df["region"].dropna().astype(str).unique().tolist()) if not output_df.empty else [],
        "countries": int(output_df["country"].nunique()) if not output_df.empty else 0,
        "clades": int(output_df["clade"].replace("", None).dropna().nunique()) if not output_df.empty else 0,
        "age_known": int((output_df["age"] != "?").sum()) if not output_df.empty else 0,
        "gender_known": int((output_df["gender"] != "?").sum()) if not output_df.empty else 0,
    }

    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# H3N2 Conversion Artifacts

## Command

```bash
python scripts/futureflu/convert_futureflu_h3n2_to_flu_forecasting.py
```

## Outputs

- `{OUTPUT_FASTA.relative_to(PROJECT_ROOT)}`
- `{OUTPUT_METADATA.relative_to(PROJECT_ROOT)}`
- `{OUTPUT_SUMMARY.relative_to(PROJECT_ROOT)}`

## Summary

- records written: {summary['records_written']}
- missing metadata matches: {summary['missing_metadata_records']}
- invalid collection dates dropped: {summary['invalid_collection_dates_dropped']}
- collection date min: {summary['date_min']}
- collection date max: {summary['date_max']}
- age known: {summary['age_known']}
- gender known: {summary['gender_known']}
- region buckets: {", ".join(summary['regions']) if summary['regions'] else "none"}
- unique countries: {summary['countries']}
- unique non-empty clades: {summary['clades']}

## Sequence Type

The generated FASTA contains aligned amino-acid HA sequences.
"""
    OUTPUT_REPORT.write_text(report, encoding="utf-8")

    if missing:
        print(f"WARNING: {len(missing)} FASTA records had no metadata match")

    print(f"Wrote {OUTPUT_FASTA}")
    print(f"Wrote {OUTPUT_METADATA}")
    print(f"Wrote {OUTPUT_SUMMARY}")
    print(f"Wrote {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
