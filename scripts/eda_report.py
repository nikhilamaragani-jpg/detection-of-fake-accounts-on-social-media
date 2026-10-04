"""Generate a text EDA report for Data Analyst interviews."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess import get_dataset  # noqa: E402


def build_report(df: pd.DataFrame) -> str:
    """Build a reproducible descriptive report from the supplied dataset."""
    label_counts = df["is_fake"].value_counts().reindex([0, 1], fill_value=0)
    quality = pd.DataFrame(
        {
            "missing_values": df.isna().sum(),
            "unique_values": df.nunique(dropna=False),
        }
    )
    label_mix = pd.DataFrame(
        {
            "accounts": label_counts,
            "share_percent": (label_counts / len(df) * 100).round(1),
        }
    ).rename(index={0: "genuine (0)", 1: "fake (1)"})

    return "\n".join(
        [
            "# EDA Report — Fake Account Dataset",
            "",
            "## Dataset overview",
            f"- Rows: {len(df)}",
            f"- Features: {len(df.columns) - 1}",
            "- Target: `is_fake` (1 = fake, 0 = genuine)",
            "",
            "## Data quality",
            quality.to_string(),
            f"\nDuplicate rows: {int(df.duplicated().sum())}",
            "",
            "## Class balance",
            label_mix.to_string(),
            "",
            "## Numeric summary",
            df.describe().round(3).to_string(),
            "",
            "## Median features by label",
            df.groupby("is_fake").median(numeric_only=True).round(3).to_string(),
            "",
            "## Interpretation limits",
            "- These summaries describe the supplied sample only; they do not establish "
            "causal relationships or production performance.",
            "- Confirm data provenance and label quality before using results for "
            "operational decisions.",
            "",
        ]
    )


def main() -> None:
    df = get_dataset(prefer_csv=True)
    out = os.path.join(ROOT, "data", "outputs", "eda_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(build_report(df))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
