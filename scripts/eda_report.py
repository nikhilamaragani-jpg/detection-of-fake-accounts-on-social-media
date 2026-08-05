"""Generate a text EDA report for Data Analyst interviews."""

from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from preprocess import get_dataset  # noqa: E402


def main() -> None:
    df = get_dataset(prefer_csv=True)
    lines = []
    lines.append("# EDA Report — Fake Account Dataset\n")
    lines.append(f"Rows: {len(df)}\n")
    lines.append("## Class balance\n")
    lines.append(df["is_fake"].value_counts().to_string())
    lines.append("\n\n## Describe\n")
    lines.append(df.describe().to_string())
    lines.append("\n\n## Mean features by label\n")
    lines.append(df.groupby("is_fake").mean(numeric_only=True).to_string())
    lines.append("\n\n## Business insight\n")
    lines.append(
        "- Fake-labeled rows tend to show low account age, high following, weak profile assets.\n"
        "- Genuine-labeled rows show higher followers and completeness flags.\n"
        "- Use F1 in model selection when class mix is uneven.\n"
    )
    out = os.path.join(ROOT, "data", "outputs", "eda_report.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
