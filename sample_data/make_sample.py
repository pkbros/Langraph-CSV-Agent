"""Regenerate sample_data/sales.csv deterministically."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

REGIONS = ["West", "East", "North", "South"]
PRODUCTS = {"Widget": 100.0, "Gadget": 130.0, "Doohickey": 75.0, "Gizmo": 210.0}
REPS = ["Ana", "Ben", "Cara", "Dev", "Eli", "Fay"]


def main() -> None:
    rng = random.Random(7)
    rows = []
    for month in range(1, 7):
        for region in REGIONS:
            for product, price in PRODUCTS.items():
                units = rng.randint(1, 12)
                if region == "West":
                    units = int(units * 1.9)
                elif region == "East":
                    units = int(units * 1.15)
                rows.append(
                    {
                        "date": f"2024-{month:02d}-{rng.randint(1, 28):02d}",
                        "region": region,
                        "product": product,
                        "rep": rng.choice(REPS),
                        "units": units,
                        "sales": round(units * price, 2),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["date", "region", "product"]).reset_index(drop=True)
    out = Path(__file__).with_name("sales.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    print(df.groupby("region")["sales"].sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
