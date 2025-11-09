from __future__ import annotations

from pathlib import Path


def write_manifests(df, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metadata_split.csv", index=False)
    for split_name in ["train", "val", "test"]:
        sub_df = df[df["Split"] == split_name]
        sub_df.to_csv(out / f"{split_name}.csv", index=False)

