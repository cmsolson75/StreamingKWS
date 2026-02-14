import json
from pathlib import Path

import pandas as pd
import plotly.express as px

def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
        

    df = pd.DataFrame(rows)

    if "split" not in df:
        df["split"] = "train"
    
    df["step"] = pd.to_numeric(df['step'], errors="coerce")
    df = df.dropna(subset=["step"]).sort_values("step")

    for c in df.columns:
        if c not in ("steps", "split"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def main(ckpt_path: str, out: str = "metrics.html"):
    ckpt_path = Path(ckpt_path)
    df = load_jsonl(ckpt_path / "metrics.jsonl")

    metrics_cols = [
        c for c in df.columns
        if c not in ("step", "split") and df[c].dtype.kind in "fc"
    ]

    long = df.melt(
        id_vars=["step", "split"],
        value_vars=metrics_cols,
        var_name="metric",
        value_name="value"
    ).dropna()

    fig = px.line(
        long,
        x="step",
        y="value",
        color="split",
        facet_row="metric",
        markers=True,
    )

    fig.update_yaxes(matches=None)
    fig.update_layout(height=300 * len(metrics_cols))
    fig.write_html(out, include_plotlyjs="cdn")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path")
    args = parser.parse_args()

    main(args.ckpt_path)
