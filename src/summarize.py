"""Aggregate *_results.json files into a comparison table."""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments")
    p.add_argument("--pattern", default="*_results.json")
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "**", args.pattern),
                                  recursive=True)):
        with open(path) as f:
            r = json.load(f)
        c = r.get("config", {})
        rows.append({
            "run": r.get("run_name"),
            "fold": r.get("fold"),
            "val_acc": round(r.get("best_val_acc", 0.0), 5),
            "best_epoch": r.get("best_epoch"),
            "ema": r.get("used_ema"),
            "params": r.get("params"),
            "size": c.get("image_size"),
            "epochs": c.get("num_epochs"),
            "opt": c.get("opt"),
            "lr": c.get("lr"),
            "ls": c.get("label_smoothing"),
            "sizes": "-".join(map(str, c.get("sizes", []))),
            "blocks": "-".join(map(str, c.get("blocks_per_stage", []))),
            "file": os.path.relpath(path, args.dir),
        })

    if not rows:
        raise SystemExit(f"No results matched {args.pattern} under {args.dir}")

    df = pd.DataFrame(rows).sort_values("val_acc", ascending=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
