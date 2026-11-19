import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.results).read_text())
    rows = []
    for arm, r in data.items():
        ev = r.get("eval")
        meta = r.get("meta", {})
        if ev:
            rows.append(
                {
                    "arm": arm,
                    "grammar": round(ev["grammar"]["accuracy"], 3),
                    "sv_ratio": round(ev["openended"]["swedish_ratio_mean"], 2),
                    "en_regr": round(ev["regression"]["accuracy"], 3),
                    "labels": ev["openended"]["label_counts"],
                    "changed_params": meta.get("diff", {}).get("changed_params"),
                }
            )
        else:
            rows.append(
                {
                    "arm": arm,
                    "sv_ratio": round(r["openended"]["swedish_ratio_mean"], 2),
                    "labels": r["openended"]["label_counts"],
                    "n_bias_tokens": meta.get("n_bias_tokens"),
                }
            )

    lines = ["| arm | grammar | sv-ratio | en-regr | labels | changed-params |"]
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        lines.append(
            "| {arm} | {grammar} | {sv_ratio} | {en_regr} | {labels} | {changed_params} |".format(
                grammar=row.get("grammar", "-"),
                sv_ratio=row.get("sv_ratio", "-"),
                en_regr=row.get("en_regr", "-"),
                labels=row.get("labels", "-"),
                changed_params=row.get("changed_params", "-"),
                arm=row["arm"],
            )
        )
    out = "\n".join(lines)
    print(out)
    if args.out:
        Path(args.out).write_text(out + "\n")


if __name__ == "__main__":
    main()