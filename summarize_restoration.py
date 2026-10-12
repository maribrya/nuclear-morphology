from pathlib import Path

import json, csv

import numpy as np

ROOT = Path(__file__).resolve().parent


def aggregate(rs):
    sums = {
        k: sum(r[k] for r in rs)
        for k in [
            "gt",
            "pred",
            "tp",
            "rare_n",
            "rare_tp",
            "typical_n",
            "typical_tp",
            "rare_error_sum",
            "rare_error_n",
            "typical_error_sum",
            "typical_error_n",
        ]
    }
    return {
        **sums,
        "object_score": sums["tp"] / (sums["gt"] + sums["pred"] - sums["tp"]),
        "rare_recall": sums["rare_tp"] / sums["rare_n"],
        "typical_recall": sums["typical_tp"] / sums["typical_n"],
        "rare_mae": sums["rare_error_sum"] / sums["rare_error_n"],
        "typical_mae": sums["typical_error_sum"] / sums["typical_error_n"],
    }


def main():
    rows = json.loads((ROOT / "restoration_rows.json").read_text())
    assert len(rows) == 350
    assert len({r["image"] for r in rows}) == 50
    rng = np.random.default_rng(20261010)
    boot = rng.integers(0, 50, size=(10000, 50))
    summary = {
        "conditions": [],
        "paired_comparisons": [],
        "bootstrap_replicates": 10000,
        "bootstrap_unit": "image; paired resampling; pooled recall differences",
        "seed": 20261010,
    }
    conditions = [("original", 0)] + [(c, s) for s in [1, 2, 3] for c in ["blurred", "restored"]]
    for condition, sigma in conditions:
        rs = [r for r in rows if r["condition"] == condition and r["sigma"] == sigma]
        assert len(rs) == 50
        summary["conditions"].append(dict(condition=condition, sigma=sigma, **aggregate(rs)))
    for sigma in [1, 2, 3]:
        a = sorted(
            [r for r in rows if r["condition"] == "blurred" and r["sigma"] == sigma],
            key=lambda r: r["image"],
        )
        b = sorted(
            [r for r in rows if r["condition"] == "restored" and r["sigma"] == sigma],
            key=lambda r: r["image"],
        )
        out = {"sigma": sigma}
        for group in ["rare", "typical"]:
            an = np.array([r[group + "_n"] for r in a])
            atp = np.array([r[group + "_tp"] for r in a])
            btp = np.array([r[group + "_tp"] for r in b])
            den = an[boot].sum(axis=1)
            delta = (btp[boot].sum(axis=1) - atp[boot].sum(axis=1)) / den
            out[group + "_recall_gain"] = float((btp.sum() - atp.sum()) / an.sum())
            out[group + "_recall_gain_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
            ae = []
            be = []
            for ra, rb in zip(a, b):
                assert ra["image"] == rb["image"]
                rare = set(map(str, ra["rare_ids"]))
                common = set(ra["matched_errors"]) & set(rb["matched_errors"])
                common = common & rare if group == "rare" else common - rare
                ae.extend(ra["matched_errors"][i] for i in common)
                be.extend(rb["matched_errors"][i] for i in common)
            out[group + "_common_matches"] = len(ae)
            out[group + "_common_blurred_mae"] = float(np.mean(ae))
            out[group + "_common_restored_mae"] = float(np.mean(be))
        summary["paired_comparisons"].append(out)
    (ROOT / "restoration_summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "restoration_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary["conditions"][0]))
        w.writeheader()
        w.writerows(summary["conditions"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
