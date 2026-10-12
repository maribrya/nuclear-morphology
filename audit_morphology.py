import json, hashlib, io

from pathlib import Path

from zipfile import ZipFile

from PIL import Image

import numpy as np

ROOT = Path(__file__).resolve().parent


def objects(a):
    # Equal-valued, eight-connected components, matching the official decoder.
    parent = []
    runs = []
    previous = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    h, w = a.shape
    for y, row in enumerate(a):
        boundaries = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, w]
        current = []
        p = 0
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            value = int(row[start])
            if not value:
                continue
            end = int(stop) - 1
            start = int(start)
            idx = len(parent)
            parent.append(idx)
            runs.append((start, end, y))
            current.append((start, end, value, idx))
            while p < len(previous) and previous[p][1] < start - 1:
                p += 1
            q = p
            while q < len(previous) and previous[q][0] <= end + 1:
                if previous[q][2] == value:
                    union(idx, previous[q][3])
                q += 1
        previous = current
    stats = {}
    for i, (x0, x1, y) in enumerate(runs):
        r = find(i)
        n = x1 - x0 + 1
        sx = (x0 + x1) * n / 2
        sx2 = x1 * (x1 + 1) * (2 * x1 + 1) / 6 - (x0 - 1) * x0 * (2 * x0 - 1) / 6
        if r not in stats:
            stats[r] = [0, 0.0, 0.0, 0.0, 0.0, 0.0, False]
        s = stats[r]
        for j, v in enumerate([n, sx, n * y, sx2, n * y * y, sx * y]):
            s[j] += v
        s[6] = s[6] or x0 == 0 or x1 == w - 1 or y == 0 or y == h - 1
    out = []
    for n, sx, sy, sxx, syy, sxy, border in stats.values():
        cov = np.array(
            [
                [sxx / n - (sx / n) ** 2, sxy / n - sx * sy / n**2],
                [sxy / n - sx * sy / n**2, syy / n - (sy / n) ** 2],
            ]
        )
        eig = np.maximum(np.linalg.eigvalsh(cov), 0)
        eccentricity = float(np.sqrt(max(0, 1 - eig[0] / eig[1]))) if eig[1] > 0 else 0.0
        out.append({"area": int(n), "eccentricity": eccentricity, "border": bool(border)})
    return out


def main():
    with ZipFile(ROOT / "bbbc039/metadata.zip") as meta, ZipFile(
        ROOT / "bbbc039/masks.zip"
    ) as masks:
        groups = {
            s: meta.read("metadata/" + s + ".txt").decode().splitlines()
            for s in ["training", "validation", "test"]
        }
        assert all(
            not (set(groups[a]) & set(groups[b]))
            for a, b in [("training", "test"), ("training", "validation"), ("test", "validation")]
        )
        data = {}
        for split, names in groups.items():
            items = []
            for name in names:
                arr = np.asarray(Image.open(io.BytesIO(masks.read("masks/" + name))))
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                obs = objects(arr)
                items.extend(obs)
            data[split] = items
            print(split, len(names), "images", len(items), "nuclei", flush=True)
        train = [o for o in data["training"] if not o["border"]]
        threshold = float(np.quantile([o["eccentricity"] for o in train], 0.90))
        summary = {
            "definition": "eccentricity >= training 90th percentile; nuclei touching image boundary excluded",
            "connectivity": 8,
            "threshold": threshold,
            "splits": {},
            "note": "Reference-annotation audit only; restoration results are reported separately in restoration_summary.json.",
        }
        for s, items in data.items():
            complete = [o for o in items if not o["border"]]
            summary["splits"][s] = {
                "images": len(groups[s]),
                "nuclei_total": len(items),
                "nuclei_complete": len(complete),
                "rare_count": sum(o["eccentricity"] >= threshold for o in complete),
                "median_area": float(np.median([o["area"] for o in complete])),
                "median_eccentricity": float(np.median([o["eccentricity"] for o in complete])),
            }
        summary["archive_sha256"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "bbbc039").glob("*.zip")
        }
        (ROOT / "morphology_audit.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
