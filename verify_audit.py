import io

from zipfile import ZipFile

from PIL import Image

import numpy as np

from audit_morphology import objects, ROOT


def flood(a):
    a = a.copy()
    out = []
    for y, x in zip(*np.nonzero(a)):
        v = a[y, x]
        if not v:
            continue
        q = [(int(y), int(x))]
        a[y, x] = 0
        pts = []
        while q:
            yy, xx = q.pop()
            pts.append((yy, xx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < a.shape[0] and 0 <= nx < a.shape[1] and a[ny, nx] == v:
                        a[ny, nx] = 0
                        q.append((ny, nx))
        pts = np.array(pts)
        cov = np.cov(pts.T, bias=True) if len(pts) > 1 else np.zeros((2, 2))
        ev = np.maximum(np.linalg.eigvalsh(cov), 0)
        ecc = np.sqrt(1 - ev[0] / ev[1]) if ev[1] > 0 else 0
        out.append((len(pts), float(ecc)))
    return sorted(out)


def main():
    with ZipFile(ROOT / "bbbc039/masks.zip") as z:
        names = sorted(n for n in z.namelist() if n.startswith("masks/") and n.endswith(".png"))
        for n in (names[0], names[len(names) // 2], names[-1]):
            a = np.asarray(Image.open(io.BytesIO(z.read(n))))[:, :, 0]
            fast = sorted((r["area"], r["eccentricity"]) for r in objects(a))
            ref = flood(a)
            assert np.allclose(np.asarray(fast), np.asarray(ref), atol=1e-9)
            assert sum(r["area"] for r in objects(a)) == np.count_nonzero(a)
            print("Verified:", n, len(fast))


if __name__ == "__main__":
    main()
