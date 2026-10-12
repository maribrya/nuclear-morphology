from pathlib import Path
import subprocess, sys, json
import numpy as np

root = Path(__file__).resolve().parent
before = json.loads((root / "restoration_summary.json").read_text())
subprocess.run(
    [sys.executable, str(root / "summarize_restoration.py")],
    check=True,
    stdout=subprocess.DEVNULL,
)
after = json.loads((root / "restoration_summary.json").read_text())
def compare(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            compare(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y)
    elif isinstance(a, (float, int)):
        assert np.isclose(a, b, rtol=1e-10, atol=1e-12), (a, b)
    else:
        assert a == b

compare(before, after)
rows = json.loads((root / "restoration_rows.json").read_text())
assert len(rows) == 350 and len({r["image"] for r in rows}) == 50
original = [r for r in rows if r["condition"] == "original"]
assert sum(r["gt"] for r in original) == 4816
assert sum(r["rare_n"] for r in original) == 498
print("50 fields, 350 result rows: summary and paired intervals reproduced")
