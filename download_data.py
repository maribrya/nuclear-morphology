from pathlib import Path

from urllib.request import urlopen

from hashlib import sha256

import json


def main():
    root = Path(__file__).resolve().parent
    data = root / "bbbc039"
    data.mkdir(exist_ok=True)
    data = root / "bbbc039"
    data.mkdir(exist_ok=True)
    expected = json.loads((root / "morphology_audit.json").read_text())["archive_sha256"]
    for name in ["images.zip", "masks.zip", "metadata.zip"]:
        p = data / name
        if not p.exists():
            p.write_bytes(
                urlopen("https://data.broadinstitute.org/bbbc/BBBC039/" + name, timeout=180).read()
            )
        assert sha256(p.read_bytes()).hexdigest() == expected[name], name
        print(name, "verified")


if __name__ == "__main__":
    main()
