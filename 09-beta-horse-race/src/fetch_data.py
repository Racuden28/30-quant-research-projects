"""
Download the three Ken French Data Library files this project uses into data/raw/ and rebuild
data/panel_daily.csv.gz.

The committed panel was built from the CRSP 202609 vintage. French revises history every month, so a
fresh download can differ slightly; the SHA-256 of the files used here are listed below and the
script reports whether your download matches them.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
VINTAGE_202609 = {
    "49_Industry_Portfolios_daily_CSV.zip": "b9d36d571e968d71f7b03207e7fed6398554f953719b8c1bf3ee3ad8d00e7520",
    "49_Industry_Portfolios_CSV.zip": "d1a3b0b9896e1e633ed2aa85e24ca144d7998f8ab2ff344e2379383ebeb7b836",
    "F-F_Research_Data_Factors_CSV.zip": "b03d39e4bb253dda0f14893db3b78e64b40b8d6549a0fc0fd4111dae84c98618",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, sha in VINTAGE_202609.items():
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "research-script"})
        blob = urllib.request.urlopen(req, timeout=120).read()
        (RAW / name).write_bytes(blob)
        got = hashlib.sha256(blob).hexdigest()
        print(f"{name}: {len(blob):,} bytes, {'matches vintage 202609' if got == sha else 'NEWER VINTAGE ' + got[:12]}")
    import data
    p = data.build_panel()
    p.to_csv(data.PANEL, float_format="%.7g")
    data.parse_section(data._read_zip_text(RAW / "49_Industry_Portfolios_CSV.zip"), "Average Firm Size", 6).to_csv(
        data.SIZE, float_format="%.2f")
    print("panel", p.shape)
