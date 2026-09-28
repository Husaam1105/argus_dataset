"""End-to-end release: build -> dirty variants -> coverage-grid headroom -> QA -> dataset card -> manifest.

    python -m argusds.release --version 1.0.0 [--window mainnet_...parquet] [--grid 0.1,0.6,1.0]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as C
from .build import sha256


def run(*args: str) -> None:
    print("[release] $", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=C.ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--window", default=None)
    ap.add_argument("--grid", default="0.1,0.6,1.0")
    a = ap.parse_args()
    rel = C.RELEASES / f"argus-dataset-v{a.version}"
    win = ["--window", a.window] if a.window else []

    run("argusds.build", "--version", a.version, *win)
    for track in ("track_r", "track_m"):
        run("argusds.dirty", "--release", str(rel), "--track", track)

    (rel / "qa").mkdir(exist_ok=True)
    main_hr = rel / "qa" / "headroom.json"
    run("argusds.headroom", "--release", str(rel), "--out", str(main_hr))
    stats = json.loads((rel / "truth" / "track_m" / "stats_realized.json").read_text())
    rows = []
    for cov in [float(x) for x in a.grid.split(",")] + [stats["sensor_peer_coverage"]]:
        if cov == stats["sensor_peer_coverage"]:
            hr = json.loads(main_hr.read_text())
        else:
            tmp = f"{a.version}-grid{int(round(cov * 100)):03d}"
            run("argusds.build", "--version", tmp, "--tracks", "m", "--formats", "parquet", "--coverage", str(cov), *win)
            out = C.RELEASES / f"argus-dataset-v{tmp}" / "headroom.json"
            run("argusds.headroom", "--release", str(out.parent), "--out", str(out))
            hr = json.loads(out.read_text())
            shutil.rmtree(out.parent)
        o = hr["overall"]
        rows.append({"coverage": cov, "cioh": o["cioh"]["f1"], "ip_hour": o["ip_hour"]["f1"],
                     "fingerprint_hubaware": o["fingerprint_hubaware"]["f1"], "dual_hubaware": o["dual_hubaware"]["f1"]})
    rows.sort(key=lambda r: r["coverage"])
    (rel / "qa" / "headroom_grid.json").write_text(json.dumps(
        {"grid": rows, "by_path": json.loads(main_hr.read_text())["by_path"]}, indent=2))

    subprocess.run([sys.executable, "-m", "argusds.qa", "--release", str(rel)], cwd=C.ROOT)   # writes its own report
    run("argusds.card", "--release", str(rel))
    shutil.copy(C.ROOT / "DATASET_HANDOFF.md", rel / "HANDOFF.md")

    m = json.loads((rel / "manifest.json").read_text())
    m["files"] = {p.relative_to(rel).as_posix(): sha256(p) for p in sorted(rel.rglob("*"))
                  if p.is_file() and p.name != "manifest.json"}
    (rel / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"[release] done: {rel} ({len(m['files'])} files)")


if __name__ == "__main__":
    main()
