#!/usr/bin/env python3
"""Download the extra training packs listed in training/packs.tsv.

Each pack is extracted into <dest>/<pack name>/ (default ~/itg-packs, kept apart from
the game's Songs folder). Already extracted packs are skipped, so the script can be
re-run after an interruption. The zip is deleted once extracted.

    python3 training/fetch_packs.py [--dest ~/itg-packs] [--jobs 4] [--max-gb 10]

Packs come from the list in the ITG Packs Release Spreadsheet (4-panel pad, singles,
no gimmick packs, >= 3 singles charts per song), downloaded from the nnty.fun mirror
of stepmaniaonline.net because the spreadsheet's search.stepmaniaonline.net links are
dead.
"""

import argparse
import concurrent.futures
import csv
import pathlib
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile

LIST = pathlib.Path(__file__).with_name("packs.tsv")
MIRROR = "https://nnty.fun/downloads/game/stepmania/stepmania_packs/"


def safe_name(name):
    return "".join(c if c not in '/\\:*?"<>|' else "_" for c in name).strip(" .")


def fetch(row, dest):
    out = dest / safe_name(row["pack"])
    if (out / ".complete").exists():
        return f"skip  {row['pack']}"
    tmp = dest / (out.name + ".zip.part")
    url = MIRROR + urllib.parse.quote(row["zip"] + ".zip")
    req = urllib.request.Request(url, headers={"User-Agent": "itg-charter"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if out.exists():
        shutil.rmtree(out)
    with zipfile.ZipFile(tmp) as z:
        for info in z.infolist():
            target = (out / info.filename).resolve()
            if not str(target).startswith(str(out.resolve())):
                raise ValueError(f"unsafe path in zip: {info.filename}")
        z.extractall(out)
    tmp.unlink()
    (out / ".complete").touch()
    return f"done  {row['pack']} ({row['size_mb']} MB)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="~/itg-packs")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-gb", type=float, default=10.0)
    a = ap.parse_args()
    dest = pathlib.Path(a.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(LIST, encoding="utf-8"), delimiter="\t"))
    total = sum(float(r["size_mb"]) for r in rows) / 1024
    print(f"{len(rows)} packs, {total:.2f} GB to download into {dest}")
    if total > a.max_gb:
        sys.exit(f"list exceeds --max-gb {a.max_gb}")
    failed = []
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(fetch, r, dest): r for r in rows}
        for fut in concurrent.futures.as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # keep going, report at the end
                failed.append(futs[fut]["pack"])
                print(f"FAIL  {futs[fut]['pack']}: {e}", flush=True)
    if failed:
        sys.exit(f"{len(failed)} failed: {', '.join(sorted(failed))}")


if __name__ == "__main__":
    main()
