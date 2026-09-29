#!/usr/bin/env python3
"""Rank candidate training packs: ITG Packs Release Spreadsheet x nnty.fun mirror.

Prints a TSV (same columns as training/packs.tsv) of every eligible pack, best
estimated charts-per-MB first. `training/packs.tsv` is the first rows of this list
(greedy, stopping at ~9000 estimated charts); to top up, append the next rows that
are not already in packs.tsv, keeping the total under 10 GB.

    python3 training/select_packs.py > candidates.tsv
"""

import csv
import html
import io
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

SHEET = "1F1IURV1UAYiICTLhAOKIJfwUN1iG12ZOufHZuDKiP48"
# Tabs with a "Pack name" column (main index + yearly release tabs).
GIDS = [1652402514, 1198223746, 1301407124, 1878312571, 27038621]
MIRROR = "https://nnty.fun/downloads/game/stepmania/stepmania_packs/"
ALREADY_HAVE = {"sexualityviolation", "sexualityviolation2", "sexualityviolation3"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "itg-charter"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s.lower()))


def mirror_zips():
    page = get(MIRROR)
    rows = re.findall(
        r'<a href="([^"]+\.zip)">.*?</a>.*?\d{4}-\d\d-\d\d \d\d:\d\d.*?([\d.]+) ([KMG]B)',
        page,
        re.S,
    )
    scale = {"KB": 1 / 1024, "MB": 1, "GB": 1024}
    out = {}
    for href, v, unit in rows:
        name = urllib.parse.unquote(html.unescape(href.split("/")[-1]))[:-4]
        out[norm(name)] = (name, float(v) * scale[unit])
    return out


def sheet_rows():
    packs = {}
    for gid in GIDS:
        text = get(f"https://docs.google.com/spreadsheets/d/{SHEET}/export?format=csv&gid={gid}")
        r = list(csv.reader(io.StringIO(text)))
        if not r or "Pack name" not in r[0]:
            continue
        h = r[0]
        for row in r[1:]:
            row += [""] * (len(h) - len(row))
            d = dict(zip(h, row))
            if d["Pack name"]:
                packs[d["Pack name"]] = d
    return packs


def main():
    zips = mirror_zips()
    out = []
    for name, d in sorted(sheet_rows().items()):
        key = norm(name)
        if key not in zips or key in ALREADY_HAVE:
            continue
        zname, mb = zips[key]
        if d.get("Format", "") not in ("Singles", "Singles + Doubles", ""):
            continue
        ct = d.get("Content Type", "").lower()
        if "mod" in ct or "gimmick" in ct:
            continue
        # "1-5S/0-4D" -> singles charts per song, averaged over the range.
        n = [int(x) for x in re.findall(r"\d+", d.get("Difficulties", "").split("S")[0].split("/")[0])]
        songs = re.sub(r"\D", "", d.get("Songcount", ""))
        if not n or not songs:
            continue
        per_song, songs = min(sum(n) / len(n), 6), int(songs)
        if songs < 5 or not 5 <= mb <= 900 or per_song < 3:
            continue
        out.append((songs * per_song / mb, name, zname, mb, songs, per_song))
    out.sort(key=lambda x: (-x[0], x[1]))
    w = sys.stdout
    w.write("pack\tzip\tsize_mb\tsongs\tsingles_per_song\n")
    for _, name, zname, mb, songs, per_song in out:
        w.write(f"{name}\t{zname}\t{mb:.0f}\t{songs}\t{per_song:g}\n")


if __name__ == "__main__":
    main()
