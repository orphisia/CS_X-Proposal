#!/usr/bin/env python3
"""Download training images from free public-domain / CC sources.

Two classes, two sources each:

    mandalas : Met Museum Open Access  +  Wikimedia Commons
    icons    : Met Museum Open Access  +  Wikimedia Commons

Every download is logged to scripts/00_provenance_log.md (source URL + license),
which is the project's licensing defense at exhibition. Re-running is safe: files
already on disk are skipped, so an interrupted scrape just resumes.

Usage
-----
    python scripts/01_scrape_data.py --class mandalas --limit 750
    python scripts/01_scrape_data.py --class icons    --limit 750

Both APIs are keyless. Wikimedia requires a descriptive User-Agent (bot policy),
so we send one with a contact email.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
CONTACT = os.environ.get("CSX_CONTACT", "scano@andrew.cmu.edu")
USER_AGENT = f"CSX-Research/1.0 (CMU CS+X art project; {CONTACT})"

MET_SEARCH = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{id}"
WIKI_API = "https://commons.wikimedia.org/w/api.php"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

# Per-class source configuration.
# Source config per class. Category names were verified live against the
# Commons API — the generic "Mandalas" category is deliberately avoided because
# its subcats are noise (Computer-generated / SVG / Modern mandalas); the real
# painted thangkas live under "Thangka". "met_accept_class" keeps only 2D
# painted/textile art, dropping the sculptures the broad search returns.
SOURCES = {
    "mandalas": {
        # "thangka" is far more specific than "mandala" (which matches unrelated
        # European paintings now that sculptures are class-filtered out).
        "met_queries": ["thangka", "mandala"],
        "met_accept_class": ("painting", "textile"),
        "wiki_categories": ["Thangka", "Sand mandalas", "Vajrayana", "Buddhist paintings"],
    },
    "icons": {
        "met_queries": ["icon", "orthodox icon", "byzantine icon"],
        "met_accept_class": ("painting", "icon"),
        "wiki_categories": ["Eastern Orthodox icons", "Byzantine icons", "Icons of Russia"],
    },
}


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def polite_get(session, url, params=None, delay=0.0, retries=3, stream=False, timeout=30):
    """GET with a politeness delay and exponential backoff."""
    if delay:
        time.sleep(delay)
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, stream=stream, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1:
                tqdm.write(f"  ! giving up on {url} ({e})")
                return None
            time.sleep(2 ** attempt)
    return None


def ext_from_url(url, default=".jpg"):
    ext = Path(urlparse(url).path).suffix.lower()
    return ext if ext in IMG_EXTS else default


def count_images(d):
    """Count real files in a dir, ignoring hidden entries like .gitkeep."""
    return len([p for p in d.glob("*") if p.is_file() and not p.name.startswith(".")])


# Public-domain / Creative-Commons allowlist for Wikimedia files. We reject
# anything unknown or non-free — this is an exhibited, grant-funded piece and
# licensing-first is the whole provenance story. Met OA is already CC0-filtered
# upstream via isPublicDomain, so this only guards the Wikimedia path.
_LICENSE_OK = ("public domain", "pd-", "pd ", "cc0", "cc-zero", "cc by", "cc-by",
               "cc sa", "cc-sa", "creative commons", "attribution")
_LICENSE_BAD = ("non-free", "fair use", "all rights reserved", "copyrighted")


def license_ok(meta):
    """Return (accepted: bool, label: str) for a Wikimedia extmetadata blob."""
    def val(key):
        return (meta.get(key, {}) or {}).get("value", "")

    short = val("LicenseShortName")
    code = val("License")               # machine-readable, e.g. "cc-by-sa-4.0", "pd"
    copyrighted = val("Copyrighted").strip().lower()  # "True" / "False"
    label = short or code or "unknown"
    text = f"{short} {code}".lower()

    if copyrighted == "false":          # Commons marks PD works as not copyrighted
        return True, label
    if any(b in text for b in _LICENSE_BAD):
        return False, label
    if any(o in text for o in _LICENSE_OK):
        return True, label
    return False, label                 # unknown / missing -> reject (conservative)


# --------------------------------------------------------------------------- #
# candidate gathering — returns list of dicts: {url, license, source, sid}
# --------------------------------------------------------------------------- #
def gather_met(session, queries, want, accept_class=()):
    """Walk Met search results, keep public-domain objects that have an image.

    ``accept_class``: only keep objects whose ``classification`` contains one of
    these substrings (e.g. "painting", "textile", "icon"). This drops the
    sculptures / ritual metalwork the broad keyword search otherwise returns.
    """
    out, seen, rej_class = [], set(), 0
    for q in queries:
        r = polite_get(session, MET_SEARCH, params={"q": q, "hasImages": "true"}, delay=0.1)
        if r is None:
            continue
        ids = (r.json().get("objectIDs") or [])[: want * 6]  # probe budget
        for oid in tqdm(ids, desc=f"met:{q}", unit="obj", leave=False):
            if oid in seen or len(out) >= want:
                continue
            seen.add(oid)
            ro = polite_get(session, MET_OBJECT.format(id=oid), delay=0.05)
            if ro is None:
                continue
            obj = ro.json()
            img = obj.get("primaryImage") or ""
            if not (obj.get("isPublicDomain") and img):
                continue
            cls = (obj.get("classification") or "").lower()
            if accept_class and not any(a in cls for a in accept_class):
                rej_class += 1
                continue
            out.append({
                "url": img,
                "license": "CC0 (Met Open Access, public domain)",
                "source": "met",
                "sid": str(oid),
            })
            if len(out) >= want:
                break
        if len(out) >= want:
            break
    if rej_class:
        print(f"  (met: rejected {rej_class} objects off-classification, e.g. sculpture)")
    return out


def _wiki_collect_titles(session, category, max_files, max_depth):
    """BFS a category and its subcategories (to max_depth) for file titles.

    cmtype=file|subcat returns both files (ns 6) and subcategories (ns 14);
    most Commons icon/mandala categories are subcategory trees, so without this
    recursion the direct-file counts fall far short.
    """
    titles, visited = [], set()
    queue = [(category, 0)]
    while queue and len(titles) < max_files:
        cat, depth = queue.pop(0)
        if cat in visited:
            continue
        visited.add(cat)
        cmcontinue = None
        for _ in range(8):  # page cap per category
            params = {
                "action": "query", "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat}",
                "cmtype": "file|subcat", "cmlimit": "500",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            r = polite_get(session, WIKI_API, params=params, delay=1.0)
            if r is None:
                break
            data = r.json()
            for m in data.get("query", {}).get("categorymembers", []):
                if m.get("ns") == 6:  # File
                    if Path(m["title"]).suffix.lower() in IMG_EXTS:
                        titles.append(m["title"])
                elif m.get("ns") == 14 and depth < max_depth:  # subcategory
                    sub = m["title"].split("Category:", 1)[-1]
                    queue.append((sub, depth + 1))
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue or len(titles) >= max_files:
                break
    return titles


def gather_wikimedia(session, categories, want, max_depth=2):
    """PD/CC files across categories (recursing subcats), license-filtered."""
    titles, seen = [], set()
    for cat in categories:
        for t in _wiki_collect_titles(session, cat, max_files=want * 3, max_depth=max_depth):
            if t not in seen:
                seen.add(t)
                titles.append(t)
        if len(titles) >= want * 3:
            break

    out, rejected = [], 0
    for i in tqdm(range(0, len(titles), 50), desc="wiki:imageinfo", unit="batch", leave=False):
        batch = titles[i : i + 50]
        params = {
            "action": "query", "format": "json",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "1024",   # ask for a 1024px-wide scaled version (thumburl)
        }
        r = polite_get(session, WIKI_API, params=params, delay=1.0)
        if r is None:
            continue
        for page in r.json().get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            if not url:
                continue
            ok, label = license_ok(info.get("extmetadata", {}))
            if not ok:
                rejected += 1
                continue
            out.append({
                "url": url,                              # canonical full URL (provenance + stable sid)
                "dl_url": info.get("thumburl") or url,   # download the small 1024px version when available
                "license": f"{label} (Wikimedia Commons)",
                "source": "wikimedia",
                "sid": hashlib.sha1(url.encode()).hexdigest()[:12],
            })
    if rejected:
        print(f"  (wikimedia: rejected {rejected} non-PD/CC or unknown-license files)")
    return out


# --------------------------------------------------------------------------- #
# download + provenance
# --------------------------------------------------------------------------- #
def download(session, cand, out_dir):
    fname = f"{cand['source']}_{cand['sid']}{ext_from_url(cand['url'])}"
    path = out_dir / fname
    if path.exists():
        return path, False  # resume: already have it
    # Prefer the small 1024px thumbnail; pace Wikimedia downloads to avoid 429s.
    delay = 0.4 if cand.get("source") == "wikimedia" else 0.0
    url = cand.get("dl_url") or cand["url"]
    r = polite_get(session, url, delay=delay, stream=True)
    if r is None:
        return None, False
    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype:
        return None, False
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        tmp.rename(path)
        return path, True
    except OSError:
        # transient FS issue (e.g. iCloud sync evicting the temp file) — skip this
        # one file rather than crash the whole run.
        try:
            tmp.unlink()
        except OSError:
            pass
        return None, False


def log_provenance(log_path, rows):
    with open(log_path, "a") as f:
        for r in rows:
            f.write(f"| {r['fname']} | {r['cls']} | {r['source']} | {r['url']} | {r['license']} |\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class", dest="cls", required=True, choices=sorted(SOURCES))
    ap.add_argument("--limit", type=int, default=750, help="target image count for this class")
    ap.add_argument("--out", default=None, help="output dir (default data/raw/<class>)")
    ap.add_argument("--log", default=str(REPO / "scripts" / "00_provenance_log.md"))
    ap.add_argument("--max-depth", type=int, default=2,
                    help="Wikimedia subcategory recursion depth")
    ap.add_argument("--dry-run", action="store_true",
                    help="gather + report candidate counts, then stop (no downloads)")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else REPO / "data" / "raw" / args.cls
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = SOURCES[args.cls]
    session = make_session()

    have = count_images(out_dir)
    need = max(0, args.limit - have)
    print(f"[{args.cls}] have {have}, target {args.limit} -> need {need}")
    if need == 0 and not args.dry_run:
        print("Already at target. Nothing to do.")
        return
    want = max(need, args.limit)  # in dry-run, need may be 0; size the gather to the target

    # Split the request roughly evenly between the two sources, with buffer.
    print("Gathering Met candidates...")
    met = gather_met(session, cfg["met_queries"], want=want // 2 + want // 4,
                     accept_class=cfg.get("met_accept_class", ()))
    print(f"  Met: {len(met)} candidates")
    print("Gathering Wikimedia candidates...")
    wiki = gather_wikimedia(session, cfg["wiki_categories"],
                            want=want // 2 + want // 4, max_depth=args.max_depth)
    print(f"  Wikimedia: {len(wiki)} candidates (post license-filter)")

    # Interleave so the dataset mixes both sources.
    candidates, seen = [], set()
    for pair in zip_longest(met, wiki):
        for c in pair:
            if c and c["url"] not in seen:
                seen.add(c["url"])
                candidates.append(c)

    print(f"  Combined unique downloadable candidates: {len(candidates)} "
          f"(target {args.limit})")
    if args.dry_run:
        if len(candidates) < args.limit:
            print(f"DRY-RUN: SHORT by {args.limit - len(candidates)} — add categories/"
                  f"queries in SOURCES or raise --max-depth before the real run.")
        else:
            print("DRY-RUN: supply looks sufficient. Re-run without --dry-run to download.")
        return

    downloaded, rows = 0, []
    for c in tqdm(candidates, desc=f"download:{args.cls}", unit="img"):
        if downloaded >= need:
            break
        path, is_new = download(session, c, out_dir)
        if path is None:
            continue
        if is_new:
            downloaded += 1
            rows.append({"fname": path.name, "cls": args.cls, "source": c["source"],
                         "url": c["url"], "license": c["license"]})

    log_provenance(args.log, rows)
    total = count_images(out_dir)
    print(f"\nDone. Downloaded {downloaded} new images this run; {total} total in {out_dir}.")
    print(f"Provenance for {len(rows)} new images appended to {args.log}.")
    if total < args.limit:
        print(f"NOTE: below target ({total} < {args.limit}). Sources may be exhausted; "
              f"add more categories/queries in SOURCES or collect the rest manually.")


def zip_longest(a, b):
    n = max(len(a), len(b))
    for i in range(n):
        yield (a[i] if i < len(a) else None, b[i] if i < len(b) else None)


if __name__ == "__main__":
    sys.exit(main())
