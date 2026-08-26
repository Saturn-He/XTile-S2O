# -*- coding: utf-8 -*-
"""Fetch and verify the archived XTile-S2O-Urban76 data record.

The imagery, DEM rasters, and city mosaics are hosted in an archived Zenodo record
(doi:10.5281/zenodo.22082922). Components are split tar archives;
this script reads scripts/record.json, downloads the requested components part by part,
verifies each part's SHA-256, reassembles, and extracts under --target.

    python scripts/download_data.py --target ./data --components corpus76 dem fullscenes
    python scripts/download_data.py --target ./data --components all --dry-run
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = os.path.join(HERE, "record.json")


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch(url, dst):
    def hook(n, bs, total):
        if total > 0:
            pct = min(100.0, 100.0 * n * bs / total)
            sys.stdout.write(f"\r  {os.path.basename(dst)}: {pct:5.1f}%")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dst, reporthook=hook)
    sys.stdout.write("\n")


def safe_extract(tar, target):
    """Extract regular files/directories without allowing links or path traversal."""
    root = os.path.realpath(target)
    for member in tar.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"unsupported archive member type: {member.name}")
        destination = os.path.realpath(os.path.join(root, member.name))
        if os.path.commonpath([root, destination]) != root:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    tar.extractall(target)


def normalize_component_root(component, target):
    """Map archive-internal legacy roots to the public data layout."""
    if component != "corpus76":
        return
    legacy = os.path.join(target, "90cities_project1_dataset_descending")
    canonical = os.path.join(target, "corpus76")
    if not os.path.isdir(legacy):
        if not os.path.isdir(canonical):
            raise RuntimeError(f"corpus76 archive did not contain the expected root: {legacy}")
        return
    if os.path.exists(canonical):
        raise RuntimeError(
            f"cannot rename extracted corpus76 root because the destination exists: {canonical}"
        )
    os.rename(legacy, canonical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="./data")
    ap.add_argument("--components", nargs="+", default=["all"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args()

    rec = json.load(open(RECORD, encoding="utf-8"))
    comps = list(rec["components"]) if "all" in args.components else args.components
    unknown = [c for c in comps if c not in rec["components"]]
    if unknown:
        sys.exit(f"unknown component(s): {unknown}; available: {list(rec['components'])}")

    print(f"record: {rec['doi']}  ({rec['access']})")
    total = sum(rec["components"][c]["size_bytes"] for c in comps)
    print(f"selected {len(comps)} component(s), {total / 1e9:.1f} GB\n")
    if args.dry_run:
        for c in comps:
            m = rec["components"][c]
            print(f"  {c:<12} {m['size_bytes']/1e9:6.2f} GB  {len(m['parts'])} part(s)  {m['description']}")
        return

    os.makedirs(args.target, exist_ok=True)
    for c in comps:
        m = rec["components"][c]
        part_paths = []
        for part in m["parts"]:
            dst = os.path.join(args.target, part["filename"])
            if not (os.path.exists(dst) and os.path.getsize(dst) == part["size_bytes"]):
                print(f"[fetch] {part['filename']}")
                fetch(part["url"], dst)
            got = sha256(dst)
            if got != part["sha256"]:
                sys.exit(f"CHECKSUM MISMATCH for {part['filename']}\n"
                         f"  expected {part['sha256']}\n  got      {got}")
            print(f"[ok]    {part['filename']} sha256 verified")
            part_paths.append(dst)

        arc = os.path.join(args.target, m["archive"])
        with open(arc, "wb") as out:
            for part_path in part_paths:
                with open(part_path, "rb") as src:
                    while True:
                        block = src.read(1 << 24)
                        if not block:
                            break
                        out.write(block)
        if os.path.getsize(arc) != m["size_bytes"]:
            sys.exit(f"SIZE MISMATCH for reassembled {m['archive']}")
        with tarfile.open(arc) as tar:
            safe_extract(tar, args.target)
        normalize_component_root(c, args.target)
        print(f"[done]  {c} extracted -> {args.target}")
        if not args.keep_archives:
            os.remove(arc)
            for part_path in part_paths:
                os.remove(part_path)
    print("\nAll requested components are present and verified.")


if __name__ == "__main__":
    main()
