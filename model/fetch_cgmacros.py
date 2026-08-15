#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Kush Shah
# SPDX-License-Identifier: Apache-2.0
"""
Fetch only the CSVs out of the CGMacros archive, without downloading all of it.

The dataset ships as one 627 MB zip, and the overwhelming majority of that is
meal photographs this project does not use -- the model takes macronutrients
and glucose, both of which live in the per-participant CSVs.

A zip's central directory sits at the END of the file, and PhysioNet's S3
mirror honours HTTP range requests. So the archive can be treated as a remote
random-access file: read the directory, then fetch only the byte ranges of the
members actually wanted. `zipfile` needs nothing more than seek/read/tell, so
it does the parsing and decompression unmodified.

    python fetch_cgmacros.py [dest_dir]

Stdlib only, so this runs under both the modern interpreter and the pinned
legacy environment.
"""

import io
import os
import sys
import zipfile
from urllib.request import Request, urlopen

URL = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"

# Participants 24, 25, 37 and 40 did not complete and are absent from the
# archive. 45 of the 49 IDs are present; that is already the post-exclusion
# count, so do not subtract the four again.
EXPECTED_PARTICIPANTS = 45


class HttpFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests."""

    def __init__(self, url, chunk=1 << 16):
        self.url = url
        self._pos = 0
        self._chunk = chunk
        self._cache = {}  # (start, end) -> bytes, so re-reads are free
        with urlopen(Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
        self.bytes_fetched = 0

    # -- io plumbing --------------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self.size + offset
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n == 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        data = self._fetch(self._pos, end)
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    # -- the actual transfer ------------------------------------------------
    def _fetch(self, start, end):
        key = (start, end)
        if key in self._cache:
            return self._cache[key]
        req = Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urlopen(req) as r:
            data = r.read()
        self.bytes_fetched += len(data)
        if len(data) < (1 << 20):  # only cache small reads (directory traffic)
            self._cache[key] = data
        return data


def wanted(name):
    """CSVs only. Skip the photographs, which are the bulk of the archive."""
    return name.lower().endswith(".csv")


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(dest, exist_ok=True)

    hf = HttpFile(URL)
    print(f"remote archive: {hf.size / 1e6:.1f} MB")

    zf = zipfile.ZipFile(hf)
    members = zf.infolist()
    csvs = [m for m in members if wanted(m.filename)]
    csv_bytes = sum(m.compress_size for m in csvs)
    all_bytes = sum(m.compress_size for m in members)
    print(f"members: {len(members)}   CSVs: {len(csvs)}")
    print(f"compressed CSV bytes: {csv_bytes / 1e6:.1f} MB "
          f"of {all_bytes / 1e6:.1f} MB  "
          f"({100.0 * csv_bytes / max(all_bytes, 1):.1f}%)")
    print(f"directory read cost so far: {hf.bytes_fetched / 1e6:.2f} MB\n")

    written = 0
    for m in sorted(csvs, key=lambda x: x.filename):
        out = os.path.join(dest, os.path.basename(m.filename))
        with zf.open(m) as src, open(out, "wb") as dst:
            dst.write(src.read())
        written += 1
        print(f"  [{written}/{len(csvs)}] {os.path.basename(m.filename)} "
              f"({m.file_size / 1e6:.2f} MB)")

    print(f"\ntransferred {hf.bytes_fetched / 1e6:.1f} MB total "
          f"(vs {hf.size / 1e6:.1f} MB for the whole archive)")

    got = sorted(
        f for f in os.listdir(dest)
        if f.startswith("CGMacros-") and f.endswith(".csv")
    )
    print(f"participant files: {len(got)} (expected {EXPECTED_PARTICIPANTS})")
    if len(got) != EXPECTED_PARTICIPANTS:
        print("  NOTE: count differs from the documented 45 -- check before use")


if __name__ == "__main__":
    main()
