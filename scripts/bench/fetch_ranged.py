#!/usr/bin/env python3
"""Download a large file in verified byte-range chunks, resuming a partial ``.part`` file.

    python fetch_ranged.py URL DEST

Each chunk must come back as HTTP 206 with a Content-Range matching the request; a server
that ignores ranges (200) or a short chunk stops the download rather than corrupting the
file. A failed chunk is retried; the final size must equal the server's Content-Length.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request

CHUNK = 256 * 1024 * 1024


def _size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def _chunk(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise SystemExit(f"server ignored the byte range (HTTP {r.status}); refusing to "
                             "guess how to resume")
        cr = r.headers.get("Content-Range", "")
        if not cr.startswith(f"bytes {start}-{end}/"):
            raise SystemExit(f"Content-Range {cr!r} does not match bytes {start}-{end}")
        data = r.read()
    if len(data) != end - start + 1:
        raise OSError(f"short chunk: {len(data)} of {end - start + 1} bytes")
    return data


def main() -> None:
    url, dest = sys.argv[1], sys.argv[2]
    if os.path.exists(dest):
        return
    total = _size(url)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > total:
        raise SystemExit(f"{part} is larger than the remote file; delete it")
    with open(part, "ab") as f:
        while have < total:
            end = min(have + CHUNK, total) - 1
            for attempt in range(10):
                try:
                    data = _chunk(url, have, end)
                    break
                except OSError as e:          # network errors: retry this chunk
                    print(f"  chunk {have}-{end} failed ({e}); retry {attempt + 1}/10",
                          flush=True)
                    time.sleep(15)
            else:
                raise SystemExit(f"chunk {have}-{end} failed 10 times; resubmit to resume")
            f.write(data)
            f.flush()
            have += len(data)
            print(f"  {have / 1e9:.2f} / {total / 1e9:.2f} GB", flush=True)
    if os.path.getsize(part) != total:
        raise SystemExit(f"{part} has {os.path.getsize(part)} bytes, expected {total}")
    os.replace(part, dest)


if __name__ == "__main__":
    main()
