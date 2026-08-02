#!/usr/bin/env python3
"""Read individual tensors out of a HuggingFace safetensors file over HTTP.

The Z-Image transformer is 24.6 GB of fp32 across three shards. Downloading a
shard to pull four blocks out of it costs 8 GB of disk for 1.5 GB of weights,
and the shard has to come down again for every part that touches it. On a box
with 8 GB of free disk that is not a tuning problem, it is a wall.

safetensors is trivially range-readable, so this skips the download entirely:

    8 bytes   little-endian uint64 header length N
    N bytes   JSON: {name: {dtype, shape, data_offsets: [start, end]}, ...}
    rest      tensor data, offsets relative to 8+N

So one small GET for the header, then one GET per (coalesced) run of tensors
actually wanted. Pulling one DiT block costs ~400 MB of transfer and ~0 disk.

    r = RemoteSafetensors("Tongyi-MAI/Z-Image-Turbo",
                          "transformer/diffusion_pytorch_model-00001-of-00003.safetensors")
    r.keys()                      # names, no data fetched
    r.get_tensors(["layers.0.attention.wq.weight", ...])   # -> {name: tensor}
"""
import json
import os
import struct
import time

import numpy as np
import torch

# safetensors dtype -> (numpy dtype for the raw bytes, torch dtype to view as).
# bf16 has no numpy equivalent, so it is read as uint16 and bit-cast.
_DTYPES = {
    "F64": (np.float64, torch.float64),
    "F32": (np.float32, torch.float32),
    "F16": (np.float16, torch.float16),
    "BF16": (np.uint16, torch.bfloat16),
    "I64": (np.int64, torch.int64),
    "I32": (np.int32, torch.int32),
    "I16": (np.int16, torch.int16),
    "I8": (np.int8, torch.int8),
    "U8": (np.uint8, torch.uint8),
    "BOOL": (np.bool_, torch.bool),
}

# Gaps smaller than this are read through rather than split into two requests:
# a extra round trip costs more than the wasted bytes.
_COALESCE_GAP = 8 << 20


class RemoteSafetensors:
    def __init__(self, repo, path, revision="main", token=None, retries=5):
        self.url = f"https://huggingface.co/{repo}/resolve/{revision}/{path}"
        self.retries = retries
        self._session = None
        self._token = token or os.environ.get("HF_TOKEN")
        raw = self._get(0, 8 + 8)
        (hlen,) = struct.unpack("<Q", raw[:8])
        header = json.loads(self._get(8, 8 + hlen).decode("utf-8"))
        header.pop("__metadata__", None)
        self.header = header
        self.data_start = 8 + hlen

    # ---- HTTP ---------------------------------------------------------------
    @property
    def session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            if self._token:
                self._session.headers["Authorization"] = f"Bearer {self._token}"
        return self._session

    def _get(self, start, end):
        """Bytes [start, end). Retries on anything transient — a multi-hour
        conversion should not die because one range read got a 503."""
        headers = {"Range": f"bytes={start}-{end - 1}"}
        last = None
        for attempt in range(self.retries):
            try:
                r = self.session.get(self.url, headers=headers, timeout=300)
                if r.status_code not in (200, 206):
                    raise OSError(f"HTTP {r.status_code} for bytes {start}-{end}")
                got = r.content
                if len(got) != end - start:
                    raise OSError(f"short read: {len(got)} != {end - start}")
                return got
            except Exception as exc:  # noqa: BLE001 - retried below, raised at the end
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise OSError(f"range read failed after {self.retries} tries: {last}")

    # ---- tensors ------------------------------------------------------------
    def keys(self):
        return list(self.header)

    def nbytes(self, name):
        a, b = self.header[name]["data_offsets"]
        return b - a

    def _decode(self, name, buf):
        info = self.header[name]
        np_dt, torch_dt = _DTYPES[info["dtype"]]
        arr = np.frombuffer(buf, dtype=np_dt).copy()
        t = torch.from_numpy(arr)
        if info["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        return t.view(info["shape"]) if info["shape"] else t

    def get_tensors(self, names, dtype=None, progress=None):
        """Fetch several tensors, coalescing neighbours into single requests.

        Tensors are returned as they are read, so the caller can cast and drop
        the fp32 original before the next range lands — the whole point is to
        never hold the full part in fp32.
        """
        names = [n for n in names if n in self.header]
        names.sort(key=lambda n: self.header[n]["data_offsets"][0])

        runs, cur = [], []
        for n in names:
            a, b = self.header[n]["data_offsets"]
            if cur and a - cur[-1][2] <= _COALESCE_GAP:
                cur.append((n, a, b))
            else:
                if cur:
                    runs.append(cur)
                cur = [(n, a, b)]
        if cur:
            runs.append(cur)

        out = {}
        for run in runs:
            lo, hi = run[0][1], run[-1][2]
            buf = self._get(self.data_start + lo, self.data_start + hi)
            for n, a, b in run:
                t = self._decode(n, buf[a - lo : b - lo])
                out[n] = t.to(dtype) if dtype is not None else t
            del buf
            if progress:
                progress(len(out), len(names), hi - lo)
        return out


def open_shards(repo, subdir, revision="main", token=None):
    """Every shard of a sharded checkpoint, plus a name -> shard map.

    Falls back to the single-file layout when there is no index, so this works
    for the VAE and text encoder too.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    prefix = f"{subdir}/" if subdir else ""
    try:
        index = hf_hub_download(
            repo, f"{prefix}diffusion_pytorch_model.safetensors.index.json",
            revision=revision, token=token)
        weight_map = json.load(open(index))["weight_map"]
    except EntryNotFoundError:
        weight_map = None

    if weight_map is None:
        shard = f"{prefix}diffusion_pytorch_model.safetensors"
        r = RemoteSafetensors(repo, shard, revision, token)
        return {shard: r}, {k: shard for k in r.keys()}

    readers = {}
    for shard in sorted(set(weight_map.values())):
        readers[shard] = RemoteSafetensors(repo, prefix + shard, revision, token)
    return readers, weight_map
