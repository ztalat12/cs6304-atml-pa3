"""Task 2: fetch the needed COCO images and cache frozen SigLIP patch features (FP16).

SigLIP-base-patch16-224: a 224 x 224 image is cut into 14 x 14 = 196 patches of 16 x 16 pixels; the encoder returns
one 768-number feature per patch. `last_hidden_state` ALREADY has SigLIP's final LayerNorm (post_layernorm) applied,
so it is taken as-is; applying the LayerNorm again would be a bug (checked on the first batch).
The 196 rows stay in the encoder's row-major 14 x 14 order: row 0 = top-left patch ... row 195 = bottom-right.
Preprocessing is deterministic: SiglipImageProcessor (PIL) resizes to 224 x 224, no crop, no augmentation.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from common.utils import Checks, gpu_peak, write_json

ENCODER = os.environ.get("PA3_T2_ENCODER", "google/siglip-base-patch16-224")   # env override: offline tests only
GRID = 14


def image_url(key):
    split, iid = key.split("/")
    return f"http://images.cocodataset.org/{split}/COCO_{split}_{int(iid):012d}.jpg"


def _names(key):
    """File names an image may have: COCO 2014 naming, or COCO 2017 naming (same image IDs, no prefix)."""
    split, iid = key.split("/")
    return [f"COCO_{split}_{int(iid):012d}.jpg", f"{int(iid):012d}.jpg"]


def find_coco_roots(search=("/kaggle/input",)):
    """Folders holding COCO images (train2014 / val2014 / train2017 / val2017 / any folder of COCO jpgs)."""
    roots = []
    for s in search:
        if not Path(s).exists():
            continue
        for d in Path(s).rglob("*"):
            if d.is_dir() and d.name in ("train2014", "val2014", "train2017", "val2017"):
                roots.append(d)
    return sorted(set(roots))


def local_index(roots):
    """{file name: path} over every attached COCO image folder."""
    idx = {}
    for r in roots:
        for p in Path(r).glob("*.jpg"):
            idx.setdefault(p.name, p)
    return idx


def fetch_images(keys, out_dir, roots=(), threads=32, log=print):
    """Returns ({key: path}, [failed keys]). Uses attached images where possible, otherwise downloads into out_dir."""
    out = Path(out_dir)
    idx = local_index(list(roots) + [out / "train2014", out / "val2014"])
    paths, todo = {}, []
    for k in keys:
        hit = next((idx[n] for n in _names(k) if n in idx), None)
        if hit is not None:
            paths[k] = hit
        else:
            todo.append(k)
    log(f"{len(paths)} images found locally, downloading {len(todo)} ...")

    def get(k):
        split, _ = k.split("/")
        dest = out / split / _names(k)[0]
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                req = urllib.request.Request(image_url(k), headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                tmp = dest.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.rename(dest)
                return k, dest
            except Exception:      # noqa: BLE001
                time.sleep(1 + 2 * attempt)
        return k, None

    t0 = time.time()
    with ThreadPoolExecutor(threads) as ex:
        for i, (k, p) in enumerate(ex.map(get, todo)):
            if p is not None:
                paths[k] = p
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(todo)} downloaded ({time.time() - t0:.0f}s)")
    failed = [k for k in keys if k not in paths]
    return paths, failed


@torch.no_grad()
def cache_features(keys, paths, out_dir, batch=64, log=print, checks: Checks | None = None):
    """Run the frozen encoder once per image and store N x 196 x 768 FP16 features (row order = `keys`)."""
    from PIL import Image
    from transformers import SiglipVisionModel
    try:                                     # PIL backend: identical resizing whether or not torchvision is installed
        from transformers import SiglipImageProcessorPil as SiglipProc
    except ImportError:
        from transformers import SiglipImageProcessor as SiglipProc

    C = checks or Checks()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    proc = SiglipProc.from_pretrained(ENCODER)
    enc = SiglipVisionModel.from_pretrained(ENCODER).to(dev).eval()      # FP32 compute, FP16 storage
    C.observe("SigLIP preprocessing (deterministic: fixed resize, no crop, no augmentation)",
              {"class": type(proc).__name__, **{k: getattr(proc, k, None) for k in ["size", "resample", "do_center_crop",
                                                                                     "image_mean", "image_std"]}})
    n_tok = (enc.config.image_size // enc.config.patch_size) ** 2
    dim = enc.config.hidden_size
    tmp = out / "features_fp16.npy.part"
    feats = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(keys), n_tok, dim))
    checked = False
    for s in range(0, len(keys), batch):
        ks = keys[s:s + batch]
        ims = [Image.open(paths[k]).convert("RGB") for k in ks]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        o = enc(pixel_values=px, output_hidden_states=not checked)
        h = o.last_hidden_state
        if not checked:
            C.check(tuple(h.shape[1:]) == (196, 768), "SigLIP output = 196 patch features x 768 numbers", (196, 768), tuple(h.shape[1:]))
            once = enc.vision_model.post_layernorm(o.hidden_states[-1]) if hasattr(enc, "vision_model") else enc.post_layernorm(o.hidden_states[-1])
            C.check(torch.allclose(once, h, atol=1e-4), "last_hidden_state = post_layernorm(encoder output): LayerNorm applied exactly once",
                    "max difference < 1e-4", f"{float((once - h).abs().max()):.2e}")
            lnmod = enc.vision_model.post_layernorm if hasattr(enc, "vision_model") else enc.post_layernorm
            C.observe("difference if the LayerNorm were applied a second time (what we avoid)", f"{float((lnmod(h) - h).abs().max()):.3f}")
            checked = True
        feats[s:s + len(ks)] = h.float().cpu().numpy().astype(np.float16)
        if (s // batch) % 20 == 0:
            log(f"  features {s + len(ks)}/{len(keys)}")
    feats.flush()
    del feats
    os.replace(tmp, out / "features_fp16.npy")
    secs = time.perf_counter() - t0
    write_json(out / "features_index.json", {"keys": keys, "encoder": ENCODER, "dtype": "float16", "shape": [len(keys), n_tok, dim],
                                             "processor": type(proc).__name__})
    stats = {"n_images": len(keys), "seconds": secs, **gpu_peak(), "bytes": (out / "features_fp16.npy").stat().st_size}
    write_json(out / "feature_cache_stats.json", stats)
    del enc
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    C.observe("feature caching cost (separate from training)", stats)
    return stats


def pool_2x2(x):
    """Average each 2 x 2 block of the 14 x 14 grid -> 7 x 7 = 49 tokens (row-major), for (B, 196, D) input."""
    b, n, d = x.shape
    g = int(round(n ** 0.5))
    return x.float().view(b, g // 2, 2, g // 2, 2, d).mean(dim=(2, 4)).reshape(b, (g // 2) ** 2, d)


class FeatureStore:
    def __init__(self, feat_dir, pool49=False):
        d = Path(feat_dir)
        self.arr = np.load(d / "features_fp16.npy", mmap_mode="r")
        self.index = {k: i for i, k in enumerate(json.load(open(d / "features_index.json"))["keys"])}
        self.pool49 = pool49

    def get(self, keys):
        x = torch.from_numpy(np.stack([self.arr[self.index[k]] for k in keys]).astype(np.float16))
        if self.pool49:
            x = pool_2x2(x).half()
        return x


def main():
    """python -m task2.t2_features --sel t2_selection.json --paths image_paths.json --out <feature dir>
    Runs in its own process, so the encoder's GPU memory is fully released before Qwen is trained."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keys = json.load(open(a.sel))["all_images"]
    paths = {k: Path(v) for k, v in json.load(open(a.paths)).items()}
    C = Checks()
    cache_features(keys, paths, a.out, checks=C)
    C.save(Path(a.out) / "feature_checks.json")
    sys_exit = 1 if C.summary() else 0
    raise SystemExit(sys_exit)


if __name__ == "__main__":
    main()
