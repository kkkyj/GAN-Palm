#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cache_esm2_3b_for_sitelevel_v2.py

Input:
  unique_seq101.tsv (from build_sitelevel_dataset_v2.py)
Output:
  out_dir/
    manifest.jsonl
    part_000000.npz, part_000001.npz, ...
  final_out.npz (optional --merge-to)

Each part npz:
  - seqs: (M,) object
  - emb_center: (M, 2560) float16/float32
  - emb_mean:   (M, 2560) float16/float32

ESM2-3B:
  esm.pretrained.esm2_t36_3B_UR50D()

For 101aa fixed window:
  token positions: 0=BOS, 1..101=AA, 102=EOS
  center AA index=50 -> token index = 1+50 = 51
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import esm


CENTER_TOKEN_IDX_101 = 51  # BOS=0, AA[0]=1, ..., AA[50]=51


def load_unique_seqs_tsv(path: Path, max_n: Optional[int] = None) -> List[str]:
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for ln in f:
            if not ln.strip():
                continue
            s = ln.split("\t")[0].strip().upper()
            if s:
                seqs.append(s)
                if max_n is not None and len(seqs) >= max_n:
                    break
    return seqs


def chunk_indices(n: int, chunk_size: int) -> List[Tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        out.append((i, j))
        i = j
    return out


def already_done_parts(manifest_path: Path) -> set:
    done = set()
    if not manifest_path.exists():
        return done
    with open(manifest_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if obj.get("status") == "ok":
                    done.add(int(obj["part_id"]))
            except Exception:
                continue
    return done


def write_manifest(manifest_path: Path, rec: dict):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unique-seq", required=True, help="unique_seq101.tsv")
    ap.add_argument("--outdir", required=True, help="output directory for parts + manifest")
    ap.add_argument("--merge-to", default="", help="if set, merge all parts into one npz at this path")

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=5000, help="how many seqs per part file")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--layer", type=int, default=36)

    ap.add_argument("--max-n", type=int, default=0, help="debug: only process first N seqs (0=all)")
    ap.add_argument("--device", default="", help="cuda / cpu / cuda:0 (default: auto)")
    ap.add_argument("--no-amp", action="store_true", help="disable autocast")
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--empty-cache-every", type=int, default=0, help="call torch.cuda.empty_cache() every K parts (0=never)")

    args = ap.parse_args()

    unique_path = Path(args.unique_seq)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "manifest.jsonl"

    max_n = args.max_n if args.max_n and args.max_n > 0 else None
    seqs = load_unique_seqs_tsv(unique_path, max_n=max_n)
    n = len(seqs)
    print(f"[Load] unique seqs = {n}")

    # device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Device]", device)

    # dtype config
    if args.dtype == "fp16":
        out_np_dtype = np.float16
        amp_dtype = torch.float16
    elif args.dtype == "bf16":
        out_np_dtype = np.float16  # still store as float16 on disk (more compact), use bf16 for inference
        amp_dtype = torch.bfloat16
    else:
        out_np_dtype = np.float32
        amp_dtype = torch.float32

    # model
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    pad = alphabet.padding_idx

    # resume support
    spans = chunk_indices(n, int(args.chunk_size))
    done_parts = already_done_parts(manifest_path)
    print(f"[Resume] total_parts={len(spans)} done_parts={len(done_parts)}")

    bs = max(1, int(args.batch_size))

    # main loop over parts
    for part_id, (st, ed) in enumerate(spans):
        if part_id in done_parts:
            continue

        part_seqs = seqs[st:ed]
        part_path = outdir / f"part_{part_id:06d}.npz"

        try:
            # prealloc
            m = len(part_seqs)
            emb_center = np.empty((m, 2560), dtype=out_np_dtype)
            emb_mean = np.empty((m, 2560), dtype=out_np_dtype)

            # process in batches within this part
            w = 0
            for i in range(0, m, bs):
                chunk = part_seqs[i:i + bs]
                data = [(f"s{st+i+j}", s) for j, s in enumerate(chunk)]
                _, _, tokens = batch_converter(data)  # (B, L)
                if device.startswith("cuda") and args.pin_memory:
                    tokens = tokens.pin_memory()
                tokens = tokens.to(device, non_blocking=True)

                # mask for AA tokens (exclude BOS/EOS/pad)
                # tokens: 0=BOS, ... , EOS, pad=pad
                # valid token positions: tokens != pad
                valid = (tokens != pad)
                # exclude BOS
                valid[:, 0] = False
                # exclude EOS: last valid token is EOS; mark it False by finding lengths
                lengths = valid.sum(dim=1)  # includes EOS at end as True currently? depends on batch_converter padding.
                # Better: compute per-row last non-pad index and set False there.
                # last_idx = (valid.long().sum(dim=1) - 1) gives last True position index after excluding BOS? Not safe.
                # We'll find last non-pad index from tokens != pad:
                nonpad = (tokens != pad)
                last_idx = nonpad.long().sum(dim=1) - 1  # index of last non-pad token (EOS)
                # set EOS false
                valid.scatter_(1, last_idx.view(-1, 1), False)

                with torch.inference_mode():
                    if (not args.no_amp) and device.startswith("cuda") and (args.dtype in ("fp16", "bf16")):
                        with torch.cuda.amp.autocast(dtype=amp_dtype):
                            out = model(tokens, repr_layers=[args.layer], return_contacts=False)
                            reps = out["representations"][args.layer]  # (B, L, d)
                    else:
                        out = model(tokens, repr_layers=[args.layer], return_contacts=False)
                        reps = out["representations"][args.layer]  # (B, L, d)

                    # mean pooling over AA tokens using mask
                    # valid: (B, L) bool -> (B, L, 1)
                    v = valid.unsqueeze(-1)
                    reps_masked = reps * v
                    denom = v.sum(dim=1).clamp(min=1)  # (B, 1)
                    mean_vec = reps_masked.sum(dim=1) / denom  # (B, d)

                    # center vec: for 101aa window center token is fixed index 51
                    # if sequence is shorter (shouldn't for seq101), fallback to mean for those rows
                    center_idx = CENTER_TOKEN_IDX_101
                    if reps.size(1) > center_idx:
                        center_vec = reps[:, center_idx, :]  # (B, d)
                    else:
                        center_vec = mean_vec

                # to cpu numpy
                center_np = center_vec.detach().to("cpu").to(torch.float16 if out_np_dtype == np.float16 else torch.float32).numpy()
                mean_np = mean_vec.detach().to("cpu").to(torch.float16 if out_np_dtype == np.float16 else torch.float32).numpy()

                bsz = len(chunk)
                emb_center[w:w + bsz] = center_np[:bsz]
                emb_mean[w:w + bsz] = mean_np[:bsz]
                w += bsz

            np.savez_compressed(
                part_path,
                seqs=np.array(part_seqs, dtype=object),
                emb_center=emb_center,
                emb_mean=emb_mean
            )

            write_manifest(manifest_path, {
                "status": "ok",
                "part_id": part_id,
                "start": st,
                "end": ed,
                "n": int(ed - st),
                "path": str(part_path),
                "dtype": str(out_np_dtype),
                "layer": int(args.layer),
            })

            print(f"[OK] part {part_id}/{len(spans)}  [{st}:{ed}]  -> {part_path.name}")

            if args.empty_cache_every and device.startswith("cuda"):
                if (part_id + 1) % int(args.empty_cache_every) == 0:
                    torch.cuda.empty_cache()

        except Exception as e:
            write_manifest(manifest_path, {
                "status": "fail",
                "part_id": part_id,
                "start": st,
                "end": ed,
                "error": repr(e),
            })
            raise

    print(f"[DONE] parts in {outdir}")
    print(f"[DONE] manifest {manifest_path}")

    # optional merge
    if args.merge_to:
        merge_path = Path(args.merge_to)
        part_files = sorted(outdir.glob("part_*.npz"))
        if not part_files:
            raise RuntimeError("No part_*.npz found to merge.")

        all_seqs = []
        all_center = []
        all_mean = []
        for p in part_files:
            z = np.load(p, allow_pickle=True)
            all_seqs.append(z["seqs"])
            all_center.append(z["emb_center"])
            all_mean.append(z["emb_mean"])

        seqs_merged = np.concatenate(all_seqs, axis=0)
        center_merged = np.concatenate(all_center, axis=0)
        mean_merged = np.concatenate(all_mean, axis=0)

        np.savez_compressed(
            merge_path,
            seqs=seqs_merged,
            emb_center=center_merged,
            emb_mean=mean_merged
        )
        print(f"[MERGE] -> {merge_path}  N={seqs_merged.shape[0]} center={center_merged.shape} mean={mean_merged.shape}")


if __name__ == "__main__":
    main()
