#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_proteome_background_embeddings.py

Build per-sample "cell background" embeddings from site-level long table.

Input:
  site_sample_long.all.parquet  (from build_sitelevel_dataset_v2.py)

Expected columns (at least):
  - dataset (e.g. Hela / Mouse_liver / PANC-1)
  - sample  (e.g. WT1..WT5, KO1..KO5)  OR (group, rep) available to construct
  - prot_abund (float)
Optional:
  - palm_prot_abund (float)  (protein-level palmitoylation abundance)

Output:
  proteome_bg_embeddings.npz with keys:
    - samples: (M,) str
    - combined_emb: (M, D) float32
    - meta: dict (np.object_)

Emb definition (robust, cheap, no ML):
  For each sample, compute distribution stats of prot_abund and palm_prot_abund:
    [mean, std, q10, q50, q90, min, max]  for each feature
  plus:
    - frac_nonzero, frac_pos, log1p_mean, log1p_std  (more stable)
  Concatenate => D dims.

Two modes of sample id:
  1) dataset-aware:   sample_id = f"{dataset}__{sample}"
  2) pooled:          sample_id = sample    (WT1..KO5 merged across datasets)

You can choose with --id-mode {dataset,pooled}.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _to_float(x: pd.Series) -> np.ndarray:
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=np.float64)
    v = v[np.isfinite(v)]
    return v


def _stats_block(v: np.ndarray) -> np.ndarray:
    """
    Return a robust stats vector for a 1D array.
    Output dims = 7 + 4 = 11:
      mean, std, q10, q50, q90, min, max,
      frac_nonzero, frac_pos, log1p_mean, log1p_std
    """
    if v.size == 0:
        return np.zeros(11, dtype=np.float32)

    # raw
    mean = float(v.mean())
    std = float(v.std())
    q10, q50, q90 = np.percentile(v, [10, 50, 90]).astype(float)
    vmin = float(v.min())
    vmax = float(v.max())

    # fractions
    frac_nonzero = float((v != 0).mean())
    frac_pos = float((v > 0).mean())

    # log1p stabilizes heavy-tailed abundance
    lv = np.log1p(np.clip(v, a_min=0.0, a_max=None))
    log_mean = float(lv.mean())
    log_std = float(lv.std())

    out = np.array(
        [mean, std, q10, q50, q90, vmin, vmax, frac_nonzero, frac_pos, log_mean, log_std],
        dtype=np.float32,
    )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def build_embeddings(
    df: pd.DataFrame,
    id_mode: str,
    feature_cols: List[str],
    min_rows: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Return:
      samples: (M,) str
      emb:     (M, D) float32
      meta:    dict
    """
    need = {"dataset"}
    if "sample" not in df.columns:
        # try construct from (group, rep)
        if ("group" in df.columns) and ("rep" in df.columns):
            df = df.copy()
            df["sample"] = df["group"].astype(str) + df["rep"].astype(int).astype(str)
        else:
            raise ValueError("Input must have 'sample' OR ('group' and 'rep') columns.")

    need.add("sample")
    for c in feature_cols:
        need.add(c)

    miss = [c for c in sorted(need) if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in input: {miss}")

    # Build sample_id
    if id_mode == "dataset":
        sample_id = df["dataset"].astype(str) + "__" + df["sample"].astype(str)
    elif id_mode == "pooled":
        sample_id = df["sample"].astype(str)
    else:
        raise ValueError("--id-mode must be dataset or pooled")

    df = df.copy()
    df["sample_id"] = sample_id

    # Group and compute embeddings
    samples = []
    embs = []
    nrows_list = []

    for sid, g in df.groupby("sample_id", sort=True):
        nrows = int(len(g))
        nrows_list.append(nrows)
        if nrows < min_rows:
            continue

        blocks = []
        for c in feature_cols:
            v = _to_float(g[c])
            blocks.append(_stats_block(v))
        emb = np.concatenate(blocks, axis=0).astype(np.float32)

        samples.append(str(sid))
        embs.append(emb)

    if not embs:
        raise RuntimeError("No embeddings produced. Try lowering --min-rows or check input data.")

    samples_arr = np.array(samples, dtype=object)
    emb_arr = np.stack(embs, axis=0).astype(np.float32)

    meta = {
        "id_mode": id_mode,
        "feature_cols": feature_cols,
        "block_dim_per_feature": 11,
        "emb_dim": int(emb_arr.shape[1]),
        "min_rows": int(min_rows),
        "n_groups_total": int(df["sample_id"].nunique()),
        "n_groups_kept": int(len(samples)),
        "rows_per_group_summary": {
            "min": int(np.min(nrows_list)) if nrows_list else 0,
            "p10": float(np.percentile(nrows_list, 10)) if nrows_list else 0.0,
            "p50": float(np.percentile(nrows_list, 50)) if nrows_list else 0.0,
            "p90": float(np.percentile(nrows_list, 90)) if nrows_list else 0.0,
            "max": int(np.max(nrows_list)) if nrows_list else 0,
        },
    }
    return samples_arr, emb_arr, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-long", required=True, help="site_sample_long.all.parquet")
    ap.add_argument("--out", required=True, help="output npz, e.g. proteome_bg_embeddings.npz")
    ap.add_argument("--id-mode", choices=["dataset", "pooled"], default="dataset",
                    help="dataset: dataset__WT1 unique; pooled: WT1 merged across datasets")
    ap.add_argument("--min-rows", type=int, default=1000,
                    help="skip groups with too few rows (avoid degenerate stats)")
    ap.add_argument("--no-palm-prot", action="store_true",
                    help="do not use palm_prot_abund even if present")
    args = ap.parse_args()

    site_long = Path(args.site_long)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[Load]", site_long)
    df = pd.read_parquet(site_long)

    # choose features
    feature_cols = ["prot_abund"]
    if (not args.no_palm_prot) and ("palm_prot_abund" in df.columns):
        feature_cols.append("palm_prot_abund")

    # build
    samples, emb, meta = build_embeddings(
        df=df,
        id_mode=args.id_mode,
        feature_cols=feature_cols,
        min_rows=args.min_rows,
    )

    # save
    np.savez_compressed(
        out,
        samples=samples,
        combined_emb=emb.astype(np.float32),
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )

    print(f"[DONE] {out}")
    print(f"  samples={len(samples)}  emb={emb.shape}  id_mode={args.id_mode}")
    print(f"  features={feature_cols}  emb_dim={emb.shape[1]}")
    print(f"  meta={meta}")


if __name__ == "__main__":
    main()
