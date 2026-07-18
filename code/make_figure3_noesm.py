#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- import model pieces from your training script ----------
# We only need: AA2IDX/IDX2AA/PAD_IDX/SEQ_LEN/CENTER_POS/encode_seq and Discriminator class.
# To avoid fragile imports, we re-define minimal constants + encoder consistent with your script.

SEQ_LEN = 101
CENTER_POS = 50
VOCAB = list("ACDEFGHIKLMNPQRSTVWYBX")
AA2IDX = {a: i for i, a in enumerate(VOCAB)}
IDX2AA = {i: a for a, i in AA2IDX.items()}
PAD_IDX = AA2IDX.get("X", len(VOCAB) - 1)
VOCAB_SIZE = len(VOCAB)

def normalize_seq101(s: str) -> str:
    s = (s or "").upper().strip()
    if len(s) >= SEQ_LEN:
        s = s[:SEQ_LEN]
    else:
        s = s + "X" * (SEQ_LEN - len(s))
    s = list(s)
    s[CENTER_POS] = "C"
    return "".join(s)

def encode_seq(seq101: str) -> np.ndarray:
    s = normalize_seq101(seq101)
    out = np.full((SEQ_LEN,), PAD_IDX, dtype=np.int64)
    for i, ch in enumerate(s):
        out[i] = AA2IDX.get(ch, PAD_IDX)
    out[CENTER_POS] = AA2IDX["C"]
    return out

class ConditionalSequenceDisc(nn.Module):
    def __init__(self, bg_dim: int, d_model=256, nhead=8, nlayers=4, dff=512, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        self.film = nn.Sequential(
            nn.Linear(bg_dim, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dff, dropout=dropout,
            activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor, bg_vec: torch.Tensor):
        x = self.emb(tokens)  # (B,L,D)
        gamma, beta = self.film(bg_vec).chunk(2, dim=-1)
        x = x * (1 + torch.tanh(gamma.unsqueeze(1))) + beta.unsqueeze(1)
        h = self.enc(x)
        h_mean = h.mean(dim=1)
        return self.cls(h_mean).squeeze(-1)

# ---------- io helpers ----------
def load_bg_db(bg_npz: str):
    """Load background embeddings.

    Supports:
      1) dict-like npz: each npz key is a bg_key -> vector
      2) packed npz: contains 'samples' (N,), 'combined_emb' (N,D), and optional 'meta'

    Returns:
      bg_map: dict[str, np.ndarray]
      bg_index: dict with optional aligned arrays: samples, dataset, cond
    """
    z = np.load(bg_npz, allow_pickle=True)

    # packed format
    if set(z.files) >= {"samples", "combined_emb"}:
        samples = z["samples"]
        emb = z["combined_emb"]
        if samples.ndim != 1 or emb.ndim != 2 or emb.shape[0] != samples.shape[0]:
            raise RuntimeError(f"Invalid packed bg npz: samples{samples.shape}, emb{emb.shape}")

        # build dict
        bg_map = {str(samples[i]): emb[i] for i in range(samples.shape[0])}

        ds = None
        cond = None
        if "meta" in z.files:
            meta = z["meta"]
            # meta may be a 0-d object array holding a dict
            try:
                meta_obj = meta.item() if meta.dtype == object and getattr(meta, "shape", None) == () else meta
            except Exception:
                meta_obj = meta

            if isinstance(meta_obj, dict):
                for k in ["dataset", "ds", "cell", "celltype"]:
                    if k in meta_obj:
                        ds = np.asarray(meta_obj[k])
                        break
                for k in ["condition", "cond", "group", "state"]:
                    if k in meta_obj:
                        cond = np.asarray(meta_obj[k])
                        break
            else:
                # structured array
                if hasattr(meta_obj, "dtype") and meta_obj.dtype.names:
                    names = list(meta_obj.dtype.names)
                    for k in ["dataset", "ds", "cell", "celltype"]:
                        if k in names:
                            ds = np.asarray(meta_obj[k])
                            break
                    for k in ["condition", "cond", "group", "state"]:
                        if k in names:
                            cond = np.asarray(meta_obj[k])
                            break

        bg_index = {"samples": np.asarray(samples).astype(str), "dataset": ds, "cond": cond}
        return bg_map, bg_index

    # dict-like format
    bg_map = {k: z[k] for k in z.files}
    bg_index = {"samples": np.asarray(list(bg_map.keys()), dtype=object).astype(str), "dataset": None, "cond": None}
    return bg_map, bg_index


def pick_reps(bg_map, dataset: str, cond: str, bg_index=None):
    """Pick bg keys for a dataset/condition.

    cond: "WT" or "KO"

    Priority:
      1) use bg_index['dataset'] and bg_index['cond'] if present (packed npz)
      2) legacy prefix match: dataset__COND
      3) substring match: contains dataset and cond (handles keys like Hela_WT_rep1)
    """
    # packed index
    if bg_index is not None:
        ds = bg_index.get("dataset", None)
        cc = bg_index.get("cond", None)
        smp = bg_index.get("samples", None)
        if ds is not None and cc is not None and smp is not None:
            ds = np.asarray(ds).astype(str)
            cc = np.asarray(cc).astype(str)
            smp = np.asarray(smp).astype(str)
            mask = (ds == str(dataset)) & (cc == str(cond))
            keys = smp[mask].tolist()
            if keys:
                return keys

    # legacy
    pref = dataset + "__" + cond
    keys = [k for k in bg_map.keys() if str(k).startswith(pref)]
    if keys:
        # sort stable: WT1,WT2...
        def repnum(k):
            suf = str(k).split("__", 1)[1] if "__" in str(k) else str(k)
            digits = "".join([c for c in suf if c.isdigit()])
            return int(digits) if digits else 0
        return sorted(keys, key=repnum)

    # substring fallback
    keys = [k for k in bg_map.keys() if (str(dataset) in str(k)) and (str(cond) in str(k))]
    if not keys:
        raise ValueError(
            f"No bg keys matched dataset={dataset} cond={cond}. "
            f"bg_npz keys look like: {list(bg_map.keys())[:5]}"
        )
    return keys


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

# ---------- parquet streaming ----------
def iter_parquet_batches(path: str, columns, batch_rows: int = 200000):
    """
    Stream parquet batches using pyarrow if available; fallback to pandas full-read.
    """
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches(batch_size=batch_rows, columns=columns):
            yield rb.to_pandas()
    except Exception as e:
        import pandas as pd
        df = pd.read_parquet(path, columns=columns)
        # yield in chunks to unify logic
        n = df.shape[0]
        for i in range(0, n, batch_rows):
            yield df.iloc[i:i + batch_rows].copy()

# ---------- PWM from sequences ----------
def pwm_from_seqs(seqs, win=10):
    # window excludes center? here: symmetric around center with center included
    L = 2 * win + 1
    mat = np.zeros((L, VOCAB_SIZE), dtype=np.float64)
    for s in seqs:
        s = normalize_seq101(s)
        seg = s[CENTER_POS - win: CENTER_POS + win + 1]
        for i, ch in enumerate(seg):
            mat[i, AA2IDX.get(ch, PAD_IDX)] += 1.0
    mat /= max(1.0, mat.sum(axis=1, keepdims=True))
    return mat

def plot_pwm_delta(pwm_a, pwm_b, outpng: Path, title: str):
    # show delta (a-b) for 20 AA only (exclude B/X) to keep readable
    aa_keep = list("ACDEFGHIKLMNPQRSTVWY")
    idx_keep = [AA2IDX[a] for a in aa_keep]
    delta = pwm_a[:, idx_keep] - pwm_b[:, idx_keep]

    plt.figure(figsize=(12, 4))
    plt.imshow(delta.T, aspect="auto")
    plt.yticks(range(len(aa_keep)), aa_keep)
    plt.xlabel("position (centered window)")
    plt.title(title)
    plt.colorbar(label="Δfreq")
    plt.tight_layout()
    outpng.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpng, dpi=200)
    plt.close()

# ---------- main scoring ----------
@torch.no_grad()
def score_batch(D, tokens_np, bg_vecs_np, device, micro_bs=4096):
    """
    tokens_np: (N,101) int64
    bg_vecs_np: (K,bgdim) float32 where K reps; we will score each rep and average
    returns: scores_mean (N,) float64 in [0,1]
    """
    N = tokens_np.shape[0]
    K = bg_vecs_np.shape[0]
    out = np.zeros((K, N), dtype=np.float64)

    tok = torch.from_numpy(tokens_np).long().to(device)

    for k in range(K):
        bg = torch.from_numpy(bg_vecs_np[k]).float().to(device).unsqueeze(0)  # (1,bgdim)
        # score in micro-batches
        preds = []
        for i in range(0, N, micro_bs):
            t = tok[i:i + micro_bs]
            b = bg.repeat(t.size(0), 1)
            # ensure inputs on same device as model
            dev = next(D.parameters()).device
            t = t.to(dev, non_blocking=True)
            b = b.to(dev, non_blocking=True)
            logit = D(t, b).detach().cpu().numpy()
            preds.append(sigmoid_np(logit))
        out[k] = np.concatenate(preds, axis=0)

    return out.mean(axis=0)
# ===================== CNN Discriminator (matches conditional_seq_gan_noesm.py) =====================
class ConditionalSequenceDiscCNN(nn.Module):
    """
    Matches your training discriminator exactly:

      self.emb: Embedding(VOCAB_SIZE, emb_dim=64, padding_idx=PAD_IDX)
      self.convs: ModuleList of parallel conv blocks:
         Conv1d(emb_dim -> channels, k, padding=k//2) + GELU + Dropout
      self.bg_ln: LayerNorm(bg_dim)
      feat_dim = channels * len(kernels) + bg_dim
      self.mlp: Linear(feat_dim -> 256) + GELU + Dropout + Linear(256 -> 1)

    Forward interface kept compatible:
        forward(tokens_or_soft, bg_vec, is_soft=False) -> logits (B,)
      - if is_soft=True, tokens_or_soft is (B,L,V) soft one-hot
      - else tokens_or_soft is (B,L) long
    """
    def __init__(self, bg_dim: int, emb_dim: int, channels: int, kernels, dropout: float = 0.0):
        super().__init__()
        self.bg_dim = int(bg_dim)
        self.emb = nn.Embedding(VOCAB_SIZE, int(emb_dim), padding_idx=PAD_IDX)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(int(emb_dim), int(channels), kernel_size=int(k), padding=int(k)//2),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
            for k in kernels
        ])

        self.bg_ln = nn.LayerNorm(self.bg_dim, eps=1e-5) if self.bg_dim > 0 else None

        feat_dim = int(channels) * len(kernels) + (self.bg_dim if self.bg_dim > 0 else 0)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 1),
        )
        # match training init
        nn.init.zeros_(self.mlp[-1].bias)

    def _embed(self, tokens_or_soft: torch.Tensor, is_soft: bool) -> torch.Tensor:
        if is_soft:
            # (B,L,V) @ (V,E) -> (B,L,E)
            return torch.matmul(tokens_or_soft, self.emb.weight)
        return self.emb(tokens_or_soft)

    def forward(self, tokens_or_soft: torch.Tensor, bg_vec: torch.Tensor, is_soft: bool = False):
        x = self._embed(tokens_or_soft, is_soft=is_soft)  # (B,L,E)
        x = x.transpose(1, 2)                             # (B,E,L)

        feats = []
        for block in self.convs:
            h = block(x)                 # (B,C,L)
            h = torch.amax(h, dim=-1)    # global max pool -> (B,C)
            feats.append(h)
        h = torch.cat(feats, dim=1)      # (B, C * nk)

        if self.bg_dim > 0:
            bg2 = self.bg_ln(bg_vec)
            h = torch.cat([h, bg2], dim=1)

        return self.mlp(h).squeeze(-1)   # (B,)


def _infer_cnn_hparams_from_state_dict(sd: dict):
    # emb dim
    if "emb.weight" not in sd:
        raise RuntimeError("CNN ckpt missing 'emb.weight'")
    vocab_size, emb_dim = sd["emb.weight"].shape

    # conv params: convs.{i}.0.weight shape (channels, emb_dim, k)
    kernels = []
    channels = None
    i = 0
    while f"convs.{i}.0.weight" in sd:
        w = sd[f"convs.{i}.0.weight"]
        out_ch, in_ch, k = w.shape
        if in_ch != emb_dim:
            raise RuntimeError(
                f"Unexpected CNN ckpt: convs.{i}.0.weight in_ch={in_ch} != emb_dim={emb_dim} "
                f"(this should be a parallel conv bank)."
            )
        if channels is None:
            channels = int(out_ch)
        elif int(out_ch) != int(channels):
            # support rare case: different out_ch per kernel
            pass
        kernels.append(int(k))
        i += 1
    if not kernels:
        raise RuntimeError("CNN ckpt has no convs.{i}.0.weight")

    # bg dim
    if "bg_ln.weight" in sd:
        bg_dim = int(sd["bg_ln.weight"].numel())
    else:
        bg_dim = 0

    # sanity: mlp input features must equal channels*len(kernels)+bg_dim
    if "mlp.0.weight" in sd:
        mlp_in = int(sd["mlp.0.weight"].shape[1])
        # allow variable out_ch per kernel by summing from weights
        sum_out = 0
        for j in range(len(kernels)):
            w = sd[f"convs.{j}.0.weight"]
            sum_out += int(w.shape[0])
        expected = sum_out + bg_dim
        if expected != mlp_in:
            raise RuntimeError(
                f"CNN feature dim mismatch inferred from ckpt: sum_out={sum_out}, bg_dim={bg_dim}, "
                f"expected={expected}, but mlp.0 expects in_features={mlp_in}. "
                f"This would mean the training D used extra pooled features (e.g., avgpool) or "
                f"different concat scheme. Please tell me the exact D forward if this happens."
            )

    return int(emb_dim), int(bg_dim), kernels


def build_discriminator_from_ckpt(sd: dict):
    # CNN discriminator (your current training script)
    if any(k.startswith("convs.") for k in sd.keys()):
        emb_dim, bg_dim, kernels = _infer_cnn_hparams_from_state_dict(sd)
        # channels inferred from conv weight out_ch (assume same across kernels; if not, still ok because we validate mlp)
        channels = int(sd["convs.0.0.weight"].shape[0])
        return ConditionalSequenceDiscCNN(bg_dim=bg_dim, emb_dim=emb_dim, channels=channels, kernels=kernels, dropout=0.0)

    # Otherwise fall back to the original Transformer-based discriminator defined above in this script
    # (ConditionalSequenceDisc)
    return ConditionalSequenceDisc(bg_dim=sd.get("bg_ln.weight", torch.empty(0)).numel() if "bg_ln.weight" in sd else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-table", required=True)
    ap.add_argument("--ckpt-dir", required=True, help=".../checkpoints containing D_best.pt or D.pt")
    ap.add_argument("--bg-npz", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--dataset", default="", help="If set, only run this dataset (e.g., Hela)")
    ap.add_argument("--only-positive", action="store_true", help="Only label_bin==1 sites")
    ap.add_argument("--max-sites", type=int, default=0, help="If >0, randomly subsample per dataset to this size")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--batch-rows", type=int, default=200000, help="parquet streaming batch rows")
    ap.add_argument("--micro-bs", type=int, default=4096, help="GPU micro batch for D scoring")

    ap.add_argument("--topk-sites", type=int, default=5000)
    ap.add_argument("--topk-proteins", type=int, default=2000)

    ap.add_argument("--pwm-win", type=int, default=10, help="for optional PWM delta heatmap")
    ap.add_argument("--make-pwm", action="store_true", help="also compute PWM delta using top ΔP sites")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load bg db
    bg_map, bg_index = load_bg_db(args.bg_npz)

    # determine datasets present in bg db (human/mouse etc.)
        # determine datasets present in bg db
    if bg_index is not None and bg_index.get('dataset', None) is not None:
        ds_all = sorted(list(set(np.asarray(bg_index['dataset']).astype(str).tolist())))
    else:
        ds_all = sorted(list({str(k).split('__', 1)[0] for k in bg_map.keys()}))
    if args.dataset:
        ds_list = [args.dataset]
    else:
        # infer from site table content later; start with bg db datasets
        ds_list = ds_all

    # load D
    ckpt_dir = Path(args.ckpt_dir)
    cand = [ckpt_dir / "D_best.pt", ckpt_dir / "D.pt", ckpt_dir / "D_best.pth", ckpt_dir / "D.pth"]
    ckpt_path = None
    for p in cand:
        if p.exists():
            ckpt_path = p
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"Cannot find D checkpoint in {ckpt_dir}. Tried: {cand}")

    # bg_dim from any sample
    bg_dim = int(next(iter(bg_map.values())).shape[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---- Build discriminator from checkpoint (auto-detect CNN vs Transformer) ----
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    D = build_discriminator_from_ckpt(sd)
    D.load_state_dict(sd, strict=True)


    # stream parquet and score per dataset
    # columns needed
    cols = ["seq101", "label_bin", "dataset", "sample", "Accession", "Position"]
    # NOTE: "Accession" and "Position" exist in your human parquet; mouse should be similar.
    # If your mouse parquet differs, adjust column names accordingly.

    # collect per-dataset arrays for histogram (we keep at most 2e6 to avoid RAM blow)
    hist_store = defaultdict(list)

    # keep topK sites/proteins using simple reservoir for each dataset
    top_sites = {ds: [] for ds in ds_list}   # list of (delta, rowdict)
    bottom_sites = {ds: [] for ds in ds_list}  # list of (-delta, rowdict) for most negative deltas
    prot_agg = {ds: defaultdict(lambda: {"sum": 0.0, "n": 0, "max": -1e9}) for ds in ds_list}

    # for PWM: store sequences of top deltas (small)
    pwm_keep_wt = {ds: [] for ds in ds_list}
    pwm_keep_ko = {ds: [] for ds in ds_list}

    # first pass: score all rows and update stats
    for df in iter_parquet_batches(args.site_table, columns=cols, batch_rows=args.batch_rows):
        # filter dataset
        if args.dataset:
            df = df[df["dataset"].astype(str) == args.dataset]
            if df.empty:
                continue

        if args.only_positive:
            df = df[df["label_bin"].astype(int) == 1]
            if df.empty:
                continue

        # optional subsample per batch (approximate) for speed
        if args.max_sites and df.shape[0] > args.max_sites:
            df = df.sample(n=int(args.max_sites), random_state=args.seed)

        # group by dataset within batch
        for ds, sub in df.groupby(df["dataset"].astype(str)):
            if ds not in ds_list:
                continue

            # prepare WT/KO bg vecs for this dataset
            wt_keys = pick_reps(bg_map, ds, "WT", bg_index)
            ko_keys = pick_reps(bg_map, ds, "KO", bg_index)
            wt_bg = np.stack([bg_map[k] for k in wt_keys], axis=0).astype(np.float32)
            ko_bg = np.stack([bg_map[k] for k in ko_keys], axis=0).astype(np.float32)

            seqs = sub["seq101"].astype(str).tolist()
            tokens = np.stack([encode_seq(s) for s in seqs], axis=0).astype(np.int64)

            score_wt = score_batch(D, tokens, wt_bg, device=device, micro_bs=args.micro_bs)
            score_ko = score_batch(D, tokens, ko_bg, device=device, micro_bs=args.micro_bs)
            delta = score_wt - score_ko

            # store for hist (cap)
            hist_store[ds].append(delta.astype(np.float32))

            # update top sites
            for i in range(len(seqs)):
                row = {
                    "dataset": ds,
                    "sample": str(sub["sample"].iloc[i]),
                    "Accession": str(sub["Accession"].iloc[i]),
                    "Position": str(sub["Position"].iloc[i]),
                    "label_bin": int(sub["label_bin"].iloc[i]),
                    "score_WT": float(score_wt[i]),
                    "score_KO": float(score_ko[i]),
                    "delta": float(delta[i]),
                    "seq101": seqs[i],
                }
                # maintain min-heap style list (simple sort since topk ~ few thousand)
                ts = top_sites[ds]
                ts.append((row["delta"], row))
                if len(ts) > args.topk_sites * 3:
                    ts.sort(key=lambda x: x[0], reverse=True)
                    del ts[args.topk_sites * 2:]  # keep buffer

                # maintain most-negative sites as well
                bs = bottom_sites[ds]
                bs.append((-row["delta"], row))
                if len(bs) > args.topk_sites * 3:
                    bs.sort(key=lambda x: x[0], reverse=True)
                    del bs[args.topk_sites * 2:]


                # protein aggregate
                acc = row["Accession"]
                pa = prot_agg[ds][acc]
                pa["sum"] += row["delta"]
                pa["n"] += 1
                if row["delta"] > pa["max"]:
                    pa["max"] = row["delta"]

            # PWM keep: collect sequences from strongest positive delta (WT-biased) and strongest negative delta (KO-biased)
            if args.make_pwm:
                # take top/bottom within this sub-batch
                idx_sort = np.argsort(delta)
                take = min(200, len(delta))  # small per batch
                # KO-biased (most negative): use as KO set; WT-biased (most positive): WT set
                for j in idx_sort[:take]:
                    pwm_keep_ko[ds].append(seqs[int(j)])
                for j in idx_sort[-take:]:
                    pwm_keep_wt[ds].append(seqs[int(j)])

    # write outputs per dataset
    import pandas as pd

    for ds in ds_list:
        if ds not in hist_store or len(hist_store[ds]) == 0:
            continue

        delta_all = np.concatenate(hist_store[ds], axis=0).astype(np.float64)
        # cap for plotting to avoid huge file
        if delta_all.size > 2_000_000:
            idx = rng.choice(delta_all.size, size=2_000_000, replace=False)
            delta_plot = delta_all[idx]
        else:
            delta_plot = delta_all

        # Fig3A: histogram (zoomed to show the bulk; plus full-range log-y)
        q01, q99 = np.quantile(delta_plot, [0.01, 0.99])
        q05, q95 = np.quantile(delta_plot, [0.05, 0.95])

        # zoomed
        plt.figure(figsize=(6,4))
        plt.hist(delta_plot, bins=120)
        plt.xlim(float(q01), float(q99))
        plt.axvline(float(q05), linestyle="--")
        plt.axvline(float(q95), linestyle="--")
        plt.xlabel("ΔP = score_WT - score_KO")
        plt.ylabel("count")
        plt.title(f"{ds}: ΔP bulk (p1={q01:.3g}, p99={q99:.3g})")
        plt.tight_layout()
        plt.savefig(outdir / f"fig3A_delta_hist_{ds}.png", dpi=200)
        plt.close()

        # full range (log y so tails are visible)
        plt.figure(figsize=(6,4))
        plt.hist(delta_plot, bins=160)
        plt.yscale("log")
        plt.xlabel("ΔP = score_WT - score_KO")
        plt.ylabel("count (log)")
        plt.title(f"{ds}: ΔP full range (log-y)")
        plt.tight_layout()
        plt.savefig(outdir / f"fig3A_delta_hist_full_{ds}.png", dpi=200)
        plt.close()

        # summary json
        summ = {
            "dataset": ds,
            "n_scored": int(delta_all.size),
            "delta_mean": float(np.mean(delta_all)),
            "delta_median": float(np.median(delta_all)),
            "delta_q05": float(np.quantile(delta_all, 0.05)),
            "delta_q95": float(np.quantile(delta_all, 0.95)),
        }
        with open(outdir / f"fig3A_summary_{ds}.json", "w", encoding="utf-8") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)

        # Fig3B sites: top ΔP
        ts = top_sites[ds]
        ts.sort(key=lambda x: x[0], reverse=True)
        ts = ts[:args.topk_sites]
        df_sites = pd.DataFrame([r for _, r in ts])
        df_sites.to_csv(outdir / f"fig3B_top_sites_{ds}.csv", index=False)

        # Fig3B sites: most negative ΔP (KO-biased)
        bs = bottom_sites[ds]
        bs.sort(key=lambda x: x[0], reverse=True)
        bs = bs[:args.topk_sites]
        df_sites_neg = pd.DataFrame([r for _, r in bs])
        df_sites_neg.to_csv(outdir / f"fig3B_bottom_sites_{ds}.csv", index=False)

        # Outlier site table (union of top/bottom)
        df_out = pd.concat([df_sites, df_sites_neg], ignore_index=True)
        # drop exact duplicates if any
        df_out.drop_duplicates(subset=["dataset","Accession","Position","sample","seq101"], inplace=True, ignore_index=True)
        df_out.sort_values("delta", ascending=False, inplace=True)
        df_out.to_csv(outdir / f"fig3A_outlier_sites_{ds}.csv", index=False)


        # Fig3B proteins: aggregate
        prot_rows = []
        for acc, st in prot_agg[ds].items():
            if st["n"] <= 0:
                continue
            prot_rows.append({
                "dataset": ds,
                "Accession": acc,
                "n_sites": int(st["n"]),
                "delta_mean": float(st["sum"] / st["n"]),
                "delta_max": float(st["max"]),
            })
        df_prot = pd.DataFrame(prot_rows)
        df_prot.sort_values(["delta_mean", "delta_max"], ascending=False, inplace=True)
        df_prot.head(args.topk_proteins).to_csv(outdir / f"fig3B_top_proteins_{ds}.csv", index=False)

        # Optional Fig3C PWM delta heatmap (simple)
        if args.make_pwm and (len(pwm_keep_wt[ds]) >= 1000) and (len(pwm_keep_ko[ds]) >= 1000):
            # downsample to stable size
            wt_seqs = pwm_keep_wt[ds]
            ko_seqs = pwm_keep_ko[ds]
            if len(wt_seqs) > 20000:
                wt_seqs = list(rng.choice(wt_seqs, size=20000, replace=False))
            if len(ko_seqs) > 20000:
                ko_seqs = list(rng.choice(ko_seqs, size=20000, replace=False))

            pwm_wt = pwm_from_seqs(wt_seqs, win=args.pwm_win)
            pwm_ko = pwm_from_seqs(ko_seqs, win=args.pwm_win)
            plot_pwm_delta(
                pwm_wt, pwm_ko,
                outpng=outdir / f"fig3C_pwm_delta_{ds}.png",
                title=f"{ds}: PWM Δfreq (WT-biased minus KO-biased) (win={args.pwm_win})"
            )

    print(f"[OK] Figure3 outputs saved to: {outdir}")

if __name__ == "__main__":
    main()