#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_sitelevel_dataset_v2.py

Build site-level palmitoylation dataset with:
- prot_abund (Protein_Expression_*.csv)
- palm_prot_abund (Palmitoylayion_Protein_*.csv)
- palm_site_abund (Palmitoylayion_Site_*.csv or Site_Expression_Liver.csv)
- labels (palm/nopalm csv) + loose negatives (negative fasta)
- outputs human/mouse/all splits
- outputs a wide table compatible with legacy training script:
    accession, prot_seq_101aa_rep,
    PROT_WT1..5, PROT_KO1..5,
    PALM_WT1..5, PALM_KO1..5
  (Note: here PROT/PALM are per-site (Position) but columns match your model interface.)

Expected folder structure:
  data/integration_model/{Hela, PANC-1, Mouse_liver}/

For each dataset folder, auto-detect:
  - Protein_Expression_*.csv
  - Palmitoylayion_Protein_*.csv
  - Palmitoylayion_Site_*.csv (or Site_Expression_Liver.csv)
  - *_cys_101aa_palm.csv
  - *_cys_101aa_nopalm.csv
  - *_cys_negative.fasta or Liver_cys_negative.fasta (optional)

Outputs (under --outdir):
  - site_sample_long.all.parquet
  - site_sample_long.human.parquet
  - site_sample_long.mouse.parquet
  - site_sample_wide_legacy.all.csv
  - unique_seq101.tsv
  - qc_report.json

Run:
  python build_sitelevel_dataset_v2.py \
    --root data/integration_model \
    --outdir data/derived_sitelevel \
    --max-rep 5 \
    --prot-threshold 0.0 \
    --loose-broadcast all

"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from Bio import SeqIO
    HAS_BIO = True
except Exception:
    HAS_BIO = False

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$", re.IGNORECASE)

# sample parsing: KO1/WT2 etc.
RE_WT = re.compile(r"\bWT\b|WTCON|WT_CON|WTCTRL|WT_CTRL|^WT\d+$", re.IGNORECASE)
RE_KO = re.compile(r"\bKO\b|KOCON|KO_CON|KOCTRL|KO_CTRL|^KO\d+$", re.IGNORECASE)

def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).strip()

def _parse_panc_ctrl_exp_sample(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Parse PANC-1 style columns:
      ctrl1/ctrl2/ctrl3 -> WT1/WT2/WT3
      exp1/exp2/exp3    -> KO1/KO2/KO3
    Also supports ctrl_1, ctrl-2, exp_3, 'ctrl 1' etc.
    """
    s = str(col).strip().lower()
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(ctrl|exp)[_-]?(\d+)$", s)
    if not m:
        return None
    grp = "WT" if m.group(1) == "ctrl" else "KO"
    rep = int(m.group(2))
    if 1 <= rep <= max_rep:
        return grp, rep
    return None



def _parse_simple_sample(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Parse sample names like WT1/KO3, also supports WT_1 / KO-2 / WT 3.
    """
    s = str(col).strip().upper()
    s = re.sub(r"\s+", "", s)  # remove spaces

    # allow separators between group and rep: _, -, nothing
    m = re.match(r"^(WT|KO)[_-]?(\d+)$", s)
    if not m:
        return None
    grp = m.group(1)
    rep = int(m.group(2))
    if 1 <= rep <= max_rep:
        return grp, rep
    return None

def _parse_label_sample_any(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Unified parser for label tables (palm / nopalm).

    Supports:
      WT1, WT_1, KO2
      ctrl1, ctrl_2  -> WT
      exp1, exp_2    -> KO
    """
    # try WT/KO first
    pr = _parse_simple_sample(col, max_rep=max_rep)
    if pr is not None:
        return pr

    # fallback: PANC-1 ctrl/exp
    s = str(col).strip().lower()
    m = re.match(r"^(ctrl|exp)\s*[_-]?\s*(\d+)$", s)
    if not m:
        return None

    grp_raw = m.group(1)
    rep = int(m.group(2))
    if not (1 <= rep <= max_rep):
        return None

    grp = "WT" if grp_raw == "ctrl" else "KO"
    return grp, rep


def _parse_abundance_dash_sample(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Parse:
      'Abundance: WT-1', 'Abundance: KO-3', (allow extra spaces)
    """
    s = str(col).strip()
    m = re.match(r"^Abundance:\s*(WT|KO)\s*-\s*(\d+)\s*$", s, flags=re.IGNORECASE)
    if not m:
        return None
    grp = m.group(1).upper()
    rep = int(m.group(2))
    if 1 <= rep <= max_rep:
        return grp, rep
    return None


def _parse_site_sample(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Parse sample names like WTCON-1, KOCON-3.
    """
    n = _norm(col).upper()
    m = re.search(r"(WT|KO)CON[-_]?(\d+)$", n)
    if not m:
        return None
    grp = m.group(1)
    rep = int(m.group(2))
    if 1 <= rep <= max_rep:
        return grp, rep
    return None

def _find_one(folder: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None

def _read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows) if nrows else pd.read_csv(path)

def _valid_seq101(s: str) -> bool:
    s = (s or "").strip().upper()
    if len(s) != 101:
        return False
    if not AA_RE.match(s):
        return False
    return s[50] == "C"

def _read_fasta_seqs(path: Path) -> List[str]:
    if path is None or (not path.exists()):
        return []
    if not HAS_BIO:
        raise RuntimeError("Biopython not installed. `pip install biopython` to use negative fasta.")
    seqs = []
    for rec in SeqIO.parse(str(path), "fasta"):
        s = str(rec.seq).strip().upper()
        if _valid_seq101(s):
            seqs.append(s)
    return seqs

def _species_of_dataset(ds: str) -> str:
    d = ds.lower()
    if d in ("hela", "panc-1", "panc1"):
        return "human"
    if "mouse" in d or "liver" in d:
        return "mouse"
    # default: treat as human
    return "human"

def _dataset_name_norm(ds: str) -> str:
    return ds.replace(" ", "_")

def _pick_seq101(palm_row_seq: str, site_row_seq: str) -> str:
    a = (palm_row_seq or "").strip().upper()
    b = (site_row_seq or "").strip().upper()
    if _valid_seq101(a):
        return a
    if _valid_seq101(b):
        return b
    # fallback: return whichever non-empty
    return a if a else b

# -------------------------
# Loaders for each table
# -------------------------

def load_palm_label_table(palm_csv: Path, ds: str, max_rep: int) -> pd.DataFrame:
    """
    Robust loader for *_cys_101aa_palm.csv

    Expected:
      - Position column exists
      - Accession column can be: Protein / Accession / PG.ProteinAccessions / PG.ProteinGroups ...
      - seq101 column can be: Sequence_Fragment / window / prot_seq_101aa_rep / cysteine_sequences_101 ...
      - label columns: WT1..WTk, KO1..KOk (0/1)
    """
    df = _read_csv(palm_csv)

    if "Position" not in df.columns:
        raise ValueError(f"[{ds}] missing Position in {palm_csv}")

    # infer accession col
    acc_col = infer_accession_col(list(df.columns))
    if acc_col is None:
        raise ValueError(f"[{ds}] cannot infer accession col in {palm_csv}. cols={df.columns.tolist()[:30]}")

    # infer seq col
    seq_cands = ["Sequence_Fragment", "window", "prot_seq_101aa_rep", "cysteine_sequences_101", "Sequence", "seq101"]
    seq_col = next((c for c in seq_cands if c in df.columns), None)
    if seq_col is None:
        raise ValueError(f"[{ds}] cannot infer seq101 col in {palm_csv}. cols={df.columns.tolist()[:30]}")

    # label columns WT1.. KO1..
    label_cols = []
    for c in df.columns:
        pr = _parse_simple_sample(c, max_rep=max_rep)
        if pr is not None:
            label_cols.append(c)

    # fallback for PANC-1 ctrl/exp replicates
    if not label_cols:
        for c in df.columns:
            pr = _parse_panc_ctrl_exp_sample(c, max_rep=max_rep)
            if pr is not None:
                label_cols.append(c)

    if not label_cols:
        raise ValueError(f"[{ds}] cannot find WT/KO label cols in {palm_csv}")

    out = df[["Position", acc_col, seq_col] + label_cols].copy()
    out = out.rename(columns={acc_col: "Accession", seq_col: "seq_palm"})
    out["Position"] = out["Position"].astype(str)
    out["Accession"] = out["Accession"].astype(str)
    out["seq_palm"] = out["seq_palm"].fillna("").astype(str).str.upper()

    m = out.melt(id_vars=["Position", "Accession", "seq_palm"], value_vars=label_cols,
                 var_name="sample", value_name="label_bin")
    m["label_bin"] = pd.to_numeric(m["label_bin"], errors="coerce").fillna(0).astype(int)
    m["_pr"] = m["sample"].map(lambda x: _parse_label_sample_any(x, max_rep=max_rep))
    if m["_pr"].isnull().any():
        bad = m.loc[m["_pr"].isnull(), "sample"].unique()[:10]
        raise ValueError(f"[{ds}] unparsed label sample cols: {bad}")

    m["group"] = m["_pr"].map(lambda x: x[0])
    m["rep"] = m["_pr"].map(lambda x: x[1]).astype(int)
    m["sample"] = m["group"] + m["rep"].astype(str)
    m = m.drop(columns=["_pr"])

    # keep only positives from palm file
    m = m[m["label_bin"] == 1].copy()
    m["neg_type"] = "none"
    return m

def infer_accession_col(cols: List[str]) -> Optional[str]:
    """
    Infer accession-like column name from a list of columns.
    Accepts variants like: Protein, Proteind, Accession, UniProt, Entry, PG.ProteinAccessions, etc.
    """
    # exact preferred names (case-insensitive)
    preferred = {
        "protein", "proteind", "accession", "uniprot", "entry",
        "pg.proteinaccessions", "pg.proteingroups", "pg.proteinaccession"
    }
    low_map = {c.lower(): c for c in cols}
    for k in preferred:
        if k in low_map:
            return low_map[k]

    # fuzzy: any column that starts with protein / contains accession/uniprot/entry
    for c in cols:
        cl = c.lower()
        if cl.startswith("protein") and cl not in ("proteinname",):
            return c
        if ("accession" in cl) or ("uniprot" in cl) or (cl == "entry"):
            return c
    return None


def load_nopalm_label_table(nopalm_csv: Path, ds: str, max_rep: int) -> pd.DataFrame:
    """
    Robust loader for *_cys_101aa_nopalm.csv

    Output long columns:
      Position, Accession, seq_palm, sample(WT1/KO1), group(WT/KO), rep(int),
      label_bin(0), neg_type('strict')
    """
    df = _read_csv(nopalm_csv)

    if "Position" not in df.columns:
        raise ValueError(f"[{ds}] missing Position in {nopalm_csv}")

    # infer accession col
    acc_col = infer_accession_col(list(df.columns))
    if acc_col is None:
        raise ValueError(f"[{ds}] cannot infer accession col in {nopalm_csv}. cols={df.columns.tolist()[:30]}")

    # infer seq col
    seq_cands = ["Sequence_Fragment", "window", "prot_seq_101aa_rep", "cysteine_sequences_101", "Sequence", "seq101"]
    seq_col = next((c for c in seq_cands if c in df.columns), None)
    if seq_col is None:
        raise ValueError(f"[{ds}] cannot infer seq101 col in {nopalm_csv}. cols={df.columns.tolist()[:30]}")

    # collect label columns (WT/KO or ctrl/exp)
    label_cols = []
    parsed = {}

    for c in df.columns:
        pr = _parse_simple_sample(c, max_rep=max_rep)
        if pr is None:
            pr = _parse_panc_ctrl_exp_sample(c, max_rep=max_rep)  # ctrl->WT, exp->KO
        if pr is not None:
            label_cols.append(c)
            parsed[c] = pr

    if not label_cols:
        raise ValueError(f"[{ds}] cannot find label cols (WT/KO or ctrl/exp) in {nopalm_csv}")

    out = df[["Position", acc_col, seq_col] + label_cols].copy()
    out = out.rename(columns={acc_col: "Accession", seq_col: "seq_palm"})
    out["Position"] = out["Position"].astype(str)
    out["Accession"] = out["Accession"].astype(str)
    out["seq_palm"] = out["seq_palm"].fillna("").astype(str).str.upper()

    m = out.melt(
        id_vars=["Position", "Accession", "seq_palm"],
        value_vars=label_cols,
        var_name="sample_raw",
        value_name="_v"
    )

    # strict negatives: label=0 regardless of _v
    m["label_bin"] = 0
    m["group"] = m["sample_raw"].map(lambda x: parsed[x][0])
    m["rep"]   = m["sample_raw"].map(lambda x: parsed[x][1]).astype(int)
    m["sample"] = m["group"] + m["rep"].astype(str)
    m["neg_type"] = "strict"

    m = m.drop(columns=["_v", "sample_raw"])
    return m


def infer_position_col(df: pd.DataFrame) -> Optional[str]:
    """
    Infer a 'Position' / site-id column.

    Priority:
      1) exact 'Position'
      2) any column name containing 'position' (case-insensitive)
      3) a column where values look like 'ACC-(C39)' (string contains '-(C' and ')')
    """
    cols = list(df.columns)
    if "Position" in cols:
        return "Position"

    for c in cols:
        if "position" in c.lower():
            return c

    # value-pattern probe
    probe_cols = [c for c in cols if df[c].dtype == object or str(df[c].dtype).startswith("string")]
    for c in probe_cols[:50]:
        s = df[c].astype(str).head(200)
        hit = s.str.contains(r"-\(C\d+\)", regex=True, na=False).mean()
        if hit > 0.5:  # majority look like ACC-(Cxx)
            return c
    return None


def infer_seq101_col(df: pd.DataFrame) -> Optional[str]:
    seq_cands = ["window", "Sequence_Fragment", "seq101", "prot_seq_101aa_rep",
                 "cysteine_sequences_101", "Sequence", "sequence"]
    low = {c.lower(): c for c in df.columns}
    for k in seq_cands:
        if k.lower() in low:
            return low[k.lower()]
    # fuzzy
    for c in df.columns:
        cl = c.lower()
        if "window" in cl:
            return c
        if "sequence" in cl and "101" in cl:
            return c
    return None

def _parse_abundance_sample(col: str, max_rep: int = 5) -> Optional[Tuple[str, int]]:
    """
    Parse sample columns like:
      'Abundance:WT1'
      'Abundance: KO3'
      'Abundance:WT5'
    Return ('WT'/'KO', rep)
    """
    s = str(col).strip()
    m = re.match(r"^Abundance:\s*(WT|KO)\s*(\d+)\s*$", s, flags=re.IGNORECASE)
    if not m:
        return None
    grp = m.group(1).upper()
    rep = int(m.group(2))
    if 1 <= rep <= max_rep:
        return grp, rep
    return None


def load_site_abundance_table(site_csv: Path, ds: str, max_rep: int) -> pd.DataFrame:
    """
    Robust site-abundance loader for both human/mouse.

    Output long columns:
      Position, Accession, seq_site, sample(WT1/KO1), group, rep, palm_site_abund

    Accepts cases where:
      - Position column may be missing
      - Accession column may be Proteind/Protein/Accession/...
      - Position can be constructed via Accession + Site (Cys index)
    """
    df = _read_csv(site_csv)

    # infer accession
    acc_col = infer_accession_col(list(df.columns))
    if acc_col is None:
        # some mouse tables might use 'Proteind' explicitly
        if "Proteind" in df.columns:
            acc_col = "Proteind"
        else:
            raise ValueError(f"[{ds}] cannot infer Accession col in {site_csv}. cols={df.columns.tolist()[:40]}")

    # infer position
    pos_col = infer_position_col(df)

    # if still none, try to construct from (Accession + Site/Cys location)
    if pos_col is None:
        # candidates for site index / location
        site_cands = ["Site", "PTM.SiteLocation", "PTM_SiteLocation", "SiteLocation", "CysSite", "Cys", "site"]
        site_col = next((c for c in site_cands if c in df.columns), None)
        if site_col is None:
            # fuzzy match 'site' and 'location'
            for c in df.columns:
                cl = c.lower()
                if "site" in cl and ("loc" in cl or "location" in cl or cl == "site"):
                    site_col = c
                    break
        if site_col is None:
            raise ValueError(f"[{ds}] missing Position and cannot find site index col in {site_csv}. cols={df.columns.tolist()[:40]}")

        # build Position like ACC-(C39)
        tmp_pos = df[acc_col].astype(str).str.strip() + "-(C" + pd.to_numeric(df[site_col], errors="coerce").fillna(-1).astype(int).astype(str) + ")"
        df = df.copy()
        df["Position"] = tmp_pos
        pos_col = "Position"

    # infer seq col (optional)
    seq_col = infer_seq101_col(df)

    # infer sample columns: WTCON-1 style or WT1 style
    sample_cols = []
    parsed = {}

    # 1) WTCON-1 / KOCON-3
    for c in df.columns:
        pr = _parse_site_sample(c, max_rep=max_rep)
        if pr is not None:
            sample_cols.append(c)
            parsed[c] = pr

    # 2) WT1 / KO2
    if not sample_cols:
        for c in df.columns:
            pr = _parse_simple_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    # 3) Abundance:WT1 / Abundance: KO3  (Mouse_liver)
    if not sample_cols:
        for c in df.columns:
            pr = _parse_abundance_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    # 4) PANC-1 ctrl/exp replicates: ctrl1..3, exp1..3
    if not sample_cols:
        for c in df.columns:
            pr = _parse_panc_ctrl_exp_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    if not sample_cols:
        raise ValueError(
            f"[{ds}] cannot find sample columns in site table: {site_csv}. cols={df.columns.tolist()[:50]}")

    keep = [pos_col, acc_col] + ([seq_col] if seq_col else []) + sample_cols
    tmp = df[keep].copy()
    tmp = tmp.rename(columns={pos_col: "Position", acc_col: "Accession"})
    tmp["Position"] = tmp["Position"].astype(str)
    tmp["Accession"] = tmp["Accession"].astype(str)

    if seq_col:
        tmp = tmp.rename(columns={seq_col: "seq_site"})
        tmp["seq_site"] = tmp["seq_site"].fillna("").astype(str).str.upper()
    else:
        tmp["seq_site"] = ""

    for c in sample_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype(np.float32)

    long = tmp.melt(id_vars=["Position", "Accession", "seq_site"], value_vars=sample_cols,
                    var_name="sample_raw", value_name="palm_site_abund")
    long["group"] = long["sample_raw"].map(lambda x: parsed[x][0])
    long["rep"] = long["sample_raw"].map(lambda x: parsed[x][1]).astype(int)
    long["sample"] = long["group"] + long["rep"].astype(str)
    long = long.drop(columns=["sample_raw"])
    return long

def load_protein_expression_table(prot_csv: Path, ds: str, max_rep: int) -> pd.DataFrame:
    """
    Protein_Expression_Hela.csv:
      Accession, ..., KO1..WT3
    Output long:
      Accession, sample(WT1/KO1), prot_abund
    """
    df = _read_csv(prot_csv)
    if "Accession" not in df.columns:
        raise ValueError(f"[{ds}] missing Accession in {prot_csv}")

    sample_cols = []
    parsed = {}

    # 1) WT1 / KO2
    for c in df.columns:
        pr = _parse_simple_sample(c, max_rep=max_rep)
        if pr is not None:
            sample_cols.append(c)
            parsed[c] = pr

    # 2) Abundance:WT1 / Abundance: KO3
    if not sample_cols:
        for c in df.columns:
            pr = _parse_abundance_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    # 3) Abundance: WT-1 / Abundance: KO-5  (Mouse liver protein expression)
    if not sample_cols:
        for c in df.columns:
            pr = _parse_abundance_dash_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr
    # 4) PANC-1 ctrl/exp replicates: ctrl1..3, exp1..3
    if not sample_cols:
        for c in df.columns:
            pr = _parse_panc_ctrl_exp_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    if not sample_cols:
        raise ValueError(f"[{ds}] cannot find WT/KO cols in {prot_csv}")

    tmp = df[["Accession"] + sample_cols].copy()
    tmp["Accession"] = tmp["Accession"].astype(str)
    for c in sample_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype(np.float32)

    long = tmp.melt(id_vars=["Accession"], value_vars=sample_cols,
                    var_name="sample_raw", value_name="prot_abund")
    long["group"] = long["sample_raw"].map(lambda x: parsed[x][0])
    long["rep"] = long["sample_raw"].map(lambda x: parsed[x][1]).astype(int)
    long["sample"] = long["group"] + long["rep"].astype(str)
    long = long.drop(columns=["sample_raw"])
    return long


def load_palm_protein_table(palm_prot_csv: Path, ds: str, max_rep: int) -> pd.DataFrame:
    """
    Palmitoylayion_Protein_*.csv

    Human style:
      Accession, KO1..WT3
    Mouse style (likely):
      Abundance: WT-1 .. Abundance: KO-5  (or Abundance:WT1 / Abundance: KO3)

    Output long:
      Accession, sample(WT1/KO1), group, rep, palm_prot_abund
    """
    df = _read_csv(palm_prot_csv)
    if "Accession" not in df.columns:
        raise ValueError(f"[{ds}] missing Accession in {palm_prot_csv}")

    sample_cols = []
    parsed = {}

    # 1) WT1 / KO2
    for c in df.columns:
        pr = _parse_simple_sample(c, max_rep=max_rep)
        if pr is not None:
            sample_cols.append(c)
            parsed[c] = pr

    # 2) Abundance:WT1 / Abundance: KO3
    if not sample_cols:
        for c in df.columns:
            pr = _parse_abundance_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    # 3) Abundance: WT-1 / Abundance: KO-5
    if not sample_cols:
        for c in df.columns:
            pr = _parse_abundance_dash_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    # 4) PANC-1 ctrl/exp replicates: ctrl1..3, exp1..3
    if not sample_cols:
        for c in df.columns:
            pr = _parse_panc_ctrl_exp_sample(c, max_rep=max_rep)
            if pr is not None:
                sample_cols.append(c)
                parsed[c] = pr

    if not sample_cols:
        raise ValueError(f"[{ds}] cannot find WT/KO cols in {palm_prot_csv}")

    tmp = df[["Accession"] + sample_cols].copy()
    tmp["Accession"] = tmp["Accession"].astype(str)
    for c in sample_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0).astype(np.float32)

    long = tmp.melt(id_vars=["Accession"], value_vars=sample_cols,
                    var_name="sample_raw", value_name="palm_prot_abund")
    long["group"] = long["sample_raw"].map(lambda x: parsed[x][0])
    long["rep"] = long["sample_raw"].map(lambda x: parsed[x][1]).astype(int)
    long["sample"] = long["group"] + long["rep"].astype(str)
    long = long.drop(columns=["sample_raw"])
    return long


# -------------------------
# Build per dataset
# -------------------------

def build_one_dataset(folder: Path,
                      max_rep: int,
                      prot_threshold: float,
                      loose_broadcast: str) -> Tuple[pd.DataFrame, Dict]:
    ds = _dataset_name_norm(folder.name)
    species = _species_of_dataset(ds)

    prot_csv = _find_one(folder, ["Protein_Expression*.csv", "*Protein_Expression*.csv"])
    palm_prot_csv = _find_one(folder, ["Palmitoylayion_Protein*.csv", "*Palmitoy*Protein*.csv"])
    site_csv = _find_one(folder, ["Site_Expression*.csv", "Palmitoylayion_Site*.csv", "*Palmitoy*Site*.csv"])
    palm_csv = _find_one(folder, ["*_cys_101aa_palm.csv", "*cys_101aa_palm.csv"])
    nopalm_csv = _find_one(folder, ["*_cys_101aa_nopalm.csv", "*cys_101aa_nopalm.csv"])
    neg_fa = _find_one(folder, ["*_cys_negative.fasta", "*cys_negative.fasta", "*_cys_negative*.fasta"])

    # fallback: some mouse site tables don't have "Position"; prefer Site_Expression*.csv if present
    if site_csv is not None:
        _head = pd.read_csv(site_csv, nrows=1)
        if "Position" not in _head.columns:
            alt = _find_one(folder, ["Site_Expression*.csv"])
            if alt is not None and alt != site_csv:
                print(f"[{ds}] site table '{site_csv.name}' has no Position; fallback to '{alt.name}'")
                site_csv = alt

    if prot_csv is None or palm_prot_csv is None or site_csv is None or palm_csv is None or nopalm_csv is None:
        raise FileNotFoundError(
            f"[{ds}] missing required files.\n"
            f"prot={prot_csv}\n"
            f"palm_prot={palm_prot_csv}\n"
            f"site={site_csv}\n"
            f"palm_label={palm_csv}\n"
            f"nopalm_label={nopalm_csv}"
        )

    # label sets
    pos = load_palm_label_table(palm_csv, ds, max_rep=max_rep)      # (Position, Accession, seq_palm, sample WT1..)
    neg = load_nopalm_label_table(nopalm_csv, ds, max_rep=max_rep)  # strict negatives

    labels = pd.concat([pos, neg], axis=0, ignore_index=True)

    # site abundance (Position+sample)
    site_long = load_site_abundance_table(site_csv, ds, max_rep=max_rep)

    # protein expression (Accession+sample)
    prot_long = load_protein_expression_table(prot_csv, ds, max_rep=max_rep)

    # palm protein (Accession+sample)
    palm_prot_long = load_palm_protein_table(palm_prot_csv, ds, max_rep=max_rep)

    # join core: labels + site abundance
    # labels has Position, Accession, seq_palm, sample(WT1..), group, rep, label_bin, neg_type
    # site_long has Position, Accession, seq_site, sample(WT1..), group, rep, palm_site_abund
    core = labels.merge(
        site_long,
        on=["Position", "Accession", "sample", "group", "rep"],
        how="left",
        suffixes=("", "_site")
    )
    core["palm_site_abund"] = pd.to_numeric(core["palm_site_abund"], errors="coerce").fillna(0.0).astype(np.float32)

    # attach prot abundance by Accession+sample
    core = core.merge(
        prot_long[["Accession", "sample", "prot_abund"]],
        on=["Accession", "sample"],
        how="left"
    )
    core["prot_abund"] = pd.to_numeric(core["prot_abund"], errors="coerce").fillna(0.0).astype(np.float32)

    # attach palm protein abundance
    core = core.merge(
        palm_prot_long[["Accession", "sample", "palm_prot_abund"]],
        on=["Accession", "sample"],
        how="left"
    )
    core["palm_prot_abund"] = pd.to_numeric(core["palm_prot_abund"], errors="coerce").fillna(0.0).astype(np.float32)

    # choose seq101: seq_palm preferred, else seq_site
    core["seq_site"] = core.get("seq_site", "").fillna("").astype(str).str.upper()
    core["seq_palm"] = core["seq_palm"].fillna("").astype(str).str.upper()
    core["seq101"] = [
        _pick_seq101(a, b) for a, b in zip(core["seq_palm"].tolist(), core["seq_site"].tolist())
    ]
    core["seq_ok"] = core["seq101"].map(_valid_seq101).astype(np.int8)

    before_seq = len(core)
    core = core[core["seq_ok"] == 1].copy()
    after_seq = len(core)

    # strict negative consistency QC: strict but palm_site_abund > 0
    strict_conflict = int(((core["neg_type"] == "strict") & (core["palm_site_abund"] > 0)).sum())
    # positive consistency QC: none but palm_site_abund == 0
    pos_missing = int(((core["neg_type"] == "none") & (core["palm_site_abund"] <= 0)).sum())

    # regression mask (match your training idea): only count when prot>threshold and palm_site>0
    core["mask_reg"] = ((core["prot_abund"] > float(prot_threshold)) & (core["palm_site_abund"] > 0)).astype(np.float32)

    # add dataset/species
    core["dataset"] = ds
    core["species"] = species

    # add loose negatives from fasta
    loose_seqs = _read_fasta_seqs(neg_fa) if neg_fa is not None else []
    loose_seqs = list(dict.fromkeys(loose_seqs))  # unique keep order
    n_loose_added = 0
    if loose_seqs:
        # determine broadcast samples
        sample_keys = core[["sample", "group", "rep"]].drop_duplicates()
        if loose_broadcast == "human":
            # still broadcast within dataset; filter is handled at output split level
            pass
        elif loose_broadcast == "mouse":
            pass
        elif loose_broadcast == "all":
            pass
        else:
            raise ValueError("--loose-broadcast must be all/human/mouse")

        lo = pd.DataFrame({"seq101": loose_seqs})
        lo["Position"] = "NA"
        lo["Accession"] = "NA"
        lo["seq_palm"] = ""
        lo["seq_site"] = ""
        lo["label_bin"] = 0
        lo["neg_type"] = "loose"
        lo["palm_site_abund"] = 0.0
        lo["palm_prot_abund"] = 0.0
        lo["prot_abund"] = 0.0
        lo["mask_reg"] = 0.0
        lo["dataset"] = ds
        lo["species"] = species

        # broadcast to samples observed in this dataset
        lo = lo.merge(sample_keys, how="cross")
        n_loose_added = int(len(lo))
        core = pd.concat([core.drop(columns=["seq_ok"]), lo], axis=0, ignore_index=True)
    else:
        core = core.drop(columns=["seq_ok"])

    meta = {
        "dataset": ds,
        "species": species,
        "files": {
            "protein_expression": str(prot_csv),
            "palm_protein": str(palm_prot_csv),
            "palm_site": str(site_csv),
            "palm_label": str(palm_csv),
            "nopalm_label": str(nopalm_csv),
            "negative_fasta": str(neg_fa) if neg_fa else None,
        },
        "rows_before_seq_filter": before_seq,
        "rows_after_seq_filter": after_seq,
        "strict_conflict_count": strict_conflict,
        "pos_missing_site_abund_count": pos_missing,
        "loose_neg_seqs": int(len(loose_seqs)),
        "loose_neg_rows_added": n_loose_added,
        "pos_rate_after": float(core["label_bin"].mean()) if len(core) else 0.0,
        "unique_seq101": int(core["seq101"].nunique()) if len(core) else 0,
    }
    return core, meta

# -------------------------
# Wide legacy table builder
# -------------------------

def to_legacy_wide(df_long: pd.DataFrame, max_rep: int = 5) -> pd.DataFrame:
    """
    Build a legacy-wide table matching your training script column names.
    One row = one site (Position).
    Columns:
      accession, prot_seq_101aa_rep,
      PROT_WT1.., PROT_KO1..,
      PALM_WT1.., PALM_KO1..
    Here:
      - accession uses Accession (protein)
      - prot_seq_101aa_rep uses seq101 (representative)
      - PROT_* uses prot_abund per sample
      - PALM_* uses palm_site_abund per sample (site-level)
    """
    # choose representative seq per (Accession, Position): highest mean palm_site_abund
    g = df_long.copy()
    g["_palm"] = pd.to_numeric(g["palm_site_abund"], errors="coerce").fillna(0.0)
    rep_seq = (g.groupby(["Accession", "Position", "seq101"], sort=False)["_palm"].mean()
                 .reset_index()
                 .sort_values("_palm", ascending=False)
                 .drop_duplicates(["Accession", "Position"])
              )[["Accession", "Position", "seq101"]]
    rep_seq = rep_seq.rename(columns={"Accession": "accession", "seq101": "prot_seq_101aa_rep"})

    # pivot prot_abund
    prot_piv = g.pivot_table(index=["Accession", "Position"], columns="sample", values="prot_abund",
                             aggfunc="first", fill_value=0.0)
    palm_piv = g.pivot_table(index=["Accession", "Position"], columns="sample", values="palm_site_abund",
                             aggfunc="first", fill_value=0.0)

    # enforce columns
    def std_cols(prefix: str):
        cols = []
        for grp in ["WT", "KO"]:
            for r in range(1, max_rep+1):
                cols.append(f"{prefix}_{grp}{r}")
        return cols

    # rename sample columns WT1 -> WT1 etc, then to PROT_WT1 style
    prot_piv = prot_piv.rename(columns={c: f"PROT_{c}" for c in prot_piv.columns})
    palm_piv = palm_piv.rename(columns={c: f"PALM_{c}" for c in palm_piv.columns})

    dfw = prot_piv.join(palm_piv, how="outer").reset_index().rename(columns={"Accession": "accession"})
    dfw = dfw.merge(rep_seq, on=["accession", "Position"], how="left")
    dfw["prot_seq_101aa_rep"] = dfw["prot_seq_101aa_rep"].fillna("").astype(str).str.upper()

    # add missing cols
    for c in std_cols("PROT") + std_cols("PALM"):
        if c not in dfw.columns:
            dfw[c] = 0.0

    # reorder
    out_cols = ["accession", "prot_seq_101aa_rep"] + std_cols("PROT") + std_cols("PALM")
    dfw = dfw[out_cols].copy()

    # numeric types
    for c in out_cols[2:]:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce").fillna(0.0).astype(np.float32)

    return dfw

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="integration_model root")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--max-rep", type=int, default=5)
    ap.add_argument("--prot-threshold", type=float, default=0.0)
    ap.add_argument("--loose-broadcast", choices=["all", "human", "mouse"], default="all",
                    help="where to include loose negatives (still broadcast within each dataset)")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    folders = sorted([p for p in root.iterdir() if p.is_dir() and p.name.lower() not in ("temp",)])

    all_parts = []
    metas = []

    for folder in folders:
        ds = folder.name
        try:
            df, meta = build_one_dataset(folder, max_rep=args.max_rep,
                                         prot_threshold=args.prot_threshold,
                                         loose_broadcast=args.loose_broadcast)
        except Exception as e:
            raise RuntimeError(f"Failed building dataset {ds}: {e}") from e
        all_parts.append(df)
        metas.append(meta)
        print(f"[OK] {meta['dataset']} rows={len(df)} unique_seq={meta['unique_seq101']} pos_rate={meta['pos_rate_after']:.4f} "
              f"strict_conflict={meta['strict_conflict_count']} pos_missing_site={meta['pos_missing_site_abund_count']}")

    df_all = pd.concat(all_parts, axis=0, ignore_index=True)

    # write splits
    p_all = outdir / "site_sample_long.all.parquet"
    p_h = outdir / "site_sample_long.human.parquet"
    p_m = outdir / "site_sample_long.mouse.parquet"

    df_all.to_parquet(p_all, index=False)
    df_all[df_all["species"] == "human"].to_parquet(p_h, index=False)
    df_all[df_all["species"] == "mouse"].to_parquet(p_m, index=False)

    # unique seq list (from ALL)
    uniq = (df_all.groupby("seq101", sort=False)
                .agg(count=("seq101", "size"),
                     n_pos=("label_bin", "sum"),
                     n_strict_neg=("neg_type", lambda x: int((x == "strict").sum())),
                     n_loose_neg=("neg_type", lambda x: int((x == "loose").sum())))
                .reset_index())
    uniq_path = outdir / "unique_seq101.tsv"
    uniq.to_csv(uniq_path, sep="\t", index=False)

    # legacy wide (ALL) — only meaningful for WT/KO samples present; missing reps will be 0
    wide_all = to_legacy_wide(df_all[df_all["sample"].str.match(r"^(WT|KO)\d+$", na=False)].copy(),
                              max_rep=args.max_rep)
    wide_path = outdir / "site_sample_wide_legacy.all.csv"
    wide_all.to_csv(wide_path, index=False)

    # QC report
    qc = {
        "root": str(root),
        "outdir": str(outdir),
        "max_rep": args.max_rep,
        "prot_threshold": args.prot_threshold,
        "loose_broadcast": args.loose_broadcast,
        "datasets": metas,
        "global": {
            "rows_all": int(len(df_all)),
            "rows_human": int((df_all["species"] == "human").sum()),
            "rows_mouse": int((df_all["species"] == "mouse").sum()),
            "unique_seq101_all": int(df_all["seq101"].nunique()),
            "pos_rate_all": float(df_all["label_bin"].mean()) if len(df_all) else 0.0,
            "strict_conflict_all": int(((df_all["neg_type"] == "strict") & (df_all["palm_site_abund"] > 0)).sum()),
            "pos_missing_site_all": int(((df_all["neg_type"] == "none") & (df_all["palm_site_abund"] <= 0)).sum()),
        }
    }
    qc_path = outdir / "qc_report.json"
    qc_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] {p_all}")
    print(f"[DONE] {p_h}")
    print(f"[DONE] {p_m}")
    print(f"[DONE] {uniq_path}")
    print(f"[DONE] {wide_path}")
    print(f"[DONE] {qc_path}")

if __name__ == "__main__":
    main()
