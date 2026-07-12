# GAN-Palm
Proteomic context-aware generative modeling reveals palmitoylation landscape shaped by ZDHHC20 in tumor cells
This repository hosts the data, code and model for the palmitoylation generative model project.


## code and script repository

```
.
├── code/                      # All scripts
│   ├── build_sitelevel_dataset_v2.py
│   ├── build_proteome_background_embeddings.py
│   ├── conditional_seq_gan_noesm_poslm.py       # Background-conditioned GAN (v1)
│   ├── conditional_seq_gan_noesm_poslm_v2.py    # Background-conditioned GAN (v2; provides model/dataset classes)
│   ├── train_classifier.py                      # Train the frozen classifier C
│   ├── train_g_with_classifier.py               # Classifier-guided generator training  ← final model
│   ├── eval_gan_discriminator.py                # Scoring/evaluation → Figure 3 data
│   ├── plot_figure2_gan_main.py                 # Plot Figure 3
│   ├── make_figure3_noesm.py                    # Figure 3 scoring / positional enrichment
│   ├── analyze_wt_ko_delta_vs_fake.py           # WT/KO delta analysis
│   ├── plot_figure3.py                          # Plot Figure in HeLa
│   ├── plot_figure3_panc1.py                    # Plot Figure in PANC-1
│   └── archive/                                 # Legacy / experimental scripts (not part of the main pipeline)
├── models/                    # Trained model weights
│   ├── classifier_posonly_human/
│   │   ├── C_best.pt          # Frozen classifier C (scores sequences)
│   │   └── C_last.pt
│   ├── gan_frozen_cls_v10/
│   │   ├── checkpoints/
│   │   │   ├── G_best.pt      # Generator G (final model, generates sequences)
│   │   │   ├── G_last.pt
│   │   │   ├── D_best.pt      # Auxiliary discriminator from training (for eval/analysis)
│   │   │   ├── D_last.pt
│   │   │   └── best.json      # Best-checkpoint metadata (epoch/step/metric)
│   │   └── train_loss.jsonl   # Training loss log
│   └── gan_frozen_cls_v10_eval/
│       ├── metrics.json       # Evaluation metrics (source of Figure 2 numbers)
│       └── per_group.csv      # Per-dataset evaluation summary
├── Fig_data/                  # Data to plot the figures
└── README.md
```

---

## Data
As the work has not yet been published, the raw data are currently kept private.
The model operates on **site-level palmitoylation data**: a **101-residue window** centered on each
cysteine (`seq101`, with position 51 fixed to C).
The final model uses a **"frozen-classifier-guided generation"** approach, in three steps:

```
                 ┌─────────────────────────────────────────────┐
 raw CSV/FASTA   │ build_sitelevel_dataset_v2.py               │
 ──────────────▶ │   → site_sample_long.*.parquet              │
                 │ build_proteome_background_embeddings.py     │
                 │   → proteome_bg_embeddings.dataset.npz      │
                 └───────────────┬─────────────────────────────┘
                                 │  (seq101 + label_bin + background vector)
              ┌──────────────────┴───────────────────┐
              ▼                                       ▼
  ① train_classifier.py                    (background-conditioned GAN;
    train binary classifier C on real       provides model/dataset classes)
    pos/neg data                            conditional_seq_gan_noesm_poslm_v2.py
    → models/.../C_best.pt (frozen)
              │
              ▼
  ② train_g_with_classifier.py
    freeze C, train generator G to maximize C's score on generated sequences
    (with LM / center-C / positional-distribution / feature-matching regularizers)
    → models/gan_frozen_cls_v10/checkpoints/G_best.pt   ← final generative model
```

**Step 1 — Train the classifier C** (`train_classifier.py`)
Train a binary classifier on real positive/negative sites to decide whether a 101-residue window
is a palmitoylation site. The classifier and data-handling classes are reused from
`conditional_seq_gan_noesm_poslm_v2.py`. The output `C_best.pt` is then **frozen**.

**Step 2 — Classifier-guided generator training** (`train_g_with_classifier.py`)
Freeze the classifier C and train the generator G: given a sample's background vector, G outputs a
101-aa sequence (central position forced to C). The objective is to **maximize the frozen classifier
C's score on the generated sequences** (`--lambda-cls`), plus masked-LM, central-C constraint,
positional amino-acid distribution, and feature-matching regularizers for training stability.
The output `G_best.pt` (`models/gan_frozen_cls_v10/checkpoints/`) is the final model.

> Earlier attempts (see `code/archive/`) used a standard GAN with an ESM2-3B-augmented or dynamic
> discriminator. Frozen-classifier guidance proved more stable and yielded higher classifier scores
> on the generated sequences, so it is the approach used in the paper.

---

## Script reference

### Main pipeline `code/`
| Script | Input | Output / purpose |
|--------|-------|------------------|
| `build_sitelevel_dataset_v2.py` | raw CSV/FASTA | site-level long table `site_sample_long.*.parquet` |
| `build_proteome_background_embeddings.py` | site-level long table | background vectors `proteome_bg_embeddings.dataset.npz` |
| `conditional_seq_gan_noesm_poslm_v2.py` | long table + background | background-conditioned GAN (v2); **provides the model & dataset classes** for training |
| `conditional_seq_gan_noesm_poslm.py` | long table + background | background-conditioned GAN (v1, predecessor, kept for reference) |
| `train_classifier.py` | long table + background | **train the frozen classifier C** → `C_best.pt` |
| `train_g_with_classifier.py` | long table + background + frozen C | **classifier-guided generator training** → `G_best.pt` (final model) |
| `eval_gan_discriminator.py` | model + long table + background | scoring & evaluation → Figure 2 data (`scores.npz`, etc.) |
| `plot_figure2_gan_main.py` | evaluation result directory | **render Figure 2** |
| `make_figure3_noesm.py` | model + long table + background | Figure 3 scoring / positional enrichment / PWM |
| `analyze_wt_ko_delta_vs_fake.py` | long table + fake sequences | background-corrected WT/KO delta analysis |
| `plot_figure3.py` | fig3 analysis + heatmap dir | **render Figure 3 (HeLa)** |
| `plot_figure3_panc1.py` | extracted fig3 CSVs | **render Figure 3 (PANC-1)** |

### Legacy / experimental `code/archive/` (not part of the main pipeline; kept as a record of method evolution)
| Script | Purpose |
|--------|---------|
| `conditional_seq_gan.py` | Earliest GAN; discriminator augmented with ESM2-3B embeddings |
| `conditional_seq_gan_noesm.py` | ESM-free token-only GAN (predecessor of the main line) |
| `conditional_seq_gan_noesm_bgcontrast.py` | Background-contrast variant of the above |
| `train_discriminator_posneg.py` | Standalone pos/neg discriminator training (CNN / token-only options) |
| `train_unified_10datasets_bgcond_residual.py` | Early unified background-conditioned residual model over 10 datasets |
| `cache_esm2_3b_for_sitelevel_v2.py` | Batch-cache ESM2-3B embeddings (for the ESM-based models) |
| `check_discriminator_effect.py` | Discriminator sanity check (ROC/PR; is it "reading only the background"?) |
| `analyze_fake_bg.py` | Use fake sequences as a controllable background for motif/logo analysis |
| `analyze_generator_conditional_motif_from_bgdb.py` | Build conditional vectors from the background DB by regex and analyze generated motifs |
| `extract_wt_ko_motifs.py` | Extract WT-vs-KO enriched motifs (k-mer / PWM) |
| `interpret_wtko_single_species_noesm.py` | Single-species WT/KO analysis (Δfreq heatmap, JS divergence) |
| `compare_species_delta_signatures.py` | Cross-species delta-signature comparison (cosine / correlation) |
| `plot_kmer_figures.py` | Generate standard result plots from k-mer enrichment tables |

---

## Models `models/`

| File | Description |
|------|-------------|
| `classifier_posonly_human/C_best.pt` | **Frozen classifier C**; scores a 101-residue sequence with a palmitoylation probability (0–1). Used to guide G and for evaluation and Figure 2/3 scoring. |
| `gan_frozen_cls_v10/checkpoints/G_best.pt` | **Final generator G**; takes a background vector and generates a 101-aa sequence. |
| `gan_frozen_cls_v10/checkpoints/D_best.pt` | Auxiliary discriminator from training; usable for evaluation/analysis. |
| `gan_frozen_cls_v10/checkpoints/best.json` | Best-checkpoint metadata (epoch=20, step=3980). |
| `gan_frozen_cls_v10/train_loss.jsonl` | Training loss log. |
| `gan_frozen_cls_v10_eval/metrics.json`, `per_group.csv` | Evaluation metrics behind the Figure 2 numbers. |

> `train_classifier.py` and `train_g_with_classifier.py` both import the model and dataset classes
> from `conditional_seq_gan_noesm_poslm_v2.py` via `importlib`. When loading the weights, keep that
> file consistent with the checkpoint definitions.

## Fig_script
### Figure 3 — Generation-quality evaluation

```
G_best.pt + C (discriminator) + site_sample_long.human.parquet + background vectors
      │
      ├─ eval_gan_discriminator.py     # score real pos/neg and G-generated sequences
      │     → scores.npz, metrics.json, per_group.csv
      ▼
   plot_figure2_gan_main.py            # read the result directory above, render the composite figure
      → Figure 2 (Panels B–E)
```

- **`eval_gan_discriminator.py`**: uses the discriminator/classifier to score and run ROC/PR
  evaluation on four comparisons — pos-vs-neg (classifier value), pos-vs-fake, neg-vs-fake, and
  real-vs-fake. Stratified by `dataset`; background alignment key is `dataset__sample`.
  Outputs `scores.npz`, `metrics.json`, `per_group.csv`.
- **`plot_figure2_gan_main.py`**: reads that `--result-dir` and renders Figure 3:
  - **B** positive vs negative discrimination (ROC-AUC ≈ 0.978)
  - **C** discrimination stratified by dataset (HeLa 0.974 / PANC-1 0.963)
  - **D** real vs generated score distributions (generated mean ≈ 0.996 > real positives 0.933)
  - **E** score summary bars for the three groups (pos / neg / generated) with bootstrap 95% CIs

### Figure 4&5 — Site-specific remodeling under ZDHHC20 perturbation

```
G_best.pt (discriminator) + site_sample_long.human.parquet + background vectors
      │
      ├─ make_figure3_noesm.py                # score all Cys windows; positional enrichment / PWM
      │     → fig4 analysis directory (heatmaps, site tables, etc.)
      ├─ analyze_wt_ko_delta_vs_fake.py       # background-corrected WT vs KO delta analysis (per dataset)
      │     → heatmap / ΔLO3 / summary
      ▼
   plot_figure3.py         (HeLa)   /   plot_figure3_panc1.py   (PANC-1)
      → Figure 4
```

- **`make_figure3_noesm.py`**: scores every Cys-centered window with the discriminator and computes
  background shift, positional enrichment, and PWMs.
- **`analyze_wt_ko_delta_vs_fake.py`**: uses the generated fake sequences as a corrected background
  and computes `ΔLO(i,a) = log2(p_WT/p_fake) − log2(p_KO/p_fake)`, producing a position × amino-acid
  heatmap, cross-center 3-mer ΔLO3, and a summary.
- **`plot_figure3.py`** (HeLa) / **`plot_figure3_panc1.py`** (PANC-1) render Figure 4:
  - **A** bar plot of top sites by delta (WT − KO)
  - **B** position-specific amino-acid enrichment heatmap (red = WT/control-biased, blue = KO/OE-biased)
  - **C** Shannon-entropy profile around the central C (top-biased sites vs background)
  - **D** cross-dataset remodeling-magnitude comparison (HeLa vs PANC-1)

---

## Fig_data
All data to plot the figures.

