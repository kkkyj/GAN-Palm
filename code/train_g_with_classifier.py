#!/usr/bin/env python3
"""Train G to maximize a frozen classifier C's score on generated sequences."""
import argparse, json, sys, importlib.util, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Import from existing GAN script
spec = importlib.util.spec_from_file_location(
    'gan', str(Path(__file__).parent / 'conditional_seq_gan_noesm_poslm_v2.py'))
gan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan)

SiteSeqDataset = gan.SiteSeqDataset
Generator = gan.Generator
ConditionalSequenceDisc = gan.ConditionalSequenceDisc
load_bg_db = gan.load_bg_db
collate_with_bg = gan.collate_with_bg
seq_mlm_loss = gan.seq_mlm_loss
pos_dist_loss = gan.pos_dist_loss
compute_pos_freq = gan.compute_pos_freq
center_penalty = gan.center_penalty
AA2IDX = gan.AA2IDX
CENTER_POS = gan.CENTER_POS

def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-table', required=True)
    ap.add_argument('--bg-npz', required=True)
    ap.add_argument('--cls-ckpt', required=True, help='Path to frozen classifier C checkpoint')
    ap.add_argument('--g-init-ckpt', default=None, help='Optional G init checkpoint')
    ap.add_argument('--seq-col', default='seq101')
    ap.add_argument('--label-col', default='label_bin')
    ap.add_argument('--dataset-col', default='dataset')
    ap.add_argument('--sample-col', default='sample')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--warmup-epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr-g', type=float, default=2e-4)
    ap.add_argument('--lr-g-warmup', type=float, default=1e-4)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--lambda-cls', type=float, default=2.0, help='Weight for -C(fake) loss')
    ap.add_argument('--lambda-lm', type=float, default=0.3)
    ap.add_argument('--lambda-center', type=float, default=0.5)
    ap.add_argument('--lambda-psd', type=float, default=0.5)
    ap.add_argument('--lambda-feat-match', type=float, default=0.5)
    ap.add_argument('--lambda-contrast', type=float, default=1.0)
    ap.add_argument('--contrast-margin', type=float, default=0.1)
    ap.add_argument('--mask-ratio', type=float, default=0.30)
    ap.add_argument('--g-grad-clip', type=float, default=1.0)
    ap.add_argument('--debug-every', type=int, default=200)
    ap.add_argument('--skip-nan-steps', action='store_true')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    (outdir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    bg_map, bg_mean = load_bg_db(args.bg_npz)
    bg_dim = int(bg_mean.shape[0])
    bg_keys = list(bg_map.keys())

    ds = SiteSeqDataset(args.site_table, seq_col=args.seq_col, label_col=args.label_col,
                        dataset_col=args.dataset_col, sample_col=args.sample_col)
    labels = ds.labels
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    print(f'[DATA] pos={pos_idx.size} neg={neg_idx.size}')

    class _Subset(Dataset):
        def __init__(self, base, indices):
            self.base, self.indices = base, indices.astype(np.int64)
        def __len__(self): return int(self.indices.size)
        def __getitem__(self, i): return self.base[int(self.indices[int(i)])]

    def _collate(batch):
        return collate_with_bg(batch, bg_map=bg_map, bg_fallback=bg_mean)

    pos_loader = DataLoader(_Subset(ds, pos_idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate)
    neg_loader = DataLoader(_Subset(ds, neg_idx), batch_size=args.batch_size,
                            shuffle=True, num_workers=0, drop_last=True, collate_fn=_collate)
    pos_iter = cycle_loader(pos_loader)
    neg_iter = cycle_loader(neg_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Frozen classifier C
    C = ConditionalSequenceDisc(bg_dim=bg_dim).to(device)
    ckpt = torch.load(args.cls_ckpt, map_location=device)
    C.load_state_dict(ckpt, strict=True)
    C.eval()
    for p in C.parameters():
        p.requires_grad_(False)
    print(f'[OK] loaded frozen C from {args.cls_ckpt}')

    # Generator
    G = Generator(bg_dim=bg_dim).to(device)
    if args.g_init_ckpt:
        g_ckpt = torch.load(args.g_init_ckpt, map_location=device)
        G.load_state_dict(g_ckpt, strict=True)
        print(f'[OK] loaded G init from {args.g_init_ckpt}')

    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

    # PSD freq
    pos_freq_np = compute_pos_freq(ds.tokens[pos_idx])
    pos_freq_tensor = torch.from_numpy(pos_freq_np).to(device)
    print(f'[INFO] pos_freq shape={pos_freq_np.shape}')

    def _set_lr(opt, lr):
        for pg in opt.param_groups:
            pg['lr'] = float(lr)

    warmup_lr = args.lr_g_warmup if args.lr_g_warmup else args.lr_g * 0.2
    logs = []
    step = 0
    best_metric = float('inf')

    for epoch in range(1, args.epochs + 1):
        for batch_i, pb in enumerate(pos_loader):
            step += 1
            pos_tok = pb['tokens'].to(device)
            pos_bg = pb['bg'].to(device)

            # --- Warmup: LM only ---
            if epoch <= args.warmup_epochs:
                _set_lr(optG, warmup_lr)
                G.train()
                optG.zero_grad()
                lm = seq_mlm_loss(G, pos_tok, pos_bg, mask_ratio=args.mask_ratio)
                lm.backward()
                nn.utils.clip_grad_norm_(G.parameters(), args.g_grad_clip)
                optG.step()
                if args.debug_every > 0 and step % args.debug_every == 0:
                    print(f'[warmup step {step}] lm={lm.item():.4f}')
                continue

            _set_lr(optG, args.lr_g)

            # --- G step with frozen C ---
            G.train()
            optG.zero_grad()

            # STE hard tokens for C scoring
            fake_st = G.sample_gumbel_soft(pos_bg, tau=args.tau, hard=True)
            logit_fake = C(fake_st, pos_bg, is_soft=True)
            cls_loss = -logit_fake.mean()  # maximize C(fake)

            # LM on positive sequences
            lm = seq_mlm_loss(G, pos_tok, pos_bg, mask_ratio=args.mask_ratio)

            # PSD (soft tokens)
            fake_soft = G.sample_gumbel_soft(pos_bg, tau=args.tau, hard=False)
            psd = pos_dist_loss(fake_soft, pos_freq_tensor)

            # Center penalty
            center = center_penalty(fake_st, weight=1.0)

            # Feature matching: fake features vs real pos features
            feat_pos = C.forward_features(pos_tok, pos_bg, is_soft=False).detach()
            feat_fake = C.forward_features(fake_st, pos_bg, is_soft=True)
            feat_match = F.mse_loss(feat_fake.mean(0), feat_pos.mean(0))

            # Contrast: push C(fake) > C(neg) + margin
            if args.lambda_contrast > 0:
                nb = next(neg_iter)
                neg_tok = nb['tokens'].to(device)
                neg_bg = nb['bg'].to(device)
                with torch.no_grad():
                    logit_neg = C(neg_tok, neg_bg, is_soft=False)
                contrast = F.relu(logit_neg.mean().detach() - logit_fake.mean() + args.contrast_margin)
            else:
                contrast = torch.zeros(1, device=device)

            loss_G = (args.lambda_cls * cls_loss +
                      args.lambda_lm * lm +
                      args.lambda_center * center +
                      args.lambda_feat_match * feat_match +
                      args.lambda_psd * psd +
                      args.lambda_contrast * contrast)

            if torch.isnan(loss_G) or torch.isinf(loss_G):
                if args.skip_nan_steps:
                    optG.zero_grad()
                    continue
                loss_G = torch.nan_to_num(loss_G, nan=0.0)

            loss_G.backward()
            nn.utils.clip_grad_norm_(G.parameters(), args.g_grad_clip)
            optG.step()

            if args.debug_every > 0 and step % args.debug_every == 0:
                # Quick C(fake) sigmoid score for monitoring
                with torch.no_grad():
                    c_score = torch.sigmoid(logit_fake).mean().item()
                print(f'[step {step}] loss_G={loss_G.item():.4f} '
                      f'cls={cls_loss.item():.4f} C(fake)={c_score:.4f} '
                      f'lm={lm.item():.4f} psd={psd.item():.4f} '
                      f'feat={feat_match.item():.4f} contrast={contrast.item():.4f}')

            logs.append({'step': step, 'epoch': epoch,
                         'loss_G': float(loss_G.item()),
                         'cls': float(cls_loss.item()),
                         'lm': float(lm.item()),
                         'psd': float(psd.item()),
                         'feat_match': float(feat_match.item()),
                         'contrast': float(contrast.item())})

        print(f'[Epoch {epoch}] done.')

        # Save
        torch.save(G.state_dict(), outdir / 'checkpoints' / 'G_last.pt')
        # Also copy C to outdir for eval convenience
        torch.save(C.state_dict(), outdir / 'checkpoints' / 'D_best.pt')
        torch.save(C.state_dict(), outdir / 'checkpoints' / 'D_last.pt')

        # Best by mean cls loss (lower = G fools C more)
        epoch_logs = [r for r in logs if r['epoch'] == epoch and r.get('cls') is not None]
        if epoch_logs:
            mean_cls = np.mean([r['cls'] for r in epoch_logs])
            metric = mean_cls
            print(f'[EPOCH {epoch}] mean_cls={mean_cls:.6f}')
            if metric < best_metric:
                best_metric = metric
                torch.save(G.state_dict(), outdir / 'checkpoints' / 'G_best.pt')
                with open(outdir / 'checkpoints' / 'best.json', 'w') as f:
                    json.dump({'epoch': epoch, 'step': step, 'metric': float(metric)}, f, indent=2)
                print(f'  [BEST] metric={metric:.6f}')

        with open(outdir / 'train_loss.jsonl', 'w') as f:
            for r in logs:
                f.write(json.dumps(r) + '\n')

    print(f'[OK] training finished. outdir={outdir}')

if __name__ == '__main__':
    main()
