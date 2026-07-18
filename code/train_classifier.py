#!/usr/bin/env python3
"""Train a pos/neg classifier C on real data (no GAN)."""
import argparse, json, sys, importlib.util, numpy as np, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# Import model & dataset from existing GAN script
spec = importlib.util.spec_from_file_location(
    'gan', str(Path(__file__).parent / 'conditional_seq_gan_noesm_poslm_v2.py'))
gan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gan)

SiteSeqDataset = gan.SiteSeqDataset
ConditionalSequenceDisc = gan.ConditionalSequenceDisc
load_bg_db = gan.load_bg_db
collate_with_bg = gan.collate_with_bg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-table', required=True)
    ap.add_argument('--bg-npz', required=True)
    ap.add_argument('--seq-col', default='seq101')
    ap.add_argument('--label-col', default='label_bin')
    ap.add_argument('--dataset-col', default='dataset')
    ap.add_argument('--sample-col', default='sample')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--pos-frac', type=float, default=0.5)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bg_map, bg_mean = load_bg_db(args.bg_npz)
    bg_dim = int(bg_mean.shape[0])

    ds = SiteSeqDataset(args.site_table, seq_col=args.seq_col, label_col=args.label_col,
                        dataset_col=args.dataset_col, sample_col=args.sample_col)

    labels = ds.labels
    pos_idx = np.where(labels > 0.5)[0]
    neg_idx = np.where(labels <= 0.5)[0]
    print(f'[DATA] pos={pos_idx.size}  neg={neg_idx.size}')

    class BalancedSampler(torch.utils.data.Sampler):
        def __init__(self, pos_idx, neg_idx, batch_size, pos_frac, epoch_len):
            self.pos_idx = pos_idx
            self.neg_idx = neg_idx
            self.batch_size = batch_size
            self.n_pos = max(1, int(batch_size * pos_frac))
            self.n_neg = batch_size - self.n_pos
            self.epoch_len = epoch_len

        def __iter__(self):
            for _ in range(self.epoch_len):
                pi = np.random.choice(self.pos_idx, self.n_pos, replace=True)
                ni = np.random.choice(self.neg_idx, self.n_neg, replace=True)
                yield from np.concatenate([pi, ni]).tolist()

        def __len__(self):
            return self.epoch_len * self.batch_size

    steps_per_epoch = max(pos_idx.size, 2000) // args.batch_size
    sampler = BalancedSampler(pos_idx, neg_idx, args.batch_size, args.pos_frac, steps_per_epoch)

    def _collate(batch):
        return collate_with_bg(batch, bg_map=bg_map, bg_fallback=bg_mean)

    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=0, drop_last=True, collate_fn=_collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C = ConditionalSequenceDisc(bg_dim=bg_dim).to(device)
    opt = torch.optim.Adam(C.parameters(), lr=args.lr, betas=(0.9, 0.999))
    bce = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        C.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in loader:
            tok = batch['tokens'].to(device)
            bg = batch['bg'].to(device)
            lbl = batch['label'].to(device).float()

            logit = C(tok, bg, is_soft=False).squeeze(-1)
            loss = bce(logit, lbl)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(C.parameters(), 1.0)
            opt.step()

            total_loss += loss.item() * tok.size(0)
            pred = (logit > 0).float()
            correct += (pred == lbl).sum().item()
            total += tok.size(0)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        print(f'[Epoch {epoch}] loss={avg_loss:.4f}  acc={acc:.4f}  ({correct}/{total})')

        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save(C.state_dict(), outdir / 'C_best.pt')
            print(f'  [BEST] acc={acc:.4f} saved')

        # Save last
        torch.save(C.state_dict(), outdir / 'C_last.pt')

    # Quick eval: score pos/neg/report
    C.eval()
    with torch.no_grad():
        pos_scores, neg_scores = [], []
        for batch in loader:
            tok = batch['tokens'].to(device)
            bg = batch['bg'].to(device)
            lbl = batch['label'].cpu().numpy()
            s = torch.sigmoid(C(tok, bg, is_soft=False)).squeeze(-1).cpu().numpy()
            pos_scores.extend(s[lbl > 0.5].tolist())
            neg_scores.extend(s[lbl <= 0.5].tolist())
            if len(pos_scores) > 10000:
                break
    print(f'[EVAL] mean C(pos)={np.mean(pos_scores):.6f}  C(neg)={np.mean(neg_scores):.6f}')
    print(f'[DONE] outdir={outdir}')

if __name__ == '__main__':
    main()
