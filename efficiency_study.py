"""
D2NN Efficiency Study
========================================
Research comparisons for optimising D2NN design choices, reusing the
physics primitives (propagation kernel, DiffractiveLayer) from d2nn.py.

Four independent studies, each trains several D2NN variants on MNIST and
plots test accuracy against a cost metric:

  1. layers     -- number of diffractive masks (optical parameters / size)
  2. resolution -- pixels per mask side, fixed physical aperture
  3. spacing    -- how masks are distributed along the optical axis:
                   equal repartition vs Gaussian-concentrated (denser
                   near the centre of the optical path)
  4. thinness   -- how the phase (-> printed relief height) is distributed
                   within each mask: equal repartition across [-pi,pi]
                   vs Gaussian concentration near 0 (thinner prints) with
                   an L1 penalty pulling training toward thin relief

Usage:
    python efficiency_study.py                     # run all studies
    python efficiency_study.py --study layers       # run one study
    python efficiency_study.py --quick              # small subset, fast smoke test
"""

import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from d2nn import (
    DEVICE, WAVELENGTH, MASK_WIDTH, LAYER_DIST, N_LAYERS, GRID_SIZE, PAD,
    N_CLASSES, BATCH_SIZE, LR,
    make_propagation_kernel, propagate, DiffractiveLayer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "data", "outputs", "efficiency_study")
os.makedirs(OUT_DIR, exist_ok=True)

BG, CYAN, AMBER, GREEN, RED, GRID_C, TXT = (
    '#0e0e1a', '#00d4ff', '#ffaa00', '#00ff88', '#ff5555', '#1e1e32', '#d0d0e8')


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TXT, fontsize=10, pad=8)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.spines[:].set_color(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.4, lw=0.5)
    ax.xaxis.label.set_color(TXT)
    ax.yaxis.label.set_color(TXT)


# ── Data (grid_size is configurable, unlike d2nn.load_mnist) ───────────────────
def load_mnist_at(grid_size, batch_size, subset=None):
    transform = transforms.Compose([
        transforms.Resize((grid_size, grid_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    train_set = torchvision.datasets.MNIST(
        os.path.join(SCRIPT_DIR, 'data'), train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        os.path.join(SCRIPT_DIR, 'data'), train=False, download=True, transform=transform)
    if subset is not None:
        train_set = Subset(train_set, range(min(subset, len(train_set))))
        test_set = Subset(test_set, range(min(subset // 5, len(test_set))))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ── Layer spacing along the optical axis ────────────────────────────────────────
def make_layer_distances(n_layers, total_length, mode='equal'):
    """
    Returns n_layers + 1 inter-plane distances (metres), summing to
    total_length, i.e. where the n_layers mask planes sit between the
    source and the detector.

      'equal'    - masks equally spaced (matches d2nn.py's fixed LAYER_DIST)
      'gaussian' - masks concentrated around the centre of the optical path,
                   sparser near source/detector
    """
    if mode == 'equal':
        edges = np.linspace(0, total_length, n_layers + 2)
    elif mode == 'gaussian':
        # Evenly spaced probability quantiles pushed through the inverse
        # normal CDF land denser near the median (where the pdf is largest)
        # and sparser near the tails -- i.e. masks pack toward the centre
        # of the optical path. A margin keeps the first/last mask off the
        # source/detector planes (zero-length hops would be unphysical).
        margin = 0.15 * total_length
        inner_length = total_length - 2 * margin
        qs = np.linspace(0.05, 0.95, n_layers)
        z = np.sqrt(2) * torch.erfinv(torch.tensor(2 * qs - 1, dtype=torch.float64)).numpy()
        z = (z - z.min()) / (z.max() - z.min())
        positions = margin + z * inner_length
        edges = np.concatenate([[0.0], positions, [total_length]])
    else:
        raise ValueError(f"unknown spacing mode: {mode}")
    return [float(d) for d in np.diff(edges)]


# ── Flexible D2NN: variable spacing + variable phase concentration ─────────────
class FlexibleD2NN(nn.Module):
    """
    Same optics as d2nn.D2NN (reuses DiffractiveLayer / propagate / kernel
    generation) but supports a non-uniform list of inter-plane distances
    and a choice of phase-mask initialisation ('uniform' vs 'gaussian'),
    needed for the spacing and thinness studies.
    """
    def __init__(self, grid_size, pad, n_layers, n_classes, wavelength,
                 pixel_size, layer_dists, phase_mode='uniform', phase_sigma=0.6):
        super().__init__()
        assert len(layer_dists) == n_layers + 1
        self.grid_size = grid_size
        self.pad = pad
        self.padded = grid_size + 2 * pad
        self.noise_std = 0.0
        self.layer_dists = layer_dists

        self.layers = nn.ModuleList([
            DiffractiveLayer(self.padded, grid_size) for _ in range(n_layers)
        ])
        with torch.no_grad():
            for layer in self.layers:
                if phase_mode == 'uniform':
                    layer.phase.uniform_(-np.pi, np.pi)
                elif phase_mode == 'gaussian':
                    layer.phase.normal_(0.0, phase_sigma)
                else:
                    raise ValueError(f"unknown phase_mode: {phase_mode}")

        for d in sorted(set(layer_dists)):
            H_re, H_im = make_propagation_kernel(self.padded, wavelength, pixel_size, d, 'cpu')
            self.register_buffer(f'H_re_{self._key(d)}', H_re)
            self.register_buffer(f'H_im_{self._key(d)}', H_im)

        self.readout = nn.Linear(grid_size * grid_size, n_classes)

    @staticmethod
    def _key(d):
        return f"{d:.8f}".replace('.', 'p').replace('-', 'm')

    def kernel(self, d):
        key = self._key(d)
        return getattr(self, f'H_re_{key}'), getattr(self, f'H_im_{key}')

    def forward(self, x):
        field_re = torch.nn.functional.pad(x.squeeze(1), [self.pad] * 4, mode='constant', value=0)
        field_im = torch.zeros_like(field_re)
        for i, layer in enumerate(self.layers):
            H_re, H_im = self.kernel(self.layer_dists[i])
            field_re, field_im = propagate(field_re, field_im, H_re, H_im)
            field_re, field_im = layer(field_re, field_im, noise_std=self.noise_std)
        H_re, H_im = self.kernel(self.layer_dists[-1])
        field_re, field_im = propagate(field_re, field_im, H_re, H_im)
        p = self.pad
        field_re = field_re[:, p:p + self.grid_size, p:p + self.grid_size]
        field_im = field_im[:, p:p + self.grid_size, p:p + self.grid_size]
        intensity = field_re ** 2 + field_im ** 2
        return torch.log_softmax(self.readout(intensity.flatten(1)), dim=1)

    def phase_l1(self):
        """Mean |phase| across all masks -- the regularisation term used to
        actively push training toward thin (near-zero height) relief."""
        return sum(layer.phase.abs().mean() for layer in self.layers) / len(self.layers)

    def mean_relief_height_mm(self, n_material=1.5):
        """Mean printable relief height (mm) across all masks -- the
        physical 'thinness' cost metric (less material, faster/cheaper print)."""
        h_max_m = WAVELENGTH / (n_material - 1)
        vals = []
        for layer in self.layers:
            wrapped = torch.remainder(layer.phase.detach(), 2 * np.pi)
            vals.append(wrapped / (2 * np.pi) * h_max_m * 1e3)
        return torch.cat([v.flatten() for v in vals]).mean().item()

    def n_optical_params(self):
        return sum(p.numel() for p in self.layers.parameters())


# ── Generic train / eval loop (supports optional phase regularisation) ─────────
def train_and_eval(model, train_loader, test_loader, epochs, lr=LR, reg_weight=0.0, tag=""):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    t0 = time.time()
    test_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = nn.functional.nll_loss(out, target)
            if reg_weight > 0:
                loss = loss + reg_weight * model.phase_l1()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            correct += out.argmax(1).eq(target).sum().item()
            total += len(target)
        scheduler.step()

        model.eval()
        c, t = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                c += model(data).argmax(1).eq(target).sum().item()
                t += len(target)
        test_acc = 100 * c / t
        print(f"    [{tag}] epoch {epoch}/{epochs}  train {100*correct/total:.1f}%  "
              f"test {test_acc:.1f}%  ({time.time()-t0:.0f}s elapsed)")

    return test_acc, time.time() - t0


def save_json(name, payload):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  Results -> {path}")


# ── Study 1: number of masks ────────────────────────────────────────────────────
def study_layers(layer_counts, epochs, subset):
    print("\n== Study: number of masks ==")
    train_loader, test_loader = load_mnist_at(GRID_SIZE, BATCH_SIZE, subset)
    results = []
    for n in layer_counts:
        distances = [LAYER_DIST] * (n + 1)
        model = FlexibleD2NN(GRID_SIZE, PAD, n, N_CLASSES, WAVELENGTH,
                              MASK_WIDTH / GRID_SIZE, distances, phase_mode='uniform')
        acc, secs = train_and_eval(model, train_loader, test_loader, epochs, tag=f"n_layers={n}")
        results.append({
            'n_layers': n, 'test_acc': acc, 'train_time_s': secs,
            'n_optical_params': model.n_optical_params(),
            'system_length_m': sum(distances),
        })
    save_json('study_layers.json', results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#080810')
    ns = [r['n_layers'] for r in results]
    accs = [r['test_acc'] for r in results]
    params = [r['n_optical_params'] for r in results]
    axes[0].plot(ns, accs, color=CYAN, marker='o', lw=2)
    axes[0].set_xlabel('Number of masks'); axes[0].set_ylabel('Test accuracy (%)')
    style(axes[0], 'Accuracy vs Number of Masks')
    axes[1].plot(params, accs, color=AMBER, marker='s', lw=2)
    for r in results:
        axes[1].annotate(f"n={r['n_layers']}", (r['n_optical_params'], r['test_acc']),
                          color=TXT, fontsize=7, xytext=(4, 4), textcoords='offset points')
    axes[1].set_xlabel('Optical parameters (pixels)'); axes[1].set_ylabel('Test accuracy (%)')
    style(axes[1], 'Accuracy vs Optical Parameter Budget')
    fig.suptitle('Efficiency Study: Number of Diffractive Masks', color='white', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUT_DIR, 'study_layers.png'), dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    return results


# ── Study 2: mask resolution ─────────────────────────────────────────────────────
def study_resolution(resolutions, epochs, subset, n_layers=N_LAYERS):
    print("\n== Study: mask resolution ==")
    results = []
    for grid_size in resolutions:
        pad = grid_size // 2
        pixel_size = MASK_WIDTH / grid_size
        train_loader, test_loader = load_mnist_at(grid_size, BATCH_SIZE, subset)
        distances = [LAYER_DIST] * (n_layers + 1)
        model = FlexibleD2NN(grid_size, pad, n_layers, N_CLASSES, WAVELENGTH,
                              pixel_size, distances, phase_mode='uniform')
        acc, secs = train_and_eval(model, train_loader, test_loader, epochs, tag=f"grid={grid_size}")
        results.append({
            'grid_size': grid_size, 'test_acc': acc, 'train_time_s': secs,
            'pixel_size_mm': pixel_size * 1e3,
            'n_optical_params': model.n_optical_params(),
        })
    save_json('study_resolution.json', results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#080810')
    gs_ = [r['grid_size'] for r in results]
    accs = [r['test_acc'] for r in results]
    axes[0].plot(gs_, accs, color=CYAN, marker='o', lw=2)
    axes[0].set_xlabel('Mask resolution (pixels/side)'); axes[0].set_ylabel('Test accuracy (%)')
    style(axes[0], 'Accuracy vs Mask Resolution (fixed 15x15cm aperture)')
    px = [r['pixel_size_mm'] for r in results]
    axes[1].plot(px, accs, color=AMBER, marker='s', lw=2)
    axes[1].invert_xaxis()
    axes[1].set_xlabel('Pixel pitch (mm) -- finer to the right'); axes[1].set_ylabel('Test accuracy (%)')
    style(axes[1], 'Accuracy vs Fabrication Pixel Pitch')
    fig.suptitle('Efficiency Study: Mask Resolution', color='white', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUT_DIR, 'study_resolution.png'), dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    return results


# ── Study 3: layer spacing distribution ──────────────────────────────────────────
def study_spacing(modes, epochs, subset, n_layers=N_LAYERS):
    print("\n== Study: layer spacing distribution ==")
    train_loader, test_loader = load_mnist_at(GRID_SIZE, BATCH_SIZE, subset)
    total_length = (n_layers + 1) * LAYER_DIST
    results = []
    for mode in modes:
        distances = make_layer_distances(n_layers, total_length, mode)
        model = FlexibleD2NN(GRID_SIZE, PAD, n_layers, N_CLASSES, WAVELENGTH,
                              MASK_WIDTH / GRID_SIZE, distances, phase_mode='uniform')
        acc, secs = train_and_eval(model, train_loader, test_loader, epochs, tag=f"spacing={mode}")
        results.append({'mode': mode, 'test_acc': acc, 'train_time_s': secs, 'distances_m': distances})
    save_json('study_spacing.json', results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#080810')
    for i, r in enumerate(results):
        edges = np.concatenate([[0], np.cumsum(r['distances_m'])])
        planes = edges[1:-1]
        axes[0].scatter(planes, [i] * len(planes), color=[CYAN, AMBER, GREEN, RED][i % 4], s=60, zorder=5)
        axes[0].axhline(i, color=GRID_C, lw=0.5, alpha=0.5)
    axes[0].set_yticks(range(len(results)))
    axes[0].set_yticklabels([r['mode'] for r in results])
    axes[0].set_xlabel('Position along optical axis (m)')
    style(axes[0], 'Mask Plane Layout by Spacing Mode')

    modes_ = [r['mode'] for r in results]
    accs = [r['test_acc'] for r in results]
    axes[1].bar(modes_, accs, color=[CYAN, AMBER, GREEN, RED][:len(modes_)])
    axes[1].set_ylabel('Test accuracy (%)')
    style(axes[1], 'Accuracy by Spacing Mode (same total length)')
    fig.suptitle('Efficiency Study: Layer Spacing Distribution', color='white', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUT_DIR, 'study_spacing.png'), dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    return results


# ── Study 4: phase / relief-thinness distribution ───────────────────────────────
def study_thinness(configs, epochs, subset, n_layers=N_LAYERS):
    """
    configs: list of dicts {'label', 'phase_mode', 'phase_sigma', 'reg_weight'}
    Compares equal repartition of phase (uniform over [-pi,pi], no
    regularisation) against gaussian concentration near zero phase
    (small-sigma init, optional L1 penalty during training) -- i.e. masks
    that are mostly thin with only a few taller printed features.
    """
    print("\n== Study: mask thinness (phase distribution) ==")
    train_loader, test_loader = load_mnist_at(GRID_SIZE, BATCH_SIZE, subset)
    distances = [LAYER_DIST] * (n_layers + 1)
    results = []
    for cfg in configs:
        model = FlexibleD2NN(GRID_SIZE, PAD, n_layers, N_CLASSES, WAVELENGTH,
                              MASK_WIDTH / GRID_SIZE, distances,
                              phase_mode=cfg['phase_mode'], phase_sigma=cfg.get('phase_sigma', 0.6))
        acc, secs = train_and_eval(model, train_loader, test_loader, epochs,
                                    reg_weight=cfg.get('reg_weight', 0.0), tag=cfg['label'])
        results.append({
            'label': cfg['label'], 'test_acc': acc, 'train_time_s': secs,
            'mean_relief_height_mm': model.mean_relief_height_mm(),
        })
    save_json('study_thinness.json', results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#080810')
    labels = [r['label'] for r in results]
    accs = [r['test_acc'] for r in results]
    heights = [r['mean_relief_height_mm'] for r in results]
    colors = [CYAN, AMBER, GREEN, RED][:len(labels)]
    axes[0].bar(labels, accs, color=colors)
    axes[0].set_ylabel('Test accuracy (%)')
    style(axes[0], 'Accuracy by Phase Distribution Mode')
    axes[1].scatter(heights, accs, color=colors, s=80, zorder=5)
    for r, c in zip(results, colors):
        axes[1].annotate(r['label'], (r['mean_relief_height_mm'], r['test_acc']),
                          color=c, fontsize=8, xytext=(6, 4), textcoords='offset points')
    axes[1].set_xlabel('Mean relief height (mm) -- thinner = cheaper/faster to print')
    axes[1].set_ylabel('Test accuracy (%)')
    style(axes[1], 'Accuracy vs Fabrication Thinness')
    fig.suptitle('Efficiency Study: Mask Thinness (Phase Concentration)', color='white', fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUT_DIR, 'study_thinness.png'), dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D2NN efficiency comparisons")
    parser.add_argument('--study', choices=['layers', 'resolution', 'spacing', 'thinness', 'all'],
                         default='all')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--quick', action='store_true',
                         help='use a small data subset for a fast smoke test')
    args = parser.parse_args()

    subset = 2000 if args.quick else None
    epochs = 2 if args.quick else args.epochs

    print(f"== D2NN Efficiency Study == epochs={epochs}  quick={args.quick}  device={DEVICE}")

    if args.study in ('layers', 'all'):
        study_layers([1, 2, 3, 5, 8], epochs, subset)
    if args.study in ('resolution', 'all'):
        study_resolution([14, 20, 28, 40], epochs, subset)
    if args.study in ('spacing', 'all'):
        study_spacing(['equal', 'gaussian'], epochs, subset)
    if args.study in ('thinness', 'all'):
        study_thinness([
            {'label': 'equal (uniform phase)', 'phase_mode': 'uniform', 'reg_weight': 0.0},
            {'label': 'gaussian init', 'phase_mode': 'gaussian', 'phase_sigma': 0.6, 'reg_weight': 0.0},
            {'label': 'gaussian + thin penalty', 'phase_mode': 'gaussian', 'phase_sigma': 0.6, 'reg_weight': 0.05},
        ], epochs, subset)

    print(f"\nAll results saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
