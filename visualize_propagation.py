"""
D2NN Propagation Visualisation
========================================
Renders the physical process a D2NN performs on a single input image:
the intensity pattern at each mask plane, the final detector pattern and
classification, and a beam-trajectory ("light cone") side view showing
how the wavefront spreads and interferes as it diffracts through the
stack of phase masks. Reuses the physics primitives from d2nn.py.

If previously exported phase masks are found (see d2nn.py's
export_masks(), default './masks') they are loaded and frozen, and only
the electronic readout layer is fit -- reusing an already-trained optical
stack instead of retraining it from scratch. Otherwise a fresh model is
trained for --epochs epochs.

Usage:
    python visualize_propagation.py
    python visualize_propagation.py --digit 7
    python visualize_propagation.py --masks_dir masks --epochs 3
"""

import os
import glob
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm, TwoSlopeNorm
import warnings
warnings.filterwarnings('ignore')

from d2nn import (
    DEVICE, WAVELENGTH, GRID_SIZE, PAD, PIXEL_SIZE, LAYER_DIST,
    N_LAYERS, N_CLASSES, BATCH_SIZE, LR, D2NN, load_mnist,
    make_propagation_kernel, propagate,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "data", "outputs", "visualisation")
os.makedirs(OUT_DIR, exist_ok=True)

BG, CYAN, AMBER, GREEN, RED, GRID_C, TXT = (
    '#0e0e1a', '#00d4ff', '#ffaa00', '#00ff88', '#ff5555', '#1e1e32', '#d0d0e8')


# ── Model setup ──────────────────────────────────────────────────────────────────
def load_pretrained_masks(model, masks_dir):
    """
    Load phase masks previously exported by d2nn.py's export_masks() and
    freeze them, so the visualisation reuses an already-trained optical
    stack and only needs to fit the linear readout. Returns False (no
    changes made) if the mask count/shape doesn't match this model.
    """
    files = sorted(glob.glob(os.path.join(masks_dir, 'mask_*_phase.npy')),
                    key=lambda f: int(os.path.basename(f).split('_')[1]))
    if len(files) != len(model.layers):
        return False
    for f, layer in zip(files, model.layers):
        phase = np.load(f)
        if phase.shape != tuple(layer.phase.shape):
            return False
    for f, layer in zip(files, model.layers):
        with torch.no_grad():
            layer.phase.copy_(torch.tensor(np.load(f), dtype=torch.float32))
        layer.phase.requires_grad_(False)
    return True


def prepare_model(masks_dir, epochs):
    model = D2NN(GRID_SIZE, PAD, N_LAYERS, N_CLASSES, WAVELENGTH, PIXEL_SIZE, LAYER_DIST).to(DEVICE)
    train_loader, test_loader = load_mnist(BATCH_SIZE)

    if load_pretrained_masks(model, masks_dir):
        print(f"Loaded pretrained masks from '{masks_dir}' -- fitting readout only ({epochs} epoch(s)).")
        trainable = model.readout.parameters()
    else:
        print(f"No compatible masks found in '{masks_dir}' -- training full model from scratch ({epochs} epoch(s)).")
        trainable = model.parameters()

    optimizer = optim.Adam(trainable, lr=LR, weight_decay=1e-5)
    for epoch in range(1, epochs + 1):
        model.train()
        correct, total = 0, 0
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = nn.functional.nll_loss(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            correct += out.argmax(1).eq(target).sum().item()
            total += len(target)
        print(f"  epoch {epoch}/{epochs}  train acc {100*correct/total:.1f}%")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            correct += model(data).argmax(1).eq(target).sum().item()
            total += len(target)
    print(f"  test accuracy: {100*correct/total:.1f}%")
    return model, test_loader


def pick_sample(test_loader, digit=None):
    for data, target in test_loader:
        for i in range(len(target)):
            if digit is None or target[i].item() == digit:
                return data[i:i+1], target[i].item()
    raise ValueError(f"digit {digit} not found in test set")


# ── Forward trace with fine-grained z sampling ──────────────────────────────────
@torch.no_grad()
def trace_forward(model, x, steps_per_hop=12):
    """
    Propagate a single input image through the model step by step,
    recording:
      - the full 2D intensity map at the input, after each mask, and at
        the detector (for the snapshot grid)
      - a 1D central cross-section at every fine z sub-step, for the
        beam-trajectory waterfall plot
    """
    x = x.to(DEVICE)
    pad = model.pad
    field_re = torch.nn.functional.pad(x.squeeze(1), [pad] * 4, mode='constant', value=0)
    field_im = torch.zeros_like(field_re)
    mid_row = model.padded // 2

    def intensity_of(re, im):
        return (re ** 2 + im ** 2).squeeze(0).cpu().numpy()

    plane_snapshots = [('Input', 0.0, intensity_of(field_re, field_im))]
    profile_rows = [intensity_of(field_re, field_im)[mid_row]]
    z_positions = [0.0]

    dz = LAYER_DIST / steps_per_hop
    H_re_fine, H_im_fine = make_propagation_kernel(model.padded, WAVELENGTH, PIXEL_SIZE, dz, DEVICE)

    z = 0.0
    for hop in range(len(model.layers) + 1):
        for _ in range(steps_per_hop):
            field_re, field_im = propagate(field_re, field_im, H_re_fine, H_im_fine)
            z += dz
            profile_rows.append(intensity_of(field_re, field_im)[mid_row])
            z_positions.append(z)
        if hop < len(model.layers):
            field_re, field_im = model.layers[hop](field_re, field_im)
            plane_snapshots.append((f'After mask {hop+1}', z, intensity_of(field_re, field_im)))

    plane_snapshots.append(('Detector (output)', z, intensity_of(field_re, field_im)))

    p = pad
    final_re = field_re[:, p:p+model.grid_size, p:p+model.grid_size]
    final_im = field_im[:, p:p+model.grid_size, p:p+model.grid_size]
    intensity = final_re ** 2 + final_im ** 2
    probs = torch.softmax(model.readout(intensity.flatten(1)), dim=1).squeeze(0).cpu().numpy()

    trajectory = np.stack(profile_rows, axis=0)  # (n_z_steps, padded_width)
    detector_crop = intensity.squeeze(0).cpu().numpy()  # (grid_size, grid_size), same region the readout sees
    readout_weights = model.readout.weight.detach().cpu().numpy().reshape(-1, model.grid_size, model.grid_size)
    return plane_snapshots, trajectory, np.array(z_positions), probs, detector_crop, readout_weights


# ── Plots ────────────────────────────────────────────────────────────────────────
def plot_snapshots(plane_snapshots, probs, true_label, detector_crop, readout_weights, out_path):
    pred = int(np.argmax(probs))
    # Evidence map: how much each detector pixel actually contributed to the
    # winning class score (readout weight for that class times the light
    # intensity landing on that pixel) -- this is *how* the digit is read
    # off the last grid, since the readout is a learned linear combination
    # of pixels rather than 10 fixed non-overlapping zones.
    evidence = readout_weights[pred] * detector_crop
    top_pixel = np.unravel_index(np.argmax(evidence), evidence.shape)

    n_intensity_panels = len(plane_snapshots)
    total_panels = n_intensity_panels + 1  # + evidence map
    cols = min(total_panels, 4)
    rows = int(np.ceil(total_panels / cols)) + 1  # extra row for the classification bar chart
    fig = plt.figure(figsize=(4 * cols, 4 * rows), facecolor='#080810')
    gs = gridspec.GridSpec(rows, cols, hspace=0.55, wspace=0.35)

    # Each physical intensity panel is contrast-stretched against its own
    # peak (gamma < 1 lifts faint diffracted light out of the near-black
    # floor) instead of one shared linear scale, which otherwise crushes
    # every panel past the first mask into dark purple/black.
    for i, (label, z, img) in enumerate(plane_snapshots):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        norm = PowerNorm(gamma=0.4, vmin=0, vmax=max(img.max(), 1e-12))
        im = ax.imshow(img, cmap='inferno', norm=norm)
        ax.set_title(f'{label}\n(z = {z*100:.0f} cm)', color=TXT, fontsize=9)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=6, colors=TXT)

    # Evidence map panel -- diverging colormap since weights can be negative
    r, c = divmod(n_intensity_panels, cols)
    ax_ev = fig.add_subplot(gs[r, c])
    vmax_ev = max(abs(evidence.min()), abs(evidence.max()), 1e-12)
    im_ev = ax_ev.imshow(evidence, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0, vmin=-vmax_ev, vmax=vmax_ev))
    ax_ev.plot(top_pixel[1], top_pixel[0], marker='*', ms=16, color='white',
               markeredgecolor='black', markeredgewidth=0.8)
    ax_ev.set_title(f'Class evidence -> predicted "{pred}"\n(weight x intensity, peak pixel marked)',
                     color=TXT, fontsize=9)
    ax_ev.axis('off')
    cb_ev = plt.colorbar(im_ev, ax=ax_ev, fraction=0.046, pad=0.04)
    cb_ev.ax.tick_params(labelsize=6, colors=TXT)

    ax_bar = fig.add_subplot(gs[rows - 1, :])
    ax_bar.set_facecolor(BG)
    colors = [GREEN if i == true_label else CYAN for i in range(len(probs))]
    if pred != true_label:
        colors[pred] = RED
    ax_bar.bar(range(len(probs)), probs, color=colors)
    ax_bar.set_xticks(range(len(probs)))
    ax_bar.set_xlabel('Digit class', color=TXT)
    ax_bar.set_ylabel('Probability', color=TXT)
    ax_bar.set_title(f'Classification output  --  true label {true_label}, predicted {pred}',
                      color=TXT, fontsize=10)
    ax_bar.tick_params(colors=TXT)
    ax_bar.spines[:].set_color(GRID_C)

    fig.suptitle('D2NN Optical Process: Input -> Diffraction -> Classification',
                 color='white', fontsize=14, fontweight='bold', y=0.995)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    print(f"Snapshots -> {out_path}")


def plot_trajectory(trajectory, z_positions, mask_planes_z, pixel_size, out_path):
    """
    trajectory: (n_z, padded_width) intensity along the central row at
    each fine z-step -- a side-view "light trajectory" of the beam as it
    diffracts and interferes through the mask stack.
    """
    n_z, width = trajectory.shape
    x_mm = (np.arange(width) - width / 2) * pixel_size * 1e3
    z_mm = z_positions * 1e3

    # Gamma < 1 lifts the faint spreading/interference fringes out of the
    # near-black floor -- a plain linear (or even log1p) scale leaves most
    # of the frame looking like flat dark purple since the input peak
    # dwarfs everything downstream of the first mask.
    norm = PowerNorm(gamma=0.35, vmin=0, vmax=trajectory.max())

    fig, ax = plt.subplots(figsize=(8, 10), facecolor='#080810')
    ax.set_facecolor(BG)
    im = ax.pcolormesh(x_mm, z_mm, trajectory, cmap='inferno', norm=norm, shading='auto')
    for i, mz in enumerate(mask_planes_z):
        ax.axhline(mz * 1e3, color=CYAN, lw=1, ls='--', alpha=0.8)
        ax.text(x_mm.max() * 0.95, mz * 1e3, f'mask {i+1}', color=CYAN, fontsize=8,
                va='bottom', ha='right')
    ax.invert_yaxis()  # source at top, detector at bottom -- light travels downward
    ax.set_xlabel('Lateral position x (mm)', color=TXT)
    ax.set_ylabel('Propagation distance z (mm)', color=TXT)
    ax.set_title('Light Trajectory Through the D2NN Stack\n(central cross-section intensity, log scale)',
                 color=TXT, fontsize=11)
    ax.tick_params(colors=TXT)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Intensity (contrast-stretched)', color=TXT, fontsize=8)
    cb.ax.tick_params(colors=TXT, labelsize=7)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close(fig)
    print(f"Trajectory -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="D2NN propagation & light-trajectory visualisation")
    parser.add_argument('--masks_dir', default=os.path.join(SCRIPT_DIR, 'masks'))
    parser.add_argument('--epochs', type=int, default=3,
                         help='epochs to fit the readout (or the full model if no pretrained masks are found)')
    parser.add_argument('--digit', type=int, default=None,
                         help='which MNIST digit to visualise (default: first test sample)')
    parser.add_argument('--steps_per_hop', type=int, default=12,
                         help='z-resolution of the beam trajectory plot between mask planes')
    args = parser.parse_args()

    print(f"== D2NN Propagation Visualisation ==  device={DEVICE}")
    model, test_loader = prepare_model(args.masks_dir, args.epochs)

    x, true_label = pick_sample(test_loader, args.digit)
    print(f"Visualising digit '{true_label}'")

    plane_snapshots, trajectory, z_positions, probs, detector_crop, readout_weights = trace_forward(
        model, x, args.steps_per_hop)
    mask_planes_z = [z for label, z, _ in plane_snapshots if label.startswith('After mask')]

    plot_snapshots(plane_snapshots, probs, true_label, detector_crop, readout_weights,
                    os.path.join(OUT_DIR, 'process_snapshots.png'))
    plot_trajectory(trajectory, z_positions, mask_planes_z, PIXEL_SIZE, os.path.join(OUT_DIR, 'beam_trajectory.png'))

    print(f"\nAll visualisations saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
