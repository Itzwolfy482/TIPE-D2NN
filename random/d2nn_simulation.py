"""
D2NN - Diffractive Deep Neural Network Simulation
Based on: Lin et al., Science 2018 + noise tolerance analysis

Architecture:
  Input plane → [free space propagation] → Phase mask 1 
             → [free space propagation] → Phase mask 2 
             → ... 
             → [free space propagation] → Output plane

Each free-space propagation is implemented as a Fourier transform
(Fraunhofer/Angular Spectrum approximation).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


# ─── Configuration ────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAVELENGTH  = 532e-9          # 532 nm green laser (metres)
PIXEL_SIZE  = 0.3e-3          # 0.3 mm pixel pitch (macro scale, 3D printable)
GRID_SIZE   = 28              # Match MNIST 28×28
LAYER_DIST  = 0.05            # 5 cm between layers (metres)
N_LAYERS    = 20          # Number of diffractive layers
N_CLASSES   = 10              # MNIST digits 0-9
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 0.002
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]  # Phase noise σ (radians)

print(f"Running on: {DEVICE}")
print(f"Wavelength: {WAVELENGTH*1e9:.0f} nm | Pixel size: {PIXEL_SIZE*1e3:.1f} mm | Grid: {GRID_SIZE}×{GRID_SIZE}")


# ─── Angular Spectrum Propagation ─────────────────────────────────────────────
def angular_spectrum_propagation(field, distance, wavelength, pixel_size):
    """
    Propagates a complex optical field using the Angular Spectrum Method.
    This is the rigorous Fourier optics propagation operator.
    
    H(fx, fy) = exp(j * 2π/λ * z * sqrt(1 - (λfx)² - (λfy)²))
    
    Args:
        field:      Complex tensor [B, H, W]
        distance:   Propagation distance z (metres)
        wavelength: Light wavelength λ (metres)
        pixel_size: Spatial sampling Δx (metres)
    Returns:
        Propagated complex field [B, H, W]
    """
    B, H, W = field.shape
    k = 2 * np.pi / wavelength

    # Spatial frequency coordinates
    fx = torch.fft.fftfreq(W, d=pixel_size).to(field.device)
    fy = torch.fft.fftfreq(H, d=pixel_size).to(field.device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    FX, FY = FX.T, FY.T  # Shape [H, W]

    # Transfer function — evanescent waves are filtered (sqrt stays real)
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    arg = torch.clamp(arg, min=0)
    H_transfer = torch.exp(1j * k * distance * torch.sqrt(arg))  # [H, W]

    # Apply in Fourier domain
    F = torch.fft.fft2(field)                     # [B, H, W]
    F_propagated = F * H_transfer.unsqueeze(0)    # broadcast over batch
    field_out = torch.fft.ifft2(F_propagated)

    return field_out


# ─── Diffractive Layer ─────────────────────────────────────────────────────────
class DiffractiveLayer(nn.Module):
    """
    A single diffractive phase mask.
    Each neuron modulates the phase of the incoming field:
        t(x,y) = exp(j * φ(x,y))
    where φ is a learnable parameter in [-π, π].
    """
    def __init__(self, size, noise_std=0.0):
        super().__init__()
        # Learnable phase values, initialised randomly
        self.phase = nn.Parameter(
            torch.empty(size, size).uniform_(-np.pi, np.pi)
        )
        self.noise_std = noise_std  # Fabrication noise level

    def forward(self, field):
        """
        field: complex tensor [B, H, W]
        """
        phase = self.phase

        # Inject fabrication noise during forward pass (inference mode)
        if self.noise_std > 0:
            noise = torch.randn_like(phase) * self.noise_std
            phase = phase + noise

        # Phase-only modulation: multiply field by exp(jφ)
        modulation = torch.exp(1j * phase)        # [H, W]
        return field * modulation.unsqueeze(0)    # [B, H, W]


# ─── Full D2NN Model ───────────────────────────────────────────────────────────
class D2NN(nn.Module):
    """
    Full Diffractive Deep Neural Network.
    
    Forward pass:
      1. Encode input image as amplitude of optical field
      2. Propagate through alternating: [free space] → [phase mask] layers
      3. Detect intensity at output plane in class-specific detector regions
      4. Softmax over detector region energies → class probabilities
    """
    def __init__(self, grid_size, n_layers, n_classes,
                 wavelength, pixel_size, layer_dist, noise_std=0.0):
        super().__init__()
        self.grid_size  = grid_size
        self.n_classes  = n_classes
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.layer_dist = layer_dist

        # Stack of learnable diffractive layers
        self.layers = nn.ModuleList([
            DiffractiveLayer(grid_size, noise_std=noise_std)
            for _ in range(n_layers)
        ])

        # Define non-overlapping detector regions for each class
        self.detector_regions = self._build_detector_regions(grid_size, n_classes)

    def _build_detector_regions(self, size, n_classes):
        """Divide output plane into n_classes rectangular detector regions."""
        regions = []
        rows = int(np.ceil(np.sqrt(n_classes)))
        cols = int(np.ceil(n_classes / rows))
        h = size // rows
        w = size // cols
        for i in range(n_classes):
            r = i // cols
            c = i % cols
            regions.append((r*h, min((r+1)*h, size), c*w, min((c+1)*w, size)))
        return regions

    def forward(self, x):
        """
        x: real tensor [B, 1, H, W] — normalised MNIST images
        returns: log-probabilities [B, n_classes]
        """
        B = x.shape[0]

        # Encode amplitude: field = amplitude * exp(0j)
        amplitude = x.squeeze(1).float()  # [B, H, W]
        field = amplitude.to(torch.complex64)

        # Propagate through layers
        for layer in self.layers:
            # Free-space propagation
            field = angular_spectrum_propagation(
                field, self.layer_dist, self.wavelength, self.pixel_size
            )
            # Phase modulation
            field = layer(field)

        # Final propagation to output/detector plane
        field = angular_spectrum_propagation(
            field, self.layer_dist, self.wavelength, self.pixel_size
        )

        # Detect: sum intensity in each class region
        intensity = field.abs()**2  # [B, H, W]
        outputs = []
        for (r0, r1, c0, c1) in self.detector_regions:
            energy = intensity[:, r0:r1, c0:c1].sum(dim=(1, 2))  # [B]
            outputs.append(energy)

        logits = torch.stack(outputs, dim=1)  # [B, n_classes]
        return torch.log_softmax(logits, dim=1)


# ─── Data Loading ──────────────────────────────────────────────────────────────
def load_mnist(batch_size=64, n_train=2000, n_test=500):
    """Load dataset from local numpy files."""
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    x_train = train_dataset.data.unsqueeze(1).float() / 255
    y_train = train_dataset.targets

    x_test = test_dataset.data.unsqueeze(1).float() / 255
    y_test = test_dataset.targets

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    test_set  = torch.utils.data.TensorDataset(x_test,  y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ─── Training ─────────────────────────────────────────────────────────────────
def train(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += len(target)
        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch} [{batch_idx*len(data)}/{len(loader.dataset)}] "
                  f"Loss: {loss.item():.4f} | Acc: {100*correct/total:.1f}%", end='\r')
    print()
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return 100 * correct / total


# ─── Noise Analysis ───────────────────────────────────────────────────────────
def noise_analysis(trained_model, test_loader, noise_levels):
    """
    Evaluate accuracy of a trained model under varying levels of phase noise.
    Noise is injected at inference time into each diffractive layer.
    """
    print("\n── Fabrication Noise Analysis ──────────────────────────────")
    accuracies = []
    for noise_std in noise_levels:
        # Inject noise level into all layers
        for layer in trained_model.layers:
            layer.noise_std = noise_std
        acc = evaluate(trained_model, test_loader)
        accuracies.append(acc)
        print(f"  σ_noise = {noise_std:.2f} rad → Accuracy: {acc:.1f}%")
    # Reset noise
    for layer in trained_model.layers:
        layer.noise_std = 0.0
    return accuracies


# ─── Visualisation ────────────────────────────────────────────────────────────
def plot_results(train_accs, test_accs, noise_levels, noise_accs, model):
    fig = plt.figure(figsize=(18, 12), facecolor='#0a0a0f')
    fig.suptitle('D2NN Simulation — Fabrication Tolerance Analysis',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color   = '#e0e0e0'
    accent_cyan  = '#00d4ff'
    accent_amber = '#ffaa00'
    accent_green = '#00ff88'
    grid_color   = '#2a2a3a'

    ax_style = dict(facecolor='#12121f', labelcolor=text_color,
                    titlecolor=text_color)

    # 1 — Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(ax_style['facecolor'])
    epochs_x = range(1, len(train_accs)+1)
    ax1.plot(epochs_x, train_accs, color=accent_cyan,  lw=2, marker='o', ms=4, label='Train')
    ax1.plot(epochs_x, test_accs,  color=accent_amber, lw=2, marker='s', ms=4, label='Test')
    ax1.set_title('Training Accuracy', color=text_color, fontsize=11)
    ax1.set_xlabel('Epoch', color=text_color)
    ax1.set_ylabel('Accuracy (%)', color=text_color)
    ax1.tick_params(colors=text_color)
    ax1.legend(facecolor='#1a1a2e', labelcolor=text_color, edgecolor=grid_color)
    ax1.grid(True, color=grid_color, alpha=0.5)
    ax1.spines[:].set_color(grid_color)

    # 2 — Noise tolerance curve (main result)
    ax2 = fig.add_subplot(gs[0, 1:])
    ax2.set_facecolor(ax_style['facecolor'])
    baseline = noise_accs[0]
    ax2.axhline(y=baseline,      color=accent_green, lw=1.5, ls='--', alpha=0.7, label=f'Clean baseline ({baseline:.1f}%)')
    ax2.axhline(y=baseline*0.9,  color='#ff6b6b',    lw=1,   ls=':',  alpha=0.6, label='10% degradation threshold')
    ax2.axhline(y=10,            color='#888',        lw=1,   ls=':',  alpha=0.4, label='Random chance (10%)')
    ax2.plot(noise_levels, noise_accs, color=accent_cyan, lw=2.5, marker='o', ms=6, zorder=5)
    ax2.fill_between(noise_levels, noise_accs, 10, alpha=0.15, color=accent_cyan)

    # Mark cliff point
    cliff_idx = next((i for i, a in enumerate(noise_accs) if a < baseline * 0.9), None)
    if cliff_idx:
        ax2.axvline(x=noise_levels[cliff_idx], color='#ff6b6b', lw=1.5, ls='--', alpha=0.8)
        ax2.annotate(f'Tolerance cliff\nσ ≈ {noise_levels[cliff_idx]:.1f} rad',
                     xy=(noise_levels[cliff_idx], noise_accs[cliff_idx]),
                     xytext=(noise_levels[cliff_idx]+0.2, noise_accs[cliff_idx]+8),
                     color='#ff6b6b', fontsize=9,
                     arrowprops=dict(arrowstyle='->', color='#ff6b6b'))

    ax2.set_title('Accuracy vs. Fabrication Phase Noise — Key Research Result', color=text_color, fontsize=11)
    ax2.set_xlabel('Phase noise σ (radians)', color=text_color)
    ax2.set_ylabel('Classification accuracy (%)', color=text_color)
    ax2.tick_params(colors=text_color)
    ax2.legend(facecolor='#1a1a2e', labelcolor=text_color, edgecolor=grid_color, fontsize=9)
    ax2.grid(True, color=grid_color, alpha=0.5)
    ax2.spines[:].set_color(grid_color)
    ax2.set_ylim(0, 105)

    # 3 — Learned phase masks (first 3 layers)
    for i in range(min(3, N_LAYERS)):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor(ax_style['facecolor'])
        phase_data = model.layers[i].phase.detach().cpu().numpy()
        im = ax.imshow(phase_data, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                       interpolation='nearest')
        ax.set_title(f'Learned Phase Mask — Layer {i+1}', color=text_color, fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Phase (rad)', color=text_color, fontsize=8)
        cbar.ax.yaxis.set_tick_params(color=text_color)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=text_color)

    plt.savefig('/mnt/user-data/outputs/d2nn_results.png', dpi=150,
                bbox_inches='tight', facecolor='#0a0a0f')
    print("\nFigure saved → d2nn_results.png")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n══ D2NN Simulation ══════════════════════════════════════════")
    print(f"  Layers: {N_LAYERS} | Grid: {GRID_SIZE}×{GRID_SIZE} | Epochs: {EPOCHS}")
    print(f"  Layer spacing: {LAYER_DIST*100:.0f} cm | λ = {WAVELENGTH*1e9:.0f} nm\n")

    # Data
    train_loader, test_loader = load_mnist(BATCH_SIZE, n_train=2000, n_test=500)

    # Model
    model = D2NN(
        grid_size  = GRID_SIZE,
        n_layers   = N_LAYERS,
        n_classes  = N_CLASSES,
        wavelength = WAVELENGTH,
        pixel_size = PIXEL_SIZE,
        layer_dist = LAYER_DIST,
        noise_std  = 0.0
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_params:,} (phase values)\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Train
    train_accs, test_accs = [], []
    print("── Training ─────────────────────────────────────────────────")
    for epoch in range(1, EPOCHS+1):
        _, tr_acc = train(model, train_loader, optimizer, epoch)
        te_acc    = evaluate(model, test_loader)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        scheduler.step()
        print(f"  → Epoch {epoch}/{EPOCHS} | Train: {tr_acc:.1f}% | Test: {te_acc:.1f}%")

    # Noise analysis
    noise_accs = noise_analysis(model, test_loader, NOISE_LEVELS)

    # Plot
    plot_results(train_accs, test_accs, NOISE_LEVELS, noise_accs, model)

    # Save model
    torch.save(model.state_dict(), '/mnt/user-data/outputs/d2nn_trained.pth')
    print("Model saved → d2nn_trained.pth")

    # Summary
    print("\n══ Summary ══════════════════════════════════════════════════")
    print(f"  Final test accuracy (no noise): {test_accs[-1]:.1f}%")
    tol_idx = next((i for i, a in enumerate(noise_accs) if a < test_accs[-1]*0.9), len(NOISE_LEVELS)-1)
    print(f"  Tolerance cliff (>10% degradation): σ ≈ {NOISE_LEVELS[tol_idx]:.2f} rad")
    print(f"  In physical terms: {NOISE_LEVELS[tol_idx]/(2*np.pi)*WAVELENGTH*1e9:.1f} nm height error on printed mask")





if __name__ == "__main__":
    main()
# il y a je pense un interet a bruteforce; en redemarrant plusieurs fois l'algo et ainsi obtenir le meilleur resultat