"""
D2NN - Diffractive Deep Neural Network Simulation  v4
Changes from v3:
  - Phase masks only cover the active 28x28 region, embedded into padded grid
  - Fixes the "all centered" visualisation issue
  - Cleaner physically: only the illuminated area has learnable phase
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WAVELENGTH   = 532e-9
PIXEL_SIZE   = 0.3e-3
GRID_SIZE    = 28
PAD          = 14
LAYER_DIST   = 0.10
N_LAYERS     = 20
N_CLASSES    = 10
BATCH_SIZE   = 128
EPOCHS       = 5    #temporary changes pb better at about 10-20
LR           = 2e-3
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0] 
# se renseigner sur les differentes manieres de gerer la config pour
#   ameliorer la precision du reseau de neuronnes --> check avec l'experience
# il faut faire en sorte que chaque layer soit utile 
#danger d'un determinisme trop important qui permette de faire des gros contrastes de phase  et ainsi d'avoir la precision necessaire avec les 10000 images test;;

PADDED = GRID_SIZE + 2 * PAD   # 56
print("""░▒▓███████▓▒░░▒▓███████▓▒░░▒▓███████▓▒░░▒▓███████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░""")





print(f"Device  : {DEVICE}")
print(f"λ = {WAVELENGTH*1e9:.0f} nm | pixel = {PIXEL_SIZE*1e3:.1f} mm | "
      f"active = {GRID_SIZE}² | propagation grid = {PADDED}²")


# ── Propagation kernel ─────────────────────────────────────────────────────────
def make_propagation_kernel(grid_size, wavelength, pixel_size, distance, device):
    k  = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(grid_size, d=pixel_size)
    fy = np.fft.fftfreq(grid_size, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    arg   = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
    arg   = np.clip(arg, 0, None)
    phase = k * distance * np.sqrt(arg)
    H_re  = torch.tensor(np.cos(phase), dtype=torch.float32, device=device).unsqueeze(0)
    H_im  = torch.tensor(np.sin(phase), dtype=torch.float32, device=device).unsqueeze(0)
    return H_re, H_im
# la partie qui cree la source de lumière "parfaite"

def propagate(field_re, field_im, H_re, H_im):
    F      = torch.fft.fft2(torch.complex(field_re, field_im))
    out_re = F.real * H_re - F.imag * H_im
    out_im = F.real * H_im + F.imag * H_re
    out    = torch.fft.ifft2(torch.complex(out_re, out_im))
    return out.real, out.imag
# propagation à travers le milieu 


# ── Diffractive Layer ──────────────────────────────────────────────────────────
class DiffractiveLayer(nn.Module):
    """
    Learnable phase mask covering only the active illuminated region.
    Embedded into the full padded grid for propagation.
    Outside the active area the mask is transparent (phase = 0).
    """
    def __init__(self, padded_size, active_size):
        super().__init__()
        self.padded = padded_size
        self.active = active_size
        self.p      = (padded_size - active_size) // 2
        # Only the active pixels are learnable
        self.phase  = nn.Parameter(
            torch.empty(active_size, active_size).uniform_(-np.pi, np.pi)
        )

    def forward(self, field_re, field_im, noise_std=0.0):
        phase = self.phase
        if noise_std > 0:
            phase = phase + torch.randn_like(phase) * noise_std

        # Embed into full padded mask (transparent outside active area)
        full_phase = torch.zeros(self.padded, self.padded, device=phase.device)
        p = self.p
        full_phase[p:p+self.active, p:p+self.active] = phase

        t_re   = torch.cos(full_phase)
        t_im   = torch.sin(full_phase)
        out_re = field_re * t_re - field_im * t_im
        out_im = field_re * t_im + field_im * t_re
        return out_re, out_im
# cree les elements pour pouvoir avoir l'effet diffractif necessaire 


# ── Full D2NN ──────────────────────────────────────────────────────────────────
class D2NN(nn.Module):
    def __init__(self, grid_size, pad, n_layers, n_classes,
                 wavelength, pixel_size, layer_dist):
        super().__init__()
        self.grid_size = grid_size
        self.pad       = pad
        self.padded    = grid_size + 2 * pad
        self.noise_std = 0.0

        self.layers = nn.ModuleList([
            DiffractiveLayer(self.padded, grid_size) for _ in range(n_layers)
        ])

        H_re, H_im = make_propagation_kernel(
            self.padded, wavelength, pixel_size, layer_dist, 'cpu'
        )
        self.register_buffer('H_re', H_re)
        self.register_buffer('H_im', H_im)

        # Linear readout: intensity over active region → class scores
        self.readout = nn.Linear(grid_size * grid_size, n_classes)

    def forward(self, x):
        # Pad input
        field_re = torch.nn.functional.pad(
            x.squeeze(1), [self.pad]*4, mode='constant', value=0
        )
        field_im = torch.zeros_like(field_re)

        # Optical path
        for layer in self.layers:
            field_re, field_im = propagate(field_re, field_im, self.H_re, self.H_im)
            field_re, field_im = layer(field_re, field_im, noise_std=self.noise_std)

        field_re, field_im = propagate(field_re, field_im, self.H_re, self.H_im)

        # Crop to active region
        p = self.pad
        field_re = field_re[:, p:p+self.grid_size, p:p+self.grid_size]
        field_im = field_im[:, p:p+self.grid_size, p:p+self.grid_size]

        intensity = field_re**2 + field_im**2
        return torch.log_softmax(self.readout(intensity.flatten(1)), dim=1)

# totalite de comment le D2NN fonctionne pour nous 


# ── Data ───────────────────────────────────────────────────────────────────────
def load_mnist(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    train_set    = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_set     = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader
#un peu inutile pour nous pourrais etre mis dans un autre fichier pour eviter de le bloat


# ── Train / Eval ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, epoch, total_epochs):
    model.train()
    correct, total, running_loss = 0, 0, 0
    for i, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        out  = model(data)
        loss = nn.functional.nll_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        correct += out.argmax(1).eq(target).sum().item()
        total   += len(target)
        if i % 50 == 0:
            print(f"  Epoch {epoch}/{total_epochs} [{total}/{len(loader.dataset)}] "
                  f"Loss: {running_loss/(i+1):.3f} | Acc: {100*correct/total:.1f}%", end='\r')
    print()
    return running_loss / len(loader), 100 * correct / total
# partie qui permet l'entrainement du reseau

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            correct += model(data).argmax(1).eq(target).sum().item()
            total   += len(target)
    return 100 * correct / total

#formule un peu basique pour connaitre a quel point le truc fut terrible par rapport a un autre set de donnees que celui entraine


# ── Noise sweep ────────────────────────────────────────────────────────────────
def noise_sweep(model, loader, levels):
    print("\n── Fabrication Noise Sweep ──────────────────────────────────")
    results = []
    for sigma in levels:
        model.noise_std = sigma
        acc = evaluate(model, loader)
        results.append(acc)
        filled = int(acc / 2)
        bar    = '█' * filled + '░' * (50 - filled)
        print(f"  σ = {sigma:.2f} rad │{bar}│ {acc:.1f}%")
    model.noise_std = 0.0
    return results

#create de bruit pour les differents masques (mouhahah tres mechant)


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot(train_accs, test_accs, noise_levels, noise_accs, model):
    rows_needed = 2 + int(np.ceil(len(model.layers) / 3))
    fig = plt.figure(figsize=(18, 6 * rows_needed), facecolor='#080810')
    gs  = gridspec.GridSpec(rows_needed, 3, hspace=0.45, wspace=0.35)
    bg, cyan, amber, green, red, grid_c, txt = (
        '#0e0e1a','#00d4ff','#ffaa00','#00ff88','#ff5555','#1e1e32','#d0d0e8')

    def style(ax, title):
        ax.set_facecolor(bg)
        ax.set_title(title, color=txt, fontsize=10, pad=8)
        ax.tick_params(colors=txt, labelsize=8)
        ax.spines[:].set_color(grid_c)
        ax.grid(True, color=grid_c, alpha=0.6, lw=0.5)
        ax.xaxis.label.set_color(txt)
        ax.yaxis.label.set_color(txt)

    # Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    ep  = range(1, len(train_accs)+1)
    ax1.plot(ep, train_accs, color=cyan,  lw=2, marker='o', ms=3, label='Train')
    ax1.plot(ep, test_accs,  color=amber, lw=2, marker='s', ms=3, label='Test')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)')
    ax1.legend(facecolor='#14142a', labelcolor=txt, edgecolor=grid_c, fontsize=8)
    style(ax1, 'Training Accuracy')

    # Noise tolerance
    ax2      = fig.add_subplot(gs[0, 1:])
    baseline = noise_accs[0]
    ax2.axhline(baseline,       color=green, lw=1.5, ls='--', alpha=0.7,
                label=f'Clean baseline ({baseline:.1f}%)')
    ax2.axhline(baseline * 0.9, color=red,   lw=1.0, ls=':',  alpha=0.6,
                label='−10% degradation threshold')
    ax2.axhline(10,             color='#666', lw=0.8, ls=':',  alpha=0.5,
                label='Random chance (10%)')
    ax2.plot(noise_levels, noise_accs, color=cyan, lw=2.5, marker='o', ms=6, zorder=5)
    ax2.fill_between(noise_levels, noise_accs, 10, alpha=0.12, color=cyan)
    cliff = next((i for i, a in enumerate(noise_accs) if a < baseline * 0.9), None)
    if cliff:
        ax2.axvline(noise_levels[cliff], color=red, lw=1.2, ls='--', alpha=0.7)
        ax2.annotate(
            f'Tolerance cliff\nσ ≈ {noise_levels[cliff]:.1f} rad',
            xy=(noise_levels[cliff], noise_accs[cliff]),
            xytext=(noise_levels[cliff] + 0.15, noise_accs[cliff] + 6),
            color=red, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=red, lw=1)
        )
    ax2.set_xlabel('Phase noise σ (radians)')
    ax2.set_ylabel('Classification accuracy (%)')
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor='#14142a', labelcolor=txt, edgecolor=grid_c, fontsize=8)
    style(ax2, 'Accuracy vs Fabrication Phase Noise  —  Key Research Result')

    # Learned phase masks — now showing only the active 28x28 region
    for i in range(len(model.layers)):
        row = 2 + i // 3
        col = i % 3
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor(bg)
        # Extract only the active learned region (not the padded zeros)
        ph = model.layers[i].phase.detach().cpu().numpy()
        im = ax.imshow(ph, cmap='hsv', vmin=-np.pi, vmax=np.pi)
        ax.set_title(f'Learned Phase Mask — Layer {i+1}\n(active {GRID_SIZE}×{GRID_SIZE} region)',
                     color=txt, fontsize=9)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Phase (rad)', color=txt, fontsize=7)
        cb.ax.yaxis.set_tick_params(color=txt, labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=txt)

    fig.suptitle('D2NN Simulation v4 — Fabrication Tolerance Analysis',
                 color='white', fontsize=14, fontweight='bold', y=0.99)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))  # use __file__ in scripts
    out_dir = os.path.join(script_dir, "data", "outputs", "images")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "d2nn_results.png")    

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#080810')
    print("Plot saved → d2nn_results.png")
    plt.close()

#partie pour le graphe, pas tres interessante d'un point de vue technique 


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"\n══ D2NN v4  |  {N_LAYERS} layers  |  {EPOCHS} epochs  |  {DEVICE} ══\n")

    train_loader, test_loader = load_mnist(BATCH_SIZE)

    model = D2NN(
        grid_size  = GRID_SIZE,
        pad        = PAD,
        n_layers   = N_LAYERS,
        n_classes  = N_CLASSES,
        wavelength = WAVELENGTH,
        pixel_size = PIXEL_SIZE,
        layer_dist = LAYER_DIST,
    ).to(DEVICE)

    n_optical    = sum(p.numel() for p in model.layers.parameters())
    n_electronic = sum(p.numel() for p in model.readout.parameters())
    print(f"  Optical parameet un chasseur de primes dans la ville de Coruscant. À un moment, les pilotes effectuentters    (phase masks) : {n_optical:,}")
    print(f"  Electronic parameters (readout)     : {n_electronic:,}")
    print(f"  Total                               : {n_optical + n_electronic:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)

    train_accs, test_accs = [], []
    print("── Training ─────────────────────────────────────────────────")
    # voila comment fonctionne le truc (les étapes parcourues)


    for epoch in range(1, EPOCHS + 1):
        _, tr = train_epoch(model, train_loader, optimizer, epoch, EPOCHS) #on entraine
        te    = evaluate(model, test_loader) #puis on teste sur un certains nombre d'autres images
        train_accs.append(tr)
        test_accs.append(te)
        scheduler.step()
        print(f"  ✓ Epoch {epoch:2d}/{EPOCHS}  train {tr:.1f}%  test {te:.1f}%")

    #du blabla pour la recherche scientifique

    noise_accs = noise_sweep(model, test_loader, NOISE_LEVELS)
    plot(train_accs, test_accs, NOISE_LEVELS, noise_accs, model)
    #torch.save(model.state_dict(), 'd2nn_trained.pth')

    print("\n══ Summary ══════════════════════════════════════════════════")
    print(f"  Peak test accuracy : {max(test_accs):.1f}%")
    baseline    = noise_accs[0]
    cliff       = next((i for i, a in enumerate(noise_accs) if a < baseline * 0.9),
                       len(NOISE_LEVELS) - 1)
    sigma_cliff = NOISE_LEVELS[cliff]
    height_nm   = sigma_cliff / (2 * np.pi) * WAVELENGTH * 1e9
    print(f"  Tolerance cliff    : σ ≈ {sigma_cliff:.2f} rad")
    print(f"  Physical meaning   : {height_nm:.1f} nm surface height error")
    print(f"  FDM printer roughness  ~50,000 nm  →  far outside tolerance")
    print(f"  SLA resin roughness    ~1,000  nm  →  borderline")
    print(f"  Optical polishing      ~10     nm  →  within tolerance")


if __name__ == "__main__":
    main()