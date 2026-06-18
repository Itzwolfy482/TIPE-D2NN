"""
D2NN Mask → 3D Printable STL Generator
========================================
Takes trained phase mask .npy files and produces:
  - One STL file per mask (printable heightmap slab)
  - Orientation arrows + layer number embossed on the base
  - A preview PNG showing all masks with their height maps
  - A summary CSV with print statistics

Usage:
    python mask_to_3d.py

    Edit the CONFIG section below to match your simulation parameters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import struct

# ── CONFIG — must match your simulation settings ───────────────────────────────
MASKS_DIR    = './masks'          # folder containing mask_N_phase.npy files
OUTPUT_DIR   = './masks_3d'      # where STL files and previews go
WAVELENGTH   = 2.5e-3            # metres — must match simulation WAVELENGTH
N_MATERIAL   = 1.5               # refractive index of PLA
GRID_SIZE    = 28               # pixels per side — must match simulation GRID_SIZE
MASK_WIDTH   = 0.15             # metres — physical mask width — must match simulation MASK_WIDTH
PIXEL_SIZE   = MASK_WIDTH / GRID_SIZE   # metres — physical size of each mask pixel
BASE_HEIGHT  = 2.0               # mm — flat base below the phase relief
ARROW_HEIGHT = 0.6               # mm — height of orientation features above base
MIN_FEATURE  = 0.15              # mm — minimum printable feature (FDM limit)

# Derived
H_MAX = (WAVELENGTH / (N_MATERIAL - 1)) * 1e3   # max relief height in mm

print(f"── D2NN Mask → STL Generator ────────────────────────────────")
print(f"   Wavelength     : {WAVELENGTH*1e3:.1f} mm  ({(3e8/WAVELENGTH)/1e9:.0f} GHz)")
print(f"   Pixel pitch    : {PIXEL_SIZE*1e3:.0f} mm")
print(f"   Max relief h   : {H_MAX:.2f} mm  (full 2π phase shift)")
print(f"   Base height    : {BASE_HEIGHT:.1f} mm")
print(f"   Min feature    : {MIN_FEATURE:.2f} mm  (FDM limit)")
print()


# ── STL writer (binary) ───────────────────────────────────────────────────────
def write_stl(filename, triangles):
    """
    Write a binary STL file.
    triangles: list of (v0, v1, v2) tuples, each vertex is (x, y, z) in mm.
    """
    with open(filename, 'wb') as f:
        f.write(b'\x00' * 80)                          # header
        f.write(struct.pack('<I', len(triangles)))      # triangle count
        for v0, v1, v2 in triangles:
            # Compute face normal
            a  = np.array(v1) - np.array(v0)
            b  = np.array(v2) - np.array(v0)
            n  = np.cross(a, b)
            nn = np.linalg.norm(n)
            n  = n / nn if nn > 0 else n
            f.write(struct.pack('<fff', *n))            # normal
            f.write(struct.pack('<fff', *v0))
            f.write(struct.pack('<fff', *v1))
            f.write(struct.pack('<fff', *v2))
            f.write(struct.pack('<H', 0))               # attribute


# ── Heightmap → STL ───────────────────────────────────────────────────────────
def corner_marker(pixel_mm, base_h, h_max):
    """
    A tall triangular-prism spike at the (0,0) corner that always rises ABOVE the
    relief, so the (0,0) pixel is unmistakable in any viewer and after printing.
    Returned as a closed (watertight) solid; the slicer unions it with the slab.
    """
    M  = pixel_mm * 2.0                 # footprint size (2 pixels)
    z0 = 0.0
    z1 = base_h + h_max + 1.0           # 1 mm above the tallest possible relief
    P0, P1, P2 = (0.0, 0.0), (M, 0.0), (0.0, M)   # right angle at the (0,0) corner
    tris = []
    # Bottom and top triangular caps
    tris += [((P0[0],P0[1],z0),(P2[0],P2[1],z0),(P1[0],P1[1],z0))]
    tris += [((P0[0],P0[1],z1),(P1[0],P1[1],z1),(P2[0],P2[1],z1))]
    # Three vertical side walls
    for (ax,ay),(bx,by) in [(P0,P1),(P1,P2),(P2,P0)]:
        tris += [
            ((ax,ay,z0),(bx,by,z0),(bx,by,z1)),
            ((ax,ay,z0),(bx,by,z1),(ax,ay,z1)),
        ]
    return tris


def pixel_box(x0, y0, x1, y1, zt):
    """
    A closed axis-aligned box from z=0 to z=zt with a perfectly flat top.
    Each pixel is one box, so the union is guaranteed watertight (no holes)
    regardless of how its neighbours step up or down.
    """
    return [
        # bottom (z=0)
        ((x0,y0,0),(x0,y1,0),(x1,y1,0)),  ((x0,y0,0),(x1,y1,0),(x1,y0,0)),
        # flat top (z=zt)
        ((x0,y0,zt),(x1,y0,zt),(x1,y1,zt)),  ((x0,y0,zt),(x1,y1,zt),(x0,y1,zt)),
        # front (y=y0)
        ((x0,y0,0),(x1,y0,0),(x1,y0,zt)),  ((x0,y0,0),(x1,y0,zt),(x0,y0,zt)),
        # back (y=y1)
        ((x1,y1,0),(x0,y1,0),(x0,y1,zt)),  ((x1,y1,0),(x0,y1,zt),(x1,y1,zt)),
        # left (x=x0)
        ((x0,y1,0),(x0,y0,0),(x0,y0,zt)),  ((x0,y1,0),(x0,y0,zt),(x0,y1,zt)),
        # right (x=x1)
        ((x1,y0,0),(x1,y1,0),(x1,y1,zt)),  ((x1,y0,0),(x1,y1,zt),(x1,y0,zt)),
    ]


def heightmap_to_stl(heights_mm, base_h, pixel_mm, layer_idx):
    """
    Convert a 2D height map to a solid, watertight STL mesh.

    Each pixel is a FLAT-TOPPED closed box at a single height (one phase
    value = one height). Emitting a full box per pixel guarantees the union
    is watertight (no holes) no matter how neighbours step up or down — the
    physically correct, fabrication-ready representation, not a smoothed
    surface. Adjacent boxes share coincident interior faces, which the slicer
    unions cleanly.

    Structure:
      - One closed box per pixel (flat top at the pixel's height)
      - Tall corner spike at (0,0) for orientation
      - Layer-number ridges on the front edge
    """
    rows, cols = heights_mm.shape
    tris = []
    top = heights_mm + base_h           # absolute z of each pixel's flat top

    W = cols * pixel_mm                 # total width  (x)
    D = rows * pixel_mm                 # total depth  (y)

    # ── One flat-topped box per pixel ────────────────────────────
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c*pixel_mm, (c+1)*pixel_mm
            y0, y1 = r*pixel_mm, (r+1)*pixel_mm
            tris += pixel_box(x0, y0, x1, y1, top[r, c])

    # ── Orientation marker: tall corner spike at (0,0) ───────────
    tris += corner_marker(pixel_mm, base_h, H_MAX)

    # ── Layer number ridge on front edge ─────────────────────────
    # Small rectangular ridges encoding the layer number (1 ridge = layer 1, etc.)
    ridge_w  = pixel_mm * 0.6
    ridge_h  = ARROW_HEIGHT
    ridge_gap = pixel_mm * 0.9
    for n in range(layer_idx):
        rx0 = (n * ridge_gap) + pixel_mm
        rx1 = rx0 + ridge_w
        if rx1 > W - pixel_mm:
            break
        z_base = 0
        z_top_r = base_h + ridge_h
        # Front face of ridge
        tris += [
            ((rx0,0,z_base),(rx1,0,z_base),(rx1,0,z_top_r)),
            ((rx0,0,z_base),(rx1,0,z_top_r),(rx0,0,z_top_r)),
        ]
        # Top of ridge
        tris += [
            ((rx0,0,z_top_r),(rx1,0,z_top_r),(rx1,ridge_w,z_top_r)),
            ((rx0,0,z_top_r),(rx1,ridge_w,z_top_r),(rx0,ridge_w,z_top_r)),
        ]
        # Back of ridge
        tris += [
            ((rx1,ridge_w,z_base),(rx0,ridge_w,z_base),(rx0,ridge_w,z_top_r)),
            ((rx1,ridge_w,z_base),(rx0,ridge_w,z_top_r),(rx1,ridge_w,z_top_r)),
        ]
        # Sides
        tris += [
            ((rx0,0,z_base),(rx0,ridge_w,z_base),(rx0,ridge_w,z_top_r)),
            ((rx0,0,z_base),(rx0,ridge_w,z_top_r),(rx0,0,z_top_r)),
            ((rx1,ridge_w,z_base),(rx1,0,z_base),(rx1,0,z_top_r)),
            ((rx1,ridge_w,z_base),(rx1,0,z_top_r),(rx1,ridge_w,z_top_r)),
        ]

    return tris


# ── Preview plot ───────────────────────────────────────────────────────────────
def plot_preview(mask_data, output_dir):
    n = len(mask_data)
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5),
                              facecolor='#080810')
    fig.suptitle('D2NN Phase Masks — Print Height Maps',
                 color='white', fontsize=14, fontweight='bold')
    axes = np.array(axes).flatten()

    for i, (idx, heights, phase) in enumerate(mask_data):
        ax = axes[i]
        ax.set_facecolor('#0e0e1a')
        im = ax.imshow(heights, cmap='plasma', origin='upper')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Height (mm)', color='#d0d0e8', fontsize=7)
        cb.ax.yaxis.set_tick_params(color='#d0d0e8', labelsize=6)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='#d0d0e8')
        ax.set_title(f'Layer {idx}  [{heights.min():.2f}–{heights.max():.2f} mm]',
                     color='#d0d0e8', fontsize=9)
        # Mark orientation corner
        ax.plot(0, 0, 'r^', ms=10, label='Corner (0,0)\nspike marker')
        ax.legend(fontsize=6, loc='upper left',
                  facecolor='#14142a', labelcolor='#d0d0e8',
                  edgecolor='#1e1e32')
        ax.tick_params(colors='#d0d0e8', labelsize=7)
        ax.spines[:].set_color('#1e1e32')
        # Axis labels in pixels
        ax.set_xlabel('Column (pixel)', color='#d0d0e8', fontsize=7)
        ax.set_ylabel('Row (pixel)',    color='#d0d0e8', fontsize=7)

    # Hide unused axes
    for j in range(len(mask_data), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, 'masks_preview.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#080810')
    plt.close()
    print(f"   Preview saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all mask files
    files = sorted(glob.glob(os.path.join(MASKS_DIR, 'mask_*_phase.npy')))
    if not files:
        print(f"ERROR: No mask_*_phase.npy files found in '{MASKS_DIR}'")
        print("       Run the D2NN simulation first to generate masks.")
        return

    print(f"Found {len(files)} mask(s) in '{MASKS_DIR}'\n")

    mask_data  = []
    csv_lines  = ["layer,min_height_mm,max_height_mm,relief_range_mm,pixel_size_mm,mask_size_mm"]

    for fpath in files:
        # Parse layer index from filename
        fname = os.path.basename(fpath)
        idx   = int(fname.split('_')[1])

        # Load raw trained phase (may run outside [-π, π])
        phase = np.load(fpath)

        # A phase mask only controls phase MODULO 2π (a 2π shift ≡ no shift),
        # exactly as the simulation uses exp(i·phase). So WRAP into [0, 2π)
        # instead of shifting-and-clipping, which would saturate out-of-range
        # pixels and distort the optical function.
        phase_wrapped = np.mod(phase, 2 * np.pi)           # [0, 2π)

        # Convert to physical relief height in mm (0 → flat with base, 2π → H_MAX)
        heights = (phase_wrapped / (2 * np.pi)) * H_MAX    # [0, H_MAX) mm

        rows, cols = heights.shape
        mask_mm    = cols * PIXEL_SIZE * 1e3

        print(f"Layer {idx:2d}  |  phase [{phase.min():.2f}, {phase.max():.2f}] rad  "
              f"|  height [{heights.min():.2f}, {heights.max():.2f}] mm  "
              f"|  mask {mask_mm:.0f}×{mask_mm:.0f} mm")

        # Generate STL
        tris     = heightmap_to_stl(heights, BASE_HEIGHT, PIXEL_SIZE * 1e3, idx)
        stl_path = os.path.join(OUTPUT_DIR, f'mask_{idx:02d}.stl')
        write_stl(stl_path, tris)
        print(f"           STL → {stl_path}  ({len(tris)} triangles)")

        mask_data.append((idx, heights, phase))
        csv_lines.append(
            f"{idx},{heights.min():.4f},{heights.max():.4f},"
            f"{heights.max()-heights.min():.4f},{PIXEL_SIZE*1e3:.1f},{mask_mm:.1f}"
        )

    # Preview
    print()
    plot_preview(mask_data, OUTPUT_DIR)

    # Summary CSV
    csv_path = os.path.join(OUTPUT_DIR, 'print_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_lines))
    print(f"   Summary CSV → {csv_path}")

    # Print instructions
    print(f"""
── Print Settings ────────────────────────────────────────────
   Material      : Natural/Clear PLA  (n ≈ 1.5 at {(3e8/WAVELENGTH)/1e9:.0f} GHz)
   Layer height  : 0.1 mm  (standard FDM)
   Infill        : 100%  (solid — phase accuracy requires no voids)
   Perimeters    : 3+
   Orientation   : flat on bed, relief facing UP
   Spike marker  : tall corner spike (rises above relief) = pixel (0,0)
   Ridge markers : small bumps on front edge = layer number

── Orientation guide ─────────────────────────────────────────
   When assembling the stack:
   1. Hold mask with relief surface facing the NEXT layer
   2. Tall corner spike goes to TOP-LEFT
   3. Ridge count on front edge tells you the layer number
   4. Stack order: laser → mask 1 → [30cm gap] → mask 2 → ... → detector

── Physical stack dimensions ─────────────────────────────────
   Mask size     : {GRID_SIZE * PIXEL_SIZE * 1e3:.0f} × {GRID_SIZE * PIXEL_SIZE * 1e3:.0f} mm
   Layer spacing : 300 mm
   Total length  : {len(files) * 300:.0f} mm  ({len(files)} layers)
""")


if __name__ == "__main__":
    main()