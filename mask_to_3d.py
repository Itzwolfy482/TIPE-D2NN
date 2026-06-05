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
PIXEL_SIZE   = 5e-3              # metres — physical size of each mask pixel (5mm)
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


def quad(x0, y0, x1, y1, z_bot, z_top):
    """Return 2 triangles forming a vertical quad (wall)."""
    return [
        ((x0,y0,z_bot),(x1,y1,z_bot),(x1,y1,z_top)),
        ((x0,y0,z_bot),(x1,y1,z_top),(x0,y0,z_top)),
    ]


# ── Heightmap → STL ───────────────────────────────────────────────────────────
def heightmap_to_stl(heights_mm, base_h, pixel_mm, layer_idx):
    """
    Convert a 2D height map to a solid STL mesh.

    Structure:
      - Flat bottom face at z=0
      - Vertical walls on 4 sides
      - Top surface following the heightmap
      - Orientation marker: a notched corner (top-left) so you know which way is up
      - Layer number ridge on the front edge
    """
    rows, cols = heights_mm.shape
    tris = []
    top = heights_mm + base_h           # absolute z of top surface

    W = cols * pixel_mm                 # total width  (x)
    D = rows * pixel_mm                 # total depth  (y)
    total_h = base_h + heights_mm.max()

    # ── Bottom face (z=0) ────────────────────────────────────────
    tris += [
        ((0,0,0),(W,0,0),(W,D,0)),
        ((0,0,0),(W,D,0),(0,D,0)),
    ]

    # ── Top surface — one quad per pixel ─────────────────────────
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c * pixel_mm, (c+1) * pixel_mm
            y0, y1 = r * pixel_mm, (r+1) * pixel_mm
            z00 = top[r,   c  ]
            z10 = top[r,   c+1] if c+1 < cols else top[r, c]
            z01 = top[r+1, c  ] if r+1 < rows else top[r, c]
            z11 = top[r+1, c+1] if (r+1<rows and c+1<cols) else top[r,c]
            tris += [
                ((x0,y0,z00),(x1,y0,z10),(x1,y1,z11)),
                ((x0,y0,z00),(x1,y1,z11),(x0,y1,z01)),
            ]

    # ── Side walls ───────────────────────────────────────────────
    # Front (y=0)
    for c in range(cols):
        x0, x1 = c*pixel_mm, (c+1)*pixel_mm
        z_top_l, z_top_r = top[0,c], top[0, min(c+1,cols-1)]
        tris += [
            ((x0,0,0),(x1,0,0),(x1,0,z_top_r)),
            ((x0,0,0),(x1,0,z_top_r),(x0,0,z_top_l)),
        ]
    # Back (y=D)
    for c in range(cols):
        x0, x1 = c*pixel_mm, (c+1)*pixel_mm
        z_top_l = top[rows-1, c]
        z_top_r = top[rows-1, min(c+1,cols-1)]
        tris += [
            ((x1,D,0),(x0,D,0),(x0,D,z_top_l)),
            ((x1,D,0),(x0,D,z_top_l),(x1,D,z_top_r)),
        ]
    # Left (x=0)
    for r in range(rows):
        y0, y1 = r*pixel_mm, (r+1)*pixel_mm
        z_top_b = top[r,   0]
        z_top_t = top[min(r+1,rows-1), 0]
        tris += [
            ((0,y1,0),(0,y0,0),(0,y0,z_top_b)),
            ((0,y1,0),(0,y0,z_top_b),(0,y1,z_top_t)),
        ]
    # Right (x=W)
    for r in range(rows):
        y0, y1 = r*pixel_mm, (r+1)*pixel_mm
        z_top_b = top[r,   cols-1]
        z_top_t = top[min(r+1,rows-1), cols-1]
        tris += [
            ((W,y0,0),(W,y1,0),(W,y1,z_top_t)),
            ((W,y0,0),(W,y1,z_top_t),(W,y0,z_top_b)),
        ]

    # ── Orientation marker ───────────────────────────────────────
    # A triangular notch cut into the top-left corner of the BASE
    # so after printing you always know which corner is (0,0)
    notch = pixel_mm * 1.5
    z_notch = base_h + ARROW_HEIGHT
    tris += [
        ((0,0,base_h),(notch,0,base_h),(0,notch,base_h)),          # notch top face
        ((0,0,base_h),(0,notch,base_h),(0,0,z_notch)),             # notch wall
        ((0,0,z_notch),(0,notch,base_h),(0,notch,z_notch)),
        ((notch,0,base_h),(0,0,z_notch),(notch,0,z_notch)),
        ((0,0,z_notch),(notch,0,z_notch),(0,notch,z_notch)),       # tip
    ]

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
        ax.plot(0, 0, 'r^', ms=10, label='Corner (0,0)\nnotch marker')
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

        # Load phase values [-π, π]
        phase = np.load(fpath)

        # Shift to [0, 2π] so all heights are positive
        phase_pos = phase + np.pi                          # [0, 2π]

        # Convert to physical height in mm
        heights = (phase_pos / (2 * np.pi)) * H_MAX       # [0, H_MAX] mm

        # Clamp to printable range
        heights = np.clip(heights, MIN_FEATURE, H_MAX)

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
   Notch marker  : top-left corner when relief faces you = pixel (0,0)
   Ridge markers : small bumps on front edge = layer number

── Orientation guide ─────────────────────────────────────────
   When assembling the stack:
   1. Hold mask with relief surface facing the NEXT layer
   2. Triangular notch goes to TOP-LEFT
   3. Ridge count on front edge tells you the layer number
   4. Stack order: laser → mask 1 → [30cm gap] → mask 2 → ... → detector

── Physical stack dimensions ─────────────────────────────────
   Mask size     : {28 * PIXEL_SIZE * 1e3:.0f} × {28 * PIXEL_SIZE * 1e3:.0f} mm
   Layer spacing : 300 mm
   Total length  : {len(files) * 300:.0f} mm  ({len(files)} layers)
""")


if __name__ == "__main__":
    main()