#!/usr/bin/env python3
"""
plot_timeseries.py
==================
Visualises a single LAMMPS run as 4 time-series panels:
  1. Contact force F_x, F_y on ore (atom 1) from mover (atom 2)
  2. Angular velocity Ω_z of atoms 1 (ore) and 2 (mover)
  3. Translational velocity v_x, v_y of atoms 1 and 2
  4. Rotational KE of each atom as % of initial translational KE of mover

Dumps at every timestep are kept in results/plots/timeseries/runs/ for reuse.

Usage:
    python3 pairwise_sweep/plot_timeseries.py [options]

    --v V        Impact velocity (m/s)   [default: U=1 threshold from CSV]
    --alpha A    Impact angle (deg)      [default: 20]
    --b B        Impact parameter (m)    [default: 0]
    --chi C      Size ratio R/r          [default: 4]
    --mu M       Friction coefficient    [default: 0.47]
    --rerun      Force re-run LAMMPS even if dumps already exist
"""

import argparse, math, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO        = SCRIPT_DIR.parent
LAMMPS_EXE  = Path("/Users/danielcui/Downloads/lammps-12Jun2025/build/lmp")
IN_TEMPLATE = REPO / "LAMMPS_model" / "timeseries.in"
CSV_PATH    = REPO / "results" / "csv" / "sweep_results.csv"
OUT_DIR     = REPO / "results" / "plots" / "timeseries"
RUNS_DIR    = OUT_DIR / "runs"

# ── Constants ──────────────────────────────────────────────────────────────
R_BIG      = 0.025
GAP        = 0.010
DT         = 1e-7        # LAMMPS timestep (s)
RHO_ORE    = 2600.0      # kg/m³  (atom 1, limestone)
RHO_BALL   = 7800.0      # kg/m³  (atom 2, steel mover)
BUFFER_STEPS = 500       # steps shown before/after contact window


# ── Physics helpers ────────────────────────────────────────────────────────
def sphere_mass(rho, R):
    return rho * (4.0/3.0) * math.pi * R**3

def sphere_moi(m, R):
    return 0.4 * m * R**2   # 2/5 m R²

def nsteps_for(v, alpha_deg, b, chi):
    """Exact quadratic nsteps calculation (same as sweep.sh)."""
    R_SMALL = R_BIG / chi
    SUM_R   = R_BIG + R_SMALL
    D       = SUM_R + GAP
    rad     = math.radians(alpha_deg)
    C2x     = SUM_R + D*math.cos(rad) + b*math.sin(rad)
    C2y     = -D*math.sin(rad) + b*math.cos(rad)
    vx      = -v*math.cos(rad)
    vy      =  v*math.sin(rad)
    dx, dy  = C2x - SUM_R, C2y
    A = vx**2 + vy**2
    B = 2*(dx*vx + dy*vy)
    C = dx**2 + dy**2 - SUM_R**2
    disc = B**2 - 4*A*C
    if disc < 0:
        return 5000
    T = (-B - math.sqrt(disc)) / (2*A)
    return max(5000, int(T/DT) + 5000)


# ── Setup & run ────────────────────────────────────────────────────────────
def make_3ps(run_dir, v, alpha_deg, b, chi, mu):
    R_SMALL = R_BIG / chi
    SUM_R   = R_BIG + R_SMALL
    D       = SUM_R + GAP
    rad     = math.radians(alpha_deg)
    C2x     = SUM_R + D*math.cos(rad) + b*math.sin(rad)
    C2y     = -D*math.sin(rad) + b*math.cos(rad)
    vx      = -v*math.cos(rad)
    vy      =  v*math.sin(rad)
    mu11    = mu**2 / 0.3

    (run_dir / "3ps.dat").write_text(f"""\
LAMMPS data file

3 atoms
2 atom types

-0.12 0.12 xlo xhi
-0.12 0.12 ylo yhi
-0.12 0.12 zlo zhi

Atoms # id type diameter density x y z omegax omegay omegaz

1 2 {2*R_SMALL:.8f} 2600 {SUM_R:.8f} 0.0 0.0 0 0 0
2 1 {2*R_BIG:.8f}   7800 {C2x:.8f} {C2y:.8f} 0.0 0 0 0
3 1 {2*R_BIG:.8f}   7800 0.0 0.0 0.0 0 0 0

Velocities

1 0 0 0 0 0 0
2 {vx:.8f} {vy:.8f} 0 0 0 0
3 0 0 0 0 0 0
""")
    return mu11


def run_lammps(run_dir, v, alpha_deg, b, chi, mu):
    mu11 = make_3ps(run_dir, v, alpha_deg, b, chi, mu)
    ns   = nsteps_for(v, alpha_deg, b, chi)
    shutil.copy(IN_TEMPLATE, run_dir / "timeseries.in")
    print(f"  Running LAMMPS: nsteps={ns}, dump_every=1 ...", flush=True)
    r = subprocess.run(
        [str(LAMMPS_EXE), "-in", "timeseries.in",
         "-v", "nsteps",     str(ns),
         "-v", "mu11",       f"{mu11:.6f}",
         "-v", "dump_every", "1",
         "-log", "none"],
        cwd=run_dir, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"LAMMPS error:\n{r.stderr[-600:]}")
        sys.exit(1)
    print("  Done.")


# ── Dump parsers ───────────────────────────────────────────────────────────
def parse_op1(op_path):
    """Return {step: {atom_id: {'vx','vy','omegaz','diam','x','y'}}}"""
    out = {}
    with open(op_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline())
            n = 0
            while True:
                h = fh.readline()
                if not h: return out
                if "NUMBER OF ATOMS" in h:
                    n = int(fh.readline()); break
            atoms_hdr = None
            while True:
                h = fh.readline()
                if not h: return out
                if h.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                elif h.startswith("ITEM: ATOMS"):
                    atoms_hdr = h; break
            toks = atoms_hdr.strip().split()[2:]
            idx  = {t: i for i, t in enumerate(toks)}
            smap = {}
            for _ in range(n):
                c = fh.readline().split()
                if not c: continue
                aid = int(c[0])
                if aid not in (1, 2): continue
                smap[aid] = {
                    'x':      float(c[idx['x']]),
                    'y':      float(c[idx['y']]),
                    'vx':     float(c[idx['vx']]),
                    'vy':     float(c[idx['vy']]),
                    'omegaz': float(c[idx['omegaz']]),
                    'diam':   float(c[idx['diameter']]),
                }
            out[step] = smap
    return out


def parse_forces(forces_path):
    """Return {step: array([fx2, fy2])} — total force on atom 2 (mover).
    F_12 (force on ore from mover) = -[fx2, fy2]  by Newton's 3rd law.
    """
    out = {}
    if not Path(forces_path).exists():
        return out
    with open(forces_path) as fh:
        while True:
            line = fh.readline()
            if not line: break
            if not line.startswith("ITEM: TIMESTEP"): continue
            step = int(fh.readline())
            n = 0
            while True:
                h = fh.readline()
                if not h: return out
                if "NUMBER OF ATOMS" in h:
                    n = int(fh.readline()); break
            while True:
                h = fh.readline()
                if not h: return out
                if h.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                elif h.startswith("ITEM: ATOMS"): break
            f2 = None
            for _ in range(n):
                c = fh.readline().split()
                if not c: continue
                if int(c[0]) == 2:
                    f2 = np.array([float(c[1]), float(c[2])])
            if f2 is not None:
                out[step] = f2
    return out


def find_contact_window(op_steps):
    """Return list of steps in the first ore–mover contact window."""
    window, in_contact, zeros = [], False, 0
    for s in sorted(op_steps):
        d = op_steps[s]
        if 1 not in d or 2 not in d: continue
        r1 = d[1]['diam'] / 2
        r2 = d[2]['diam'] / 2
        dist = math.hypot(d[2]['x'] - d[1]['x'], d[2]['y'] - d[1]['y'])
        overlap = r1 + r2 - dist
        if not in_contact:
            if overlap > 0:
                in_contact = True; zeros = 0; window.append(s)
        else:
            window.append(s)
            if overlap <= 0:
                zeros += 1
                if zeros >= 1:
                    window.pop(); break
            else:
                zeros = 0
    return window


def get_threshold_v(chi=4.0, mu=0.47):
    """Interpolate U=1 threshold velocity from sweep CSV."""
    if not CSV_PATH.exists():
        return 2.53
    df   = pd.read_csv(CSV_PATH)
    mask = ((df.a_deg.abs() < 0.5) & (df.b_m < 1e-5) &
            ((df.chi - chi).abs() < 0.1) & ((df.mu - mu).abs() < 0.01))
    sub  = df[mask].sort_values("v_mps").drop_duplicates("v_mps")
    if len(sub) < 2:
        return 2.53
    from scipy.interpolate import interp1d
    return float(interp1d(sub.U_max, sub.v_mps)(1.0))


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v",     type=float, default=None,  help="Impact velocity (m/s)")
    ap.add_argument("--alpha", type=float, default=20.0,  help="Impact angle (deg)")
    ap.add_argument("--b",     type=float, default=0.0,   help="Impact parameter (m)")
    ap.add_argument("--chi",   type=float, default=4.0,   help="Size ratio R/r")
    ap.add_argument("--mu",    type=float, default=0.47,  help="Friction coefficient")
    ap.add_argument("--rerun", action="store_true",       help="Force re-run LAMMPS")
    args = ap.parse_args()

    chi, mu = args.chi, args.mu
    v     = args.v if args.v is not None else get_threshold_v(chi, mu)
    alpha = args.alpha
    b     = args.b

    R_SMALL = R_BIG / chi
    M1  = sphere_mass(RHO_ORE,  R_SMALL)
    M2  = sphere_mass(RHO_BALL, R_BIG)
    I1  = sphere_moi(M1, R_SMALL)
    I2  = sphere_moi(M2, R_BIG)
    KE0 = 0.5 * M2 * v**2   # initial translational KE of mover

    print(f"Parameters: v={v:.4f} m/s, α={alpha}°, b={b} m, χ={chi}, μ={mu}")
    print(f"M1={M1*1e3:.3f} g, M2={M2*1e3:.1f} g, I1={I1:.3e}, I2={I2:.3e} kg·m²")
    print(f"Initial KE (mover) = {KE0:.4f} J")

    # ── Run directory ──────────────────────────────────────────────────────
    def lbl(x):
        return f"{x:.4g}".replace('-', 'm').replace('.', 'p')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tag     = f"chi{lbl(chi)}_v{lbl(v)}_a{lbl(alpha)}_b{lbl(b)}_mu{lbl(mu)}"
    run_dir = RUNS_DIR / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    op_path     = run_dir / "op1.dump"
    forces_path = run_dir / "forces.dump"

    if not op_path.exists() or not forces_path.exists() or args.rerun:
        run_lammps(run_dir, v, alpha, b, chi, mu)
    else:
        print("  Dumps exist — skipping LAMMPS (pass --rerun to force)")

    # ── Parse ──────────────────────────────────────────────────────────────
    print("  Parsing dumps ...", flush=True)
    op_steps    = parse_op1(op_path)
    force_steps = parse_forces(forces_path)
    window      = find_contact_window(op_steps)

    if not window:
        print("ERROR: No contact window found."); sys.exit(1)
    w0, w1 = window[0], window[-1]
    print(f"  Contact: steps {w0}–{w1}  ({len(window)} steps = {len(window)*DT*1e6:.2f} µs)")

    # ── Assemble time series ───────────────────────────────────────────────
    all_steps  = sorted(op_steps)
    plot_steps = [s for s in all_steps if w0 - BUFFER_STEPS <= s <= w1 + BUFFER_STEPS]
    t_us       = np.array([(s - w0) * DT * 1e6 for s in plot_steps])

    def get(aid, key):
        return np.array([op_steps[s][aid][key] if (s in op_steps and aid in op_steps[s])
                         else np.nan for s in plot_steps])

    vx1, vy1, oz1 = get(1,'vx'), get(1,'vy'), get(1,'omegaz')
    vx2, vy2, oz2 = get(2,'vx'), get(2,'vy'), get(2,'omegaz')

    # F_12 = force on ore FROM mover = -f_mover  (Newton's 3rd)
    Fx12 = np.array([-force_steps[s][0] if s in force_steps else np.nan for s in plot_steps])
    Fy12 = np.array([-force_steps[s][1] if s in force_steps else np.nan for s in plot_steps])

    # Rotational KE (normalised by initial KE)
    KE_rot1 = 0.5 * I1 * oz1**2 / KE0 * 100   # percent
    KE_rot2 = 0.5 * I2 * oz2**2 / KE0 * 100
    KE_rot_total = KE_rot1 + KE_rot2

    peak_pct = np.nanmax(KE_rot_total)
    print(f"  Peak total rotational KE / KE_initial = {peak_pct:.4f}%")
    print(f"    Mover (atom 2): {np.nanmax(KE_rot2):.4f}%")
    print(f"    Ore   (atom 1): {np.nanmax(KE_rot1):.6f}%")

    # ── Plot ───────────────────────────────────────────────────────────────
    fs = 13
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs,
                         "xtick.labelsize": fs-1, "ytick.labelsize": fs-1})

    fig = plt.figure(figsize=(10, 12))
    gs  = gridspec.GridSpec(4, 1, hspace=0.55, figure=fig)
    axs = [fig.add_subplot(gs[i]) for i in range(4)]

    # Grey band marks the actual contact window
    shade = dict(alpha=0.10, color="steelblue")
    t_shade = [(w0 - w0) * DT * 1e6, (w1 - w0) * DT * 1e6]

    # Panel 1 — contact force
    axs[0].plot(t_us, Fx12, color="royalblue", lw=1.5, label=r"$F_x$")
    axs[0].plot(t_us, Fy12, color="firebrick",  lw=1.5, label=r"$F_y$")
    axs[0].axhline(0, color="black", lw=0.7, alpha=0.3)
    axs[0].axvspan(*t_shade, **shade)
    axs[0].set_ylabel("Force (N)")
    axs[0].set_title(r"Contact force on ore (1) from mover (2):  $\mathbf{F}_{12} = -\mathbf{f}_2$")
    axs[0].legend(fontsize=fs-2, loc="upper right")

    # Panel 2 — angular velocity
    axs[1].plot(t_us, oz1, color="royalblue", lw=1.5, label=r"$\Omega_{z,1}$ (ore)")
    axs[1].plot(t_us, oz2, color="firebrick",  lw=1.5, label=r"$\Omega_{z,2}$ (mover)")
    axs[1].axhline(0, color="black", lw=0.7, alpha=0.3)
    axs[1].axvspan(*t_shade, **shade)
    axs[1].set_ylabel(r"$\Omega_z$ (rad/s)")
    axs[1].set_title(r"Angular velocity ($z$-component)")
    axs[1].legend(fontsize=fs-2, loc="upper right")

    # Panel 3 — translational velocity
    axs[2].plot(t_us, vx1, color="royalblue", lw=1.5, label=r"$v_{x,1}$ ore")
    axs[2].plot(t_us, vy1, color="royalblue", lw=1.5, linestyle="--", label=r"$v_{y,1}$ ore")
    axs[2].plot(t_us, vx2, color="firebrick",  lw=1.5, label=r"$v_{x,2}$ mover")
    axs[2].plot(t_us, vy2, color="firebrick",  lw=1.5, linestyle="--", label=r"$v_{y,2}$ mover")
    axs[2].axhline(0, color="black", lw=0.7, alpha=0.3)
    axs[2].axvspan(*t_shade, **shade)
    axs[2].set_ylabel("Velocity (m/s)")
    axs[2].set_title("Translational velocity")
    axs[2].legend(fontsize=fs-2, loc="upper right", ncol=2)

    # Panel 4 — rotational energy fraction
    axs[3].plot(t_us, KE_rot1, color="royalblue", lw=1.5, label=r"Ore (1)")
    axs[3].plot(t_us, KE_rot2, color="firebrick",  lw=1.5, label=r"Mover (2)")
    axs[3].plot(t_us, KE_rot_total, color="black", lw=2.0,
                linestyle="--", label=rf"Total  (peak = {peak_pct:.3f}%)")
    axs[3].axvspan(*t_shade, **shade)
    axs[3].set_ylabel(r"$KE_\mathrm{rot}\,/\,KE_0$ (%)")
    axs[3].set_title(r"Rotational KE as % of initial mover KE  —  "
                     r"quantifies error in no-rotation assumption")
    axs[3].legend(fontsize=fs-2, loc="upper right")

    for ax in axs:
        ax.set_xlabel(r"Time from contact start ($\mu$s)")
        ax.grid(True, alpha=0.25)
        ax.tick_params(length=5)

    fig.suptitle(
        rf"$v = {v:.3f}$ m/s,  $\alpha = {alpha:.0f}°$,  "
        rf"$b = {b:.4f}$ m,  $\chi = {chi:.0f}$,  $\mu = {mu:.2f}$"
        "\n(grey band = contact window)",
        fontsize=fs, y=1.01)

    plt.tight_layout()
    out_png = OUT_DIR / f"timeseries_{tag}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")


if __name__ == "__main__":
    main()
