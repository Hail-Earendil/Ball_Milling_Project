#!/usr/bin/env python3
"""
dump_interval_study.py
======================
Timestep-sensitivity study for the Christensen U_max.

Strategy (efficient):
  1. Run THREE LAMMPS simulations at dump_every=1 (every timestep saved),
     one each for v ≈ 0.95, 1.0, 1.05 × v_c  (where v_c is the head-on U=1 threshold).
  2. From each full-resolution dump, subsample at intervals
     N = 1, 2, 5, 10, 15, 20, ..., 400 — all in Python, no extra LAMMPS runs.
  3. Plot U_max vs N to show convergence; highlight the chosen interval (100 steps).

Dump files are kept in results/plots/dump_study/de1_runs/ for reuse.

Usage:
    python3 pairwise_sweep/dump_interval_study.py [--rerun]

    --rerun  : force re-run of the 3 LAMMPS simulations even if dumps exist
"""

import argparse, math, shutil, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
REPO        = SCRIPT_DIR.parent
LAMMPS_EXE  = Path("/Users/danielcui/Downloads/lammps-12Jun2025/build/lmp")
IN_TEMPLATE = REPO / "LAMMPS_model" / "dump_study.in"
ANALYSE_PY  = SCRIPT_DIR / "analyse_run.py"
CSV_PATH    = REPO / "results" / "csv" / "sweep_results.csv"
OUT_DIR     = REPO / "results" / "plots" / "dump_study"
RUNS_DIR    = OUT_DIR / "de1_runs"

# ── Fixed physics (head-on collision at default params) ────────────────────
R_BIG   = 0.025
GAP     = 0.010
CHI     = 4.0
MU_EFF  = 0.47
DT      = 1e-7
R_SMALL = R_BIG / CHI
SUM_R   = R_BIG + R_SMALL
D       = SUM_R + GAP
MU11    = MU_EFF**2 / 0.3

# Subsample intervals to test: 1,2,5 then 10,20,...,400
SUBSAMPLE_INTERVALS = (
    [1, 2, 5] +
    list(range(10, 101, 10)) +
    list(range(120, 401, 20))
)


def nsteps_for(v: float) -> int:
    return int(D / v / DT) + 5000


def make_3ps(run_dir: Path, v: float):
    c2x = SUM_R + D
    (run_dir / "3ps.dat").write_text(f"""\
LAMMPS data file

3 atoms
2 atom types

-0.12 0.12 xlo xhi
-0.12 0.12 ylo yhi
-0.12 0.12 zlo zhi

Atoms # id type diameter density x y z omegax omegay omegaz

1 2 {2*R_SMALL:.8f} 2600 {SUM_R:.8f} 0.0 0.0 0 0 0
2 1 {2*R_BIG:.8f}   7800 {c2x:.8f} 0.0 0.0 0 0 0
3 1 {2*R_BIG:.8f}   7800 0.0 0.0 0.0 0 0 0

Velocities

1 0 0 0 0 0 0
2 {-v:.8f} 0 0 0 0 0
3 0 0 0 0 0 0
""")


def find_target_velocities() -> dict:
    """Interpolate v for U_max = 0.95, 1.00, 1.05 from the sweep CSV."""
    if not CSV_PATH.exists():
        print("  [warn] CSV not found, using fallback velocities.")
        return {0.95: 2.44, 1.00: 2.53, 1.05: 2.62}

    df = pd.read_csv(CSV_PATH)
    mask = (
        (df["a_deg"].abs() < 0.5) &
        (df["b_m"]   < 1e-5) &
        ((df["chi"] - CHI).abs()    < 0.1) &
        ((df["mu"]  - MU_EFF).abs() < 0.01)
    )
    sub = df[mask].sort_values("v_mps").drop_duplicates("v_mps")
    if len(sub) < 2:
        print("  [warn] Too few CSV rows, using fallback velocities.")
        return {0.95: 2.44, 1.00: 2.53, 1.05: 2.62}

    from scipy.interpolate import interp1d
    f = interp1d(sub["U_max"], sub["v_mps"], kind="linear", fill_value="extrapolate")
    return {u: float(f(u)) for u in (0.95, 1.00, 1.05)}


def run_lammps_de1(run_dir: Path, v: float):
    """Run LAMMPS with dump_every=1. Keeps op1.dump, stress.dump, forces.dump."""
    run_dir.mkdir(parents=True, exist_ok=True)
    make_3ps(run_dir, v)
    shutil.copy(IN_TEMPLATE, run_dir / "dump_study.in")

    ns = nsteps_for(v)
    print(f"    Running LAMMPS: v={v:.4f} m/s, nsteps={ns}, dump_every=1 ...", flush=True)
    r = subprocess.run(
        [str(LAMMPS_EXE),
         "-in", "dump_study.in",
         "-v", "nsteps",     str(ns),
         "-v", "mu11",       f"{MU11:.6f}",
         "-v", "dump_every", "1",
         "-log", "none"],
        cwd=run_dir, capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  LAMMPS FAILED:\n{r.stderr[-600:]}")
        sys.exit(1)
    print(f"    Done.")


# ── Analysis helpers (inline, to avoid subprocess overhead per subsample) ──
def _parse_op(op_path):
    """Parse op1.dump → {step: {atom_id: (xyz, diam)}}"""
    import re
    out = {}
    with open(op_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            n = 0
            while True:
                h = fh.readline()
                if not h:
                    return out
                if "NUMBER OF ATOMS" in h:
                    n = int(fh.readline().strip())
                    break
            while True:
                h = fh.readline()
                if h.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                    break
            while True:
                h = fh.readline()
                if not h:
                    return out
                if h.startswith("ITEM: ATOMS"):
                    break
            smap = {}
            for _ in range(n):
                c = fh.readline().split()
                if not c:
                    continue
                aid = int(c[0])
                if aid not in (1, 2, 3):
                    continue
                smap[aid] = (np.array([float(c[2]), float(c[3]), float(c[4])]), float(c[-1]))
            out[step] = smap
    return out


def _overlap(xi, di, xj, dj):
    return 0.5*di + 0.5*dj - float(np.linalg.norm(xi - xj))


def _contact_window(op_steps, gap_end=1, min_window=3):
    window, max_delta, in_contact, zeros, chosen = [], 0.0, False, 0, None
    for s in sorted(op_steps.keys()):
        d = op_steps[s]
        if 2 not in d:
            continue
        xM, dM = d[2]
        d12 = _overlap(xM, dM, *d[1]) if 1 in d else -1e30
        d23 = _overlap(xM, dM, *d[3]) if 3 in d else -1e30
        if not in_contact:
            if d12 > 0 or d23 > 0:
                in_contact = True; zeros = 0
                chosen = "2-1" if (d12 > 0 and d12 >= d23) else "2-3"
                window.append(s)
                max_delta = max(max_delta, d12 if chosen == "2-1" else d23)
        else:
            delta = d12 if chosen == "2-1" else d23
            window.append(s)
            if delta <= 0:
                zeros += 1
                if zeros >= gap_end:
                    window.pop(); break
            else:
                zeros = 0; max_delta = max(max_delta, delta)
    if len(window) < min_window:
        return [], 0.0
    return window, max_delta


def _parse_stress(stress_path, op_steps, steps_wanted):
    """Yield (step, sig6_cauchy) for atom 1 within steps_wanted."""
    if not steps_wanted:
        return
    steps_wanted = set(steps_wanted)
    with open(stress_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            n = 0
            atoms_hdr = None
            while True:
                h = fh.readline()
                if not h:
                    return
                if "NUMBER OF ATOMS" in h:
                    n = int(fh.readline().strip())
                elif h.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                elif h.startswith("ITEM: ATOMS"):
                    atoms_hdr = h; break
            if step not in steps_wanted or step not in op_steps or 1 not in op_steps[step]:
                for _ in range(n): fh.readline()
                continue
            # build column map
            toks = atoms_hdr.strip().split()[2:]
            idx = {t: i for i, t in enumerate(toks)}
            keys = [f"c_astress[{i}]" for i in range(1, 7)]
            if all(k in idx for k in keys):
                s_idx = [idx[k] for k in keys]
            else:
                s_idx = [idx[k] for k in ("sxx","syy","szz","sxy","sxz","syz")]
            d_small = op_steps[step][1][1]
            V = (math.pi / 6.0) * d_small**3
            sig6 = None
            for _ in range(n):
                cols = fh.readline().split()
                if not cols:
                    continue
                if int(cols[0]) != 1:
                    continue
                sig6 = tuple(float(cols[i]) / V for i in s_idx)
            if sig6 is not None:
                yield step, sig6


def _christensen_U(sig6, chi):
    SIGMA_T = 22.4e6; SIGMA_C = 147e6
    alpha_c = SIGMA_T / SIGMA_C - 1.0
    k = SIGMA_C / math.sqrt(3.0)
    lam = 2.0 / (1.0 + chi)
    sig6 = tuple(lam * x for x in sig6)
    sxx, syy, szz, sxy, sxz, syz = sig6
    tr = sxx + syy + szz
    dev = tr / 3.0
    dxx, dyy, dzz = sxx-dev, syy-dev, szz-dev
    q = dxx**2 + dyy**2 + dzz**2 + 2*(sxy**2 + sxz**2 + syz**2)
    phi = (alpha_c * k / math.sqrt(3.0)) * tr + 0.5 * (1 + alpha_c) * q
    return max(phi / (k * k), 0.0)


def compute_U_subsampled(run_dir: Path, interval: int, op_steps: dict, full_window: list) -> float:
    """Compute U_max using only every `interval`-th step from the contact window."""
    if not full_window:
        return float("nan")
    # subsample: take every Nth step from the window
    subsampled = full_window[::interval]
    if len(subsampled) < 2:
        return float("nan")
    steps_set = set(subsampled)
    stress_path = run_dir / "stress.dump"
    U_max = 0.0
    for step, sig6 in _parse_stress(stress_path, op_steps, steps_set):
        U = _christensen_U(sig6, CHI)
        U_max = max(U_max, U)
    return U_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", action="store_true",
                    help="Re-run LAMMPS even if dump files already exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    targets = find_target_velocities()
    print("Target velocities:")
    for u_t, v in targets.items():
        print(f"  U_target={u_t:.2f}  →  v = {v:.4f} m/s")

    # ── Phase 1: run at dump_every=1 ──────────────────────────────────────
    print("\nPhase 1: Running 3 LAMMPS simulations at dump_every=1")
    for u_t, v in targets.items():
        run_dir = RUNS_DIR / f"U{u_t:.2f}"
        op_exists = (run_dir / "op1.dump").exists()
        stress_exists = (run_dir / "stress.dump").exists()
        if op_exists and stress_exists and not args.rerun:
            print(f"  U_target={u_t:.2f}: dumps exist, skipping (use --rerun to force)")
        else:
            run_lammps_de1(run_dir, v)

    # ── Phase 2: subsample and compute U ──────────────────────────────────
    print(f"\nPhase 2: Subsampling at {len(SUBSAMPLE_INTERVALS)} intervals ...")
    results = {}
    for u_t, v in targets.items():
        run_dir = RUNS_DIR / f"U{u_t:.2f}"
        print(f"  Parsing dumps for U_target={u_t:.2f} ...", flush=True)
        op_steps = _parse_op(run_dir / "op1.dump")
        full_window, _ = _contact_window(op_steps)
        print(f"    Contact window: {len(full_window)} steps")

        U_vs_interval = {}
        for N in SUBSAMPLE_INTERVALS:
            U_vs_interval[N] = compute_U_subsampled(run_dir, N, op_steps, full_window)
        results[u_t] = U_vs_interval
        print(f"    U at N=1: {U_vs_interval[1]:.4f}  |  N=100: {U_vs_interval.get(100, float('nan')):.4f}")

    # ── Phase 3: plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    color_map = {0.95: "royalblue", 1.00: "firebrick", 1.05: "darkorange"}
    label_map = {
        0.95: r"$v = 0.95\,v_c$  (sub-threshold)",
        1.00: r"$v = v_c$  (threshold)",
        1.05: r"$v = 1.05\,v_c$  (above threshold)",
    }

    for u_t, U_vs_N in results.items():
        Ns = sorted(U_vs_N)
        Us = [U_vs_N[n] for n in Ns]
        # normalise by U at N=1 (ground truth) so all curves start at 1.0
        U1 = U_vs_N[1] if U_vs_N[1] > 0 else 1.0
        Us_norm = [u / U1 for u in Us]
        ax.plot(Ns, Us_norm, "o-", color=color_map[u_t],
                label=label_map[u_t], linewidth=2, markersize=5)

    ax.axvline(100, color="grey", linestyle=":", linewidth=1.5, label="Chosen interval (100 steps)")
    ax.axhline(1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.3)

    ax.set_xlabel("Dump interval $N$ (steps)  [= output every $N \\times 0.1\\,\\mu$s]", fontsize=13)
    ax.set_ylabel(r"$U_\mathrm{max}(N)\,/\,U_\mathrm{max}(1)$", fontsize=13)
    ax.set_title(
        r"Sensitivity of $U_\mathrm{max}$ to dump interval"
        "\n"
        r"(head-on, $\alpha=0$, $b=0$, $\chi=4$, $\mu=0.47$)",
        fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0.9, 1.1)

    out_png = OUT_DIR / "dump_interval_sensitivity.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out_png}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n── U_max table (absolute) ──")
    header = f"{'N':>6}" + "".join(f"  U(target={u:.2f})" for u in sorted(results))
    print(header)
    for N in SUBSAMPLE_INTERVALS:
        row = f"{N:>6}"
        for u_t in sorted(results):
            row += f"  {results[u_t].get(N, float('nan')):>16.4f}"
        print(row)


if __name__ == "__main__":
    main()
