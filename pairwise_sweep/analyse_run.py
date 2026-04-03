#!/usr/bin/env python3
"""
analyse_run.py — single-run Christensen U analysis.
Usage: python3 analyse_run.py <run_dir> <v> <alpha_deg> <b_m> <chi> <mu> <csv_path>
Appends one row to csv_path (thread-safe via fcntl lock), then exits.
"""
import sys, csv, math, fcntl
from pathlib import Path
import numpy as np

# ── Christensen material constants ────────────────────────────────────
SIGMA_T = 22.4e6
SIGMA_C = 147e6
DT      = 1e-7
R_BIG   = 0.025  # m (fixed)

# ── Weibull constants (limestone particle-crushing study, Table 1) ─────
W_MOD  = 2.73        # Weibull modulus
W_D0   = 0.029       # m  — reference particle diameter
W_SIG0 = 12.858e6    # Pa — reference particle crushing stress σ_{pc,0}

SMALL_ID, BIG1_ID, BIG2_ID = 1, 2, 3
MOVING_ID = BIG1_ID
FIXED_BIG = BIG2_ID
GAP_END   = 1          # consecutive zero-overlap steps to end window
MIN_WINDOW = 3         # minimum frames for a real contact (shorter = grazing artifact)
APPLY_BRANCH_CORRECTION = True

def _christensen_params(sig_t, sig_c):
    alpha = sig_t / sig_c - 1.0
    k     = sig_c / math.sqrt(3.0)
    return alpha, k

ALPHA_C, K = _christensen_params(SIGMA_T, abs(SIGMA_C))

# ── op1.dump parser ───────────────────────────────────────────────────
def parse_op(op_path):
    """Return {step: {atom_id: (xyz_array, diameter)}}"""
    out = {}
    with open(op_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            while True:
                line = fh.readline()
                if not line:
                    return out
                if line.startswith("ITEM: NUMBER OF ATOMS"):
                    n = int(fh.readline().strip())
                    break
            while True:
                line = fh.readline()
                if line.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                    break
            while True:
                line = fh.readline()
                if not line:
                    return out
                if line.startswith("ITEM: ATOMS"):
                    break
            smap = {}
            for _ in range(n):
                c = fh.readline().split()
                if not c:
                    continue
                aid = int(c[0])
                if aid not in (SMALL_ID, BIG1_ID, BIG2_ID):
                    continue
                x, y, z = float(c[2]), float(c[3]), float(c[4])
                d = float(c[-1])
                smap[aid] = (np.array([x, y, z]), d)
            out[step] = smap
    return out

def _overlap(xi, di, xj, dj):
    return 0.5*di + 0.5*dj - float(np.linalg.norm(xi - xj))

def first_window(op_steps):
    """Return (window_steps, max_delta, chosen_pair)."""
    window, max_delta, in_contact, zeros, chosen = [], 0.0, False, 0, None
    for s in sorted(op_steps.keys()):
        d = op_steps[s]
        if MOVING_ID not in d:
            continue
        xM, dM = d[MOVING_ID]
        d12 = d23 = -1e30
        if SMALL_ID in d:
            x1, d1 = d[SMALL_ID];  d12 = _overlap(xM, dM, x1, d1)
        if FIXED_BIG in d:
            x3, d3 = d[FIXED_BIG]; d23 = _overlap(xM, dM, x3, d3)
        if not in_contact:
            if d12 > 0 or d23 > 0:
                in_contact = True
                zeros  = 0
                chosen = "2-1" if (d12 > 0 and d12 >= d23) else "2-3"
                window.append(s)
                max_delta = max(max_delta, d12 if chosen == "2-1" else d23)
        else:
            delta = d12 if chosen == "2-1" else d23
            window.append(s)
            if delta <= 0:
                zeros += 1
                if zeros >= GAP_END:
                    window.pop()
                    break
            else:
                zeros = 0
                max_delta = max(max_delta, delta)
    return window, max_delta, chosen

# ── stress.dump parser ────────────────────────────────────────────────
def _index_map(header_line):
    toks = header_line.strip().split()[2:]
    idx  = {name: i for i, name in enumerate(toks)}
    imap = {"id": idx["id"]}
    keys = [f"c_astress[{i}]" for i in range(1, 7)]
    if all(k in idx for k in keys):
        for i, k in enumerate(keys, 1):
            imap[f"s{i}"] = idx[k]
    else:
        for i, k in enumerate(("sxx","syy","szz","sxy","sxz","syz"), 1):
            imap[f"s{i}"] = idx[k]
    return imap

def iter_stress(stress_path, op_steps, steps_wanted):
    """Yield (step, sig6_cauchy) for SMALL_ID within steps_wanted."""
    if not steps_wanted:
        return
    with open(stress_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            while True:
                line = fh.readline()
                if not line:
                    return
                if line.startswith("ITEM: NUMBER OF ATOMS"):
                    n = int(fh.readline().strip())
                    break
            # advance to ITEM: ATOMS header
            atoms_hdr = None
            while True:
                line = fh.readline()
                if not line:
                    return
                if line.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                elif line.startswith("ITEM: ATOMS"):
                    atoms_hdr = line
                    break
            try:
                imap = _index_map(atoms_hdr)
            except Exception:
                for _ in range(n): fh.readline()
                continue
            if step not in steps_wanted or step not in op_steps or SMALL_ID not in op_steps[step]:
                for _ in range(n): fh.readline()
                continue
            d_small = op_steps[step][SMALL_ID][1]
            V = (math.pi / 6.0) * (d_small ** 3)
            sig6 = None
            for _ in range(n):
                cols = fh.readline().split()
                if not cols:
                    continue
                if int(cols[imap["id"]]) != SMALL_ID:
                    continue
                vir6 = tuple(float(cols[imap[f"s{i}"]]) for i in range(1, 7))
                sig6 = tuple(v / V for v in vir6)
            if sig6 is not None:
                yield step, sig6

# ── forces.dump parser (custom per-atom dump: id fx fy fz for atoms 1 & 2) ──
def parse_forces(forces_path):
    """
    Parse a custom atom dump with columns id fx fy fz for atoms 1 (ore) and 2 (mover).
    Recovers individual contact forces via Newton's 3rd law:
        F_ore_from_mover = -f_mover   (mover has no wall contact given geometry)
        F_ore_from_fixed = f_ore + f_mover
    Returns {step: {MOVING_ID: F_12, FIXED_BIG: F_13}}
    """
    out = {}
    if not Path(forces_path).exists():
        return out
    with open(forces_path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(fh.readline().strip())
            n = 0
            while True:
                hdr = fh.readline()
                if not hdr:
                    return out
                if "NUMBER OF ATOMS" in hdr:
                    n = int(fh.readline().strip())
                elif hdr.startswith("ITEM: BOX BOUNDS"):
                    fh.readline(); fh.readline(); fh.readline()
                elif hdr.startswith("ITEM: ATOMS"):
                    break
            f_ore = f_mover = None
            for _ in range(n):
                cols = fh.readline().split()
                if not cols:
                    continue
                aid = int(cols[0])
                fx, fy, fz = float(cols[1]), float(cols[2]), float(cols[3])
                if aid == SMALL_ID:
                    f_ore   = np.array([fx, fy, fz])
                elif aid == MOVING_ID:
                    f_mover = np.array([fx, fy, fz])
            if f_ore is not None and f_mover is not None:
                F_12 = -f_mover              # force on ore from mover
                F_13 = f_ore + f_mover       # force on ore from fixed
                out[step] = {MOVING_ID: F_12, FIXED_BIG: F_13}
    return out

# ── Weibull failure load threshold ────────────────────────────────────
def weibull_Ff(d_p):
    """F_{f,W}(d_p) in N: deterministic Weibull failure load threshold."""
    return W_SIG0 * (d_p / W_D0) ** (-3.0 / W_MOD) * d_p ** 2

# ── Weibull U for one timestep ─────────────────────────────────────────
def weibull_U_step(step_forces, step_atoms, d_p):
    """
    Compute U_W for a single timestep.
    step_forces: {ball_id: force_on_ore_vector} from parse_forces
    step_atoms:  {atom_id: (position_array, diameter)} from parse_op
    d_p: ore diameter (m)
    Returns U_W >= 0.  Returns 0.0 if either contact is missing.
    """
    F_mover = step_forces.get(MOVING_ID)
    F_fixed = step_forces.get(FIXED_BIG)
    if F_mover is None or F_fixed is None:
        return 0.0

    pos_ore   = step_atoms[SMALL_ID][0]
    pos_mover = step_atoms[MOVING_ID][0]
    pos_fixed = step_atoms[FIXED_BIG][0]

    # Unit vectors from ore toward each ball
    r21 = pos_fixed - pos_ore
    r23 = pos_mover - pos_ore
    r21_hat = r21 / np.linalg.norm(r21)
    r23_hat = r23 / np.linalg.norm(r23)

    # Compression axis: bisector n̂ = normalize(r̂₂₁ − r̂₂₃)
    # At head-on (α=0,b=0): r̂₂₁=−x̂, r̂₂₃=+x̂ → n̂=−x̂ (toward fixed ball).
    # This gives F₁·n̂ > 0 and −F₂·n̂ > 0 for all compression events.
    n_raw  = r21_hat - r23_hat
    n_norm = np.linalg.norm(n_raw)
    n_hat  = n_raw / n_norm   # always non-zero in our geometry

    # Project forces onto compression axis (both positive in compression)
    F1n = np.dot(F_mover, n_hat)
    F2n = -np.dot(F_fixed, n_hat)
    F_eff = max((F1n + F2n) / 2.0, 0.0)

    Ff = weibull_Ff(d_p)
    return F_eff / Ff if Ff > 0 else 0.0

# ── Christensen U ─────────────────────────────────────────────────────
def christensen_U(sig6, alpha, k):
    sxx, syy, szz, sxy, sxz, syz = sig6
    tr     = sxx + syy + szz
    dev_tr = tr / 3.0
    dxx, dyy, dzz = sxx-dev_tr, syy-dev_tr, szz-dev_tr
    q   = dxx*dxx + dyy*dyy + dzz*dzz + 2*(sxy*sxy + sxz*sxz + syz*syz)
    phi = (alpha * k / math.sqrt(3.0)) * tr + 0.5 * (1 + alpha) * q
    return max(phi / (k * k), 0.0)

# ── CSV writer (file-locked) ──────────────────────────────────────────
HEADER = ["v_mps", "a_deg", "b_m", "chi", "mu", "U_max",
          "step_of_Umax", "contact_len_s", "max_overlap_m", "UW_max"]

def write_row(csv_path, row):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            is_empty = f.tell() == 0
            wr = csv.writer(f)
            if is_empty:
                wr.writerow(HEADER)
            wr.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

# ── Main ──────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) != 8:
        print(f"Usage: {sys.argv[0]} <run_dir> <v> <alpha_deg> <b_m> <chi> <mu> <csv_path>")
        sys.exit(1)

    run_dir  = Path(sys.argv[1])
    v        = float(sys.argv[2])
    alpha    = float(sys.argv[3])
    b        = float(sys.argv[4])
    chi      = float(sys.argv[5])
    mu       = float(sys.argv[6])
    csv_path = Path(sys.argv[7])

    op_path     = run_dir / "op1.dump"
    stress_path = run_dir / "stress.dump"
    forces_path = run_dir / "forces.dump"

    if not op_path.exists() or not stress_path.exists():
        write_row(csv_path, [v, alpha, b, chi, mu, float("nan"), "", 0.0, 0.0, float("nan")])
        return

    lam = (2.0 / (1.0 + chi)) if APPLY_BRANCH_CORRECTION else 1.0

    op_steps              = parse_op(op_path)
    window, delta_max, _  = first_window(op_steps)

    if not window or len(window) < MIN_WINDOW:
        write_row(csv_path, [v, alpha, b, chi, mu, 0.0, "", 0.0, 0.0, 0.0])
        return

    steps_set = set(window)
    tlen      = (window[-1] - window[0]) * DT
    U_max, step_umax = 0.0, ""
    UW_max = 0.0

    forces_steps = parse_forces(forces_path)

    for step, sig6 in iter_stress(stress_path, op_steps, steps_set):
        if lam != 1.0:
            sig6 = tuple(lam * x for x in sig6)
        U = christensen_U(sig6, ALPHA_C, K)
        if U >= U_max:
            U_max      = U
            step_umax  = str(step)

        if step in forces_steps and step in op_steps and SMALL_ID in op_steps[step]:
            d_p = op_steps[step][SMALL_ID][1]
            UW = weibull_U_step(forces_steps[step], op_steps[step], d_p)
            UW_max = max(UW_max, UW)

    write_row(csv_path, [v, alpha, b, chi, mu, U_max, step_umax, tlen, delta_max, UW_max])

if __name__ == "__main__":
    main()
