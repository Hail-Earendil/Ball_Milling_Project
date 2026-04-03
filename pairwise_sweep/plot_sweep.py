#!/usr/bin/env python3
"""
plot_sweep.py — heatmap of Christensen U or Weibull U_W for a pairwise sweep.

Usage:
    python3 plot_sweep.py <PAIR> [--weibull]

PAIR: v_alpha | v_b | v_chi | v_mu | alpha_b | alpha_chi | alpha_mu |
      b_chi   | b_mu | chi_mu

Reads results/sweep_results.csv, filters to the chosen pair
(other 3 params fixed at defaults), plots heatmap of U_max.
Pass --weibull to plot UW_max (Weibull criterion) rescaled to the
fixed reference dp0=120 mm, sigma_pc0=5.579 MPa.

Output directories:
    results/plots/christensen/   (default)
    results/plots/weibull/       (--weibull)
"""
import sys, math
from pathlib import Path
import numpy as np
import matplotlib.lines as mlines
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FormatStrFormatter

# ── Defaults & tolerances ────────────────────────────────────────────
DEF = {"v_mps": 3.0, "a_deg": 0.0, "b_m": 0.0, "chi": 4.0, "mu": 0.47}
TOL = {"v_mps": 0.05, "a_deg": 0.5, "b_m": 1e-5, "chi": 0.1, "mu": 0.01}
R_BIG = 0.025

# ── Theory 2: analytic Christensen fracture boundary ─────────────────
# Material constants from LAMMPS pair_coeff (ore type1, ball type2)
_E_ORE,  _NU_ORE  = 7.782e10, 0.28
_E_BALL, _NU_BALL = 2.08e11,  0.27
_RHO_MOVER = 7800.0   # kg/m³ (steel grinding balls)
_SIGMA_C   = 147e6    # Pa (limestone compressive strength)

# ── Weibull scaling constants ─────────────────────────────────────────
_W_MOD      = 2.73
_W_D0_CSV   = 0.029       # reference used when generating the CSV
_W_SIG0_CSV = 12.858e6    # Pa
_W_D0_PLOT  = 0.120       # fixed reference for plotting (dp0 = 120 mm)
_W_SIG0_PLOT = 5.579e6    # Pa

def _weibull_scale_factor():
    """Multiplier: CSV UW_max → UW_max at the fixed plotting reference."""
    exp = 3.0 / _W_MOD
    return (_W_SIG0_CSV * _W_D0_CSV**exp) / (_W_SIG0_PLOT * _W_D0_PLOT**exp)

def _vc0_weibull(chi):
    """Head-on critical velocity [m/s] from Weibull energy balance (Eq. 22)."""
    RS     = R_BIG / chi
    dp     = 2.0 * RS
    R_star = 1.0 / (1.0/RS + 1.0/R_BIG)
    E_star = 1.0 / ((1 - _NU_ORE**2)/_E_ORE + (1 - _NU_BALL**2)/_E_BALL)
    Epc = (4.989 / math.pi) \
          * E_star**(-2.0/3.0) \
          * (RS / R_star)**(1.0/3.0) \
          * _W_SIG0_PLOT**(5.0/3.0) \
          * _W_D0_PLOT**(5.0/_W_MOD) \
          * dp**(-5.0/_W_MOD)
    Vp     = (4.0/3.0) * math.pi * RS**3
    M_proj = (4.0/3.0) * math.pi * R_BIG**3 * _RHO_MOVER
    return math.sqrt(2.0 * Vp * Epc / M_proj)

_E_EFF = 1.0 / ((1 - _NU_ORE**2) / _E_ORE + (1 - _NU_BALL**2) / _E_BALL)

def _vc0(chi):
    """Head-on critical fracture velocity [m/s] as a function of size ratio chi."""
    num = 2**(2/3) * math.sqrt(5) * math.pi**(1/3) \
          * _SIGMA_C**(5/6) * chi**(-5/3) * (1 + chi)**(1/6)
    den = 5 * _RHO_MOVER**0.5 * _E_EFF**(1/3)
    return num / den

def _theory2_curves(pair, x_vals, y_vals):
    """
    Return list of (x_arr, y_arr, label) tuples for Theory 2 boundary curves.
    Only the 6 non-mu pairs are supported; others return [].
    """
    v_fix   = DEF["v_mps"]
    chi_fix = DEF["chi"]
    Rr_fix  = R_BIG * (1.0 + 1.0 / chi_fix)   # R+r at default chi

    if pair == "v_alpha":
        # x=v, y=alpha_deg; b=0, chi=chi_fix
        # boundary: alpha_c = ±arccos(vc0/v)
        vc = _vc0(chi_fix)
        v_arr = np.linspace(vc, x_vals.max(), 400)
        a_arr = np.degrees(np.arccos(np.clip(vc / v_arr, 0, 1)))
        return [(v_arr,  a_arr, "Theory 2"),
                (v_arr, -a_arr, None)]

    elif pair == "v_b":
        # x=v, y=b_m; alpha=0, chi=chi_fix
        # boundary: b_c = (R+r)*sqrt(1-(vc0/v)^2)
        vc = _vc0(chi_fix)
        v_arr = np.linspace(vc, x_vals.max(), 400)
        b_arr = Rr_fix * np.sqrt(np.clip(1 - (vc / v_arr)**2, 0, None))
        return [(v_arr, b_arr, "Theory 2")]

    elif pair == "v_chi":
        # x=v, y=chi; alpha=0, b=0
        # boundary: v = vc0(chi)  → x = vc0(y)
        chi_arr = np.linspace(y_vals.min(), y_vals.max(), 400)
        vc_arr  = np.array([_vc0(c) for c in chi_arr])
        mask = (vc_arr >= x_vals.min()) & (vc_arr <= x_vals.max())
        if not mask.any():
            return []
        return [(vc_arr[mask], chi_arr[mask], "Theory 2")]

    elif pair == "v_mu":
        # x=v, y=mu; alpha=20°, b=0, chi=chi_fix
        # boundary: v_c = vc0(chi_fix) / |cos alpha|  — vertical line, independent of mu
        a_fix = math.radians(PAIR_DEF_OVERRIDE["v_mu"]["a_deg"])
        vc = _vc0(chi_fix) / abs(math.cos(a_fix))
        if vc < x_vals.min() or vc > x_vals.max():
            return []
        mu_full = np.array([float(y_vals.min()), float(y_vals.max())])
        return [(np.array([vc, vc]), mu_full, "Analytic Christensen theory")]

    elif pair == "chi_mu":
        # x=chi, y=mu; alpha=20°, b=0, v=v_fix
        # boundary: v_c0(chi) / |cos alpha| = v_fix → v_c0(chi) = v_fix * |cos alpha|
        a_fix  = math.radians(PAIR_DEF_OVERRIDE["chi_mu"]["a_deg"])
        v_thr  = v_fix * abs(math.cos(a_fix))
        chi_arr = np.linspace(y_vals.min(), y_vals.max(), 400)
        vc_arr  = np.array([_vc0(c) for c in chi_arr])
        mask = (vc_arr >= v_thr - 0.01) & (vc_arr <= v_thr + 0.01)
        # Find chi where vc0(chi) = v_thr by interpolation
        from scipy.interpolate import interp1d
        try:
            f = interp1d(vc_arr[::-1], chi_arr[::-1])  # vc0 decreasing in chi
            chi_c = float(f(v_thr))
        except Exception:
            return []
        if chi_c < x_vals.min() or chi_c > x_vals.max():
            return []
        mu_full = np.array([float(y_vals.min()), float(y_vals.max())])
        return [(np.array([chi_c, chi_c]), mu_full, "Analytic Christensen theory")]

    elif pair == "alpha_b":
        # x=alpha_deg, y=b_m; v=v_fix, chi=chi_fix
        # boundary: b_c(alpha) = (R+r)*sqrt(1-(vc0/(v|cos a|))^2)
        vc = _vc0(chi_fix)
        if v_fix < vc:
            return []   # v below threshold even head-on → no fracture at all
        a_arr = np.linspace(x_vals.min(), x_vals.max(), 400)
        cos_a = np.abs(np.cos(np.radians(a_arr)))
        ratio = np.where(cos_a > 1e-9, vc / (v_fix * cos_a), np.inf)
        mask  = ratio <= 1.0
        b_arr = np.where(mask, Rr_fix * np.sqrt(np.clip(1 - ratio**2, 0, None)), np.nan)
        return [(a_arr, b_arr, "Theory 2")]

    elif pair == "alpha_chi":
        # x=chi, y=alpha_deg; v=v_fix, b=0
        # boundary: alpha_c(chi) = ±arccos(vc0(chi)/v)
        chi_arr = np.linspace(x_vals.min(), x_vals.max(), 400)
        vc_arr  = np.array([_vc0(c) for c in chi_arr])
        ratio   = vc_arr / v_fix
        mask    = ratio <= 1.0
        if not mask.any():
            return []
        a_c = np.degrees(np.arccos(np.clip(ratio[mask], 0, 1)))
        return [(chi_arr[mask],  a_c, "Theory 2"),
                (chi_arr[mask], -a_c, None)]

    elif pair == "b_chi":
        # x=b_frac, y=chi; v=v_fix, alpha=0
        # b_frac = b/(R+r) = sin(gamma), boundary: b_frac_c = sqrt(1-(vc0/v)^2)
        chi_arr = np.linspace(y_vals.min(), y_vals.max(), 400)
        vc_arr  = np.array([_vc0(c) for c in chi_arr])
        ratio   = vc_arr / v_fix
        mask    = ratio <= 1.0
        if not mask.any():
            return []
        bf_c = np.sqrt(np.clip(1 - ratio[mask]**2, 0, None))
        return [(bf_c, chi_arr[mask], "Theory 2")]

    return []


def _weibull_theory_curves(pair, x_vals, y_vals):
    """
    Analytic Weibull fracture boundary curves (same geometry as Theory 2 but
    uses vc0 from the Weibull energy balance at the fixed dp0=120mm reference).
    Returns list of (x_arr, y_arr, label) tuples.
    """
    v_fix   = DEF["v_mps"]
    chi_fix = DEF["chi"]
    Rr_fix  = R_BIG * (1.0 + 1.0 / chi_fix)

    if pair == "v_alpha":
        vc = _vc0_weibull(chi_fix)
        if vc > x_vals.max():
            return []
        v_arr = np.linspace(vc, x_vals.max(), 400)
        a_arr = np.degrees(np.arccos(np.clip(vc / v_arr, 0, 1)))
        return [(v_arr,  a_arr, "Analytic Weibull theory"),
                (v_arr, -a_arr, None)]

    elif pair == "v_b":
        vc = _vc0_weibull(chi_fix)
        if vc > x_vals.max():
            return []
        v_arr = np.linspace(vc, x_vals.max(), 400)
        b_arr = Rr_fix * np.sqrt(np.clip(1 - (vc / v_arr)**2, 0, None))
        return [(v_arr, b_arr, "Analytic Weibull theory")]

    elif pair == "v_chi":
        chi_arr = np.linspace(y_vals.min(), y_vals.max(), 400)
        vc_arr  = np.array([_vc0_weibull(c) for c in chi_arr])
        mask = (vc_arr >= x_vals.min()) & (vc_arr <= x_vals.max())
        if not mask.any():
            return []
        return [(vc_arr[mask], chi_arr[mask], "Analytic Weibull theory")]

    elif pair == "alpha_b":
        vc = _vc0_weibull(chi_fix)
        if v_fix < vc:
            return []
        a_arr = np.linspace(x_vals.min(), x_vals.max(), 400)
        cos_a = np.abs(np.cos(np.radians(a_arr)))
        ratio = np.where(cos_a > 1e-9, vc / (v_fix * cos_a), np.inf)
        mask  = ratio <= 1.0
        b_arr = np.where(mask, Rr_fix * np.sqrt(np.clip(1 - ratio**2, 0, None)), np.nan)
        return [(a_arr, b_arr, "Analytic Weibull theory")]

    elif pair == "alpha_chi":
        chi_arr = np.linspace(x_vals.min(), x_vals.max(), 400)
        vc_arr  = np.array([_vc0_weibull(c) for c in chi_arr])
        ratio   = vc_arr / v_fix
        mask    = ratio <= 1.0
        if not mask.any():
            return []
        a_c = np.degrees(np.arccos(np.clip(ratio[mask], 0, 1)))
        return [(chi_arr[mask],  a_c, "Analytic Weibull theory"),
                (chi_arr[mask], -a_c, None)]

    elif pair == "b_chi":
        chi_arr = np.linspace(y_vals.min(), y_vals.max(), 400)
        vc_arr  = np.array([_vc0_weibull(c) for c in chi_arr])
        ratio   = vc_arr / v_fix
        mask    = ratio <= 1.0
        if not mask.any():
            return []
        bf_c = np.sqrt(np.clip(1 - ratio[mask]**2, 0, None))
        return [(bf_c, chi_arr[mask], "Analytic Weibull theory")]

    return []


def _geometry_curves(pair, x_vals, y_vals):
    """
    Return list of (x_arr, y_arr, label) tuples for the geometry-limiting
    trapping boundary.  Derived from the Coulomb cone condition:
        μ = tan(½(α + arcsin(b/(R+r))))
    which gives:
        α_c = 2·arctan(μ) − arcsin(b/(R+r))
        b_c = (R+r)·sin(2·arctan(μ) − α)

    Pairs v_chi, v_mu, chi_mu are always at α=0, b=0 → ore always trapped
    → no geometry-limiting boundary exists.
    """
    mu_fix  = DEF["mu"]
    chi_fix = DEF["chi"]
    Rr_fix  = R_BIG * (1.0 + 1.0 / chi_fix)   # R+r at default chi

    # Half-angle at default μ
    theta_mu = math.atan(mu_fix)   # θ = arctan(μ)
    ac_mu    = math.degrees(2.0 * theta_mu)   # α_c = 2·arctan(μ)  [degrees]

    if pair == "v_alpha":
        # Geometry boundary: horizontal lines at α = ±2·arctan(μ)
        # Label shows the equation
        v_full = np.array([float(x_vals.min()), float(x_vals.max())])
        label  = r"$|\alpha| = 2\arctan\mu$"
        return [(v_full, np.array([ ac_mu,  ac_mu]), label),
                (v_full, np.array([-ac_mu, -ac_mu]), None)]

    elif pair == "v_b":
        # Geometry boundary: horizontal line at b_c = (R+r)·sin(2·arctan(μ))
        bc = Rr_fix * math.sin(2.0 * theta_mu)
        v_full = np.array([float(x_vals.min()), float(x_vals.max())])
        label  = r"$b = (R{+}r)\sin(2\arctan\mu)$"
        return [(v_full, np.array([bc, bc]), label)]

    elif pair == "alpha_b":
        # Two symmetric trapping boundaries forming a tent shape:
        #   Right: b = (R+r)·sin(2arctan(μ) − α)  [upper-left → lower-right]
        #   Left:  b = (R+r)·sin(2arctan(μ) + α)  [upper-right → lower-left]
        # Both meet at apex (α=0, b_max) and touch b=0 at α=±2arctan(μ)
        a_arr = np.linspace(float(x_vals.min()), float(x_vals.max()), 400)
        a_rad = np.radians(a_arr)
        bc_left  = Rr_fix * np.sin(2.0 * theta_mu + a_rad)
        # Duplicate translated right by 2·ac so it starts at +ac instead of -ac
        bc_right = Rr_fix * np.sin(a_rad - 2.0 * theta_mu)
        # Clip each curve at the ascending portion only (stop at b_max peak, no descent)
        mask_l = (bc_left  >= 0.0) & (a_rad <= (math.pi/2.0 - 2.0*theta_mu))
        mask_r = (bc_right >= 0.0) & (a_rad <= (math.pi/2.0 + 2.0*theta_mu))
        label = r"$b = (R{+}r)\sin(2\arctan\mu \pm \alpha)$"
        curves = []
        if mask_l.any():
            curves.append((a_arr[mask_l], bc_left[mask_l], label))
        if mask_r.any():
            curves.append((a_arr[mask_r], bc_right[mask_r], None))
        return curves

    elif pair == "alpha_chi":
        # x=chi, y=alpha_deg: horizontal lines at α = ±2·arctan(μ)  (independent of χ)
        chi_full = np.array([float(x_vals.min()), float(x_vals.max())])
        label    = r"$|\alpha| = 2\arctan\mu$"
        return [(chi_full, np.array([ ac_mu,  ac_mu]), label),
                (chi_full, np.array([-ac_mu, -ac_mu]), None)]

    elif pair == "alpha_mu":
        # x=mu, y=alpha_deg: α_c(μ) = ±2·arctan(μ)  (varies with μ)
        mu_arr = np.linspace(float(x_vals.min()), float(x_vals.max()), 400)
        ac_arr = np.degrees(2.0 * np.arctan(mu_arr))
        label  = r"$|\alpha| = 2\arctan\mu$"
        return [(mu_arr,  ac_arr, label),
                (mu_arr, -ac_arr, None)]

    elif pair == "b_chi":
        # Geometry boundary: b_frac_c = sin(2·arctan(μ))  (independent of χ)
        bfc    = math.sin(2.0 * theta_mu)
        chi_full = np.array([float(y_vals.min()), float(y_vals.max())])
        label  = r"$b/(R{+}r) = \sin(2\arctan\mu)$"
        return [(np.array([bfc, bfc]), chi_full, label)]

    elif pair == "b_mu":
        # Geometry boundary: b_c(μ) = (R+r)·sin(2·arctan(μ))  (varies with μ)
        mu_arr = np.linspace(float(x_vals.min()), float(x_vals.max()), 400)
        bc_arr = Rr_fix * np.sin(2.0 * np.arctan(mu_arr))
        label  = r"$b = (R{+}r)\sin(2\arctan\mu)$"
        return [(mu_arr, bc_arr, label)]

    elif pair == "v_mu":
        # α=20° fixed: geometry boundary at μ_c = tan(|α|/2), independent of v
        a_fix = math.radians(PAIR_DEF_OVERRIDE["v_mu"]["a_deg"])
        mu_c  = math.tan(abs(a_fix) / 2.0)
        v_full = np.array([float(x_vals.min()), float(x_vals.max())])
        label  = r"$\mu = \tan(|\alpha|/2)$"
        return [(v_full, np.array([mu_c, mu_c]), label)]

    elif pair == "chi_mu":
        # α=20° fixed: geometry boundary at μ_c = tan(|α|/2), independent of χ
        a_fix  = math.radians(PAIR_DEF_OVERRIDE["chi_mu"]["a_deg"])
        mu_c   = math.tan(abs(a_fix) / 2.0)
        chi_full = np.array([float(x_vals.min()), float(x_vals.max())])
        label  = r"$\mu = \tan(|\alpha|/2)$"
        return [(chi_full, np.array([mu_c, mu_c]), label)]

    # v_chi: ore always trapped (α=0, b=0) → no boundary
    return []


PAIR_TO_COLS = {
    "v_alpha":   ("v_mps",   "a_deg"),
    "v_b":       ("v_mps",   "b_m"),
    "v_chi":     ("v_mps",   "chi"),
    "v_mu":      ("v_mps",   "mu"),
    "alpha_b":   ("a_deg",   "b_m"),
    "alpha_chi": ("chi",     "a_deg"),
    "alpha_mu":  ("mu",      "a_deg"),
    "b_chi":     ("b_frac",  "chi"),   # b normalised by b_max(chi)
    "b_mu":      ("b_m",     "mu"),
    "chi_mu":    ("chi",     "mu"),
}

AXIS_LABELS = {
    "v_mps":  r"Impact velocity $v$ (m/s)",
    "a_deg":  r"Impact angle $\alpha$ (°)",
    "b_m":    r"Impact parameter $b$ (m)",
    "b_frac": r"Normalised impact parameter $b\,/\,b_\mathrm{max}$",
    "chi":    r"Size ratio $\chi = R/r$",
    "mu":     r"Friction coefficient $\mu$",
}

# Pairs that require b/b_max normalisation (b_max = R*(1+1/chi) varies with chi)
NORMALISE_B = {"b_chi"}

# Per-pair default overrides (only non-zero entries need listing)
PAIR_DEF_OVERRIDE = {
    "chi_mu": {"a_deg": 20.0},
    "v_mu":   {"a_deg": 20.0},
}

CSV_PATH  = Path(__file__).parent.parent / "results" / "csv" / "sweep_results.csv"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots"

# ── Main ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pair = sys.argv[1].lower()
    if pair not in PAIR_TO_COLS:
        print(f"Unknown pair '{pair}'. Valid: {', '.join(PAIR_TO_COLS)}")
        sys.exit(1)

    use_weibull = "--weibull" in sys.argv
    u_col = "UW_max" if use_weibull else "U_max"
    u_label = r"$U_W$" if use_weibull else r"$U$"

    xcol, ycol = PAIR_TO_COLS[pair]

    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # For b_chi: normalise b by b_max(chi) so the grid aligns across chi values
    tol = dict(TOL)
    if pair in NORMALISE_B:
        df["b_frac"] = df["b_m"] / (R_BIG * (1.0 + 1.0 / df["chi"]))
        # Round to nearest 1/44 step to cancel float truncation from bash %.8f
        df["b_frac"] = (df["b_frac"] * 44).round() / 44
        tol["b_frac"] = 0.02
    else:
        tol = TOL

    # Filter: non-swept params must be near their defaults (with per-pair overrides)
    # If b_frac is an axis, also exclude b_m from the fixed-param filter
    exclude = set([xcol, ycol])
    if "b_frac" in exclude:
        exclude.add("b_m")
    pair_def = {**DEF, **PAIR_DEF_OVERRIDE.get(pair, {})}
    fixed = {k: v for k, v in pair_def.items() if k not in exclude}
    for col, val in fixed.items():
        if col in df.columns:
            df = df[np.abs(df[col] - val) <= tol[col]]

    if df.empty:
        print(f"No rows match defaults for fixed params: {fixed}")
        sys.exit(1)

    # Grid
    if u_col not in df.columns:
        print(f"Column '{u_col}' not found in CSV. Re-run sweep to generate it.")
        sys.exit(1)
    g = df.groupby([xcol, ycol], as_index=False)[u_col].max()
    x_vals = np.sort(g[xcol].unique())
    y_vals = np.sort(g[ycol].unique())
    Xi = {x: i for i, x in enumerate(x_vals)}
    Yi = {y: i for i, y in enumerate(y_vals)}
    Z = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, r in g.iterrows():
        Z[Yi[r[ycol]], Xi[r[xcol]]] = r[u_col]

    # Rescale Weibull values to fixed reference (dp0=120mm, sigma_pc0=5.579 MPa)
    if use_weibull:
        Z = Z * _weibull_scale_factor()

    # Mask no-collision cells
    U_plot = Z.copy()
    U_plot[np.isfinite(U_plot) & (U_plot <= 1e-15)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        Zlog = np.log10(U_plot)

    finite = Zlog[np.isfinite(Zlog)]
    if finite.size == 0:
        print("All U values are zero or NaN — nothing to plot.")
        sys.exit(0)
    span = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))), 0.1)

    norm = colors.TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=+span)
    cmap = colors.LinearSegmentedColormap.from_list(
        "gwr", ["darkgreen", "white", "firebrick"], N=256)
    cmap.set_bad("black")

    def edges(vals):
        h = np.diff(vals) / 2.0
        return np.concatenate(([vals[0]-h[0]], vals[:-1]+h, [vals[-1]+h[-1]]))

    fs = 16
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs,
                          "xtick.labelsize": fs, "ytick.labelsize": fs})
    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.pcolormesh(edges(x_vals), edges(y_vals), Zlog,
                       cmap=cmap, norm=norm, shading="auto")

    # U=1 contour from DEM (blue) — proxy artist for legend
    legend_handles = []
    u_label_dem = (r"DEM Weibull Theory ($U_W = 1$)" if use_weibull
                   else r"DEM Christensen Theory ($U = 1$)")
    if np.any(np.isfinite(Zlog)):
        try:
            ax.contour(x_vals, y_vals, Zlog, levels=[0.0],
                       colors=["royalblue"], linewidths=[2.5])
            legend_handles.append(mlines.Line2D([], [], color="royalblue",
                linewidth=2.5, label=u_label_dem))
        except Exception:
            pass

    # Analytic boundary (orange dashed)
    analytic_curves = (_weibull_theory_curves(pair, x_vals, y_vals) if use_weibull
                       else _theory2_curves(pair, x_vals, y_vals))
    analytic_label_str = ("Analytic Weibull theory" if use_weibull
                          else "Analytic Christensen theory")
    analytic_label_added = False
    for xs, ys, label in analytic_curves:
        ax.plot(xs, ys, color="darkorange", linewidth=2.5, linestyle="--")
        if label and not analytic_label_added:
            legend_handles.append(mlines.Line2D([], [], color="darkorange",
                linewidth=2.5, linestyle="--", label=analytic_label_str))
            analytic_label_added = True

    # Geometry-limiting trapping boundary (solid black)
    geo_curves = _geometry_curves(pair, x_vals, y_vals)
    geo_label_added = False
    for xs, ys, label in geo_curves:
        kw = dict(color="black", linewidth=2.0, linestyle="-")
        ax.plot(xs, ys, **kw)
        if label and not geo_label_added:
            legend_handles.append(mlines.Line2D([], [], color="black",
                linewidth=2.0, linestyle="-", label=label))
            geo_label_added = True

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=fs - 2, loc="best")

    ax.set_xlabel(AXIS_LABELS.get(xcol, xcol))
    ax.set_ylabel(AXIS_LABELS.get(ycol, ycol))
    ax.set_xlim(float(x_vals.min()), float(x_vals.max()))
    ax.set_ylim(float(y_vals.min()), float(y_vals.max()))
    ax.tick_params(axis="both", length=6, width=1.5)

    ticks = np.arange(-math.ceil(span), math.ceil(span) + 1, dtype=float)
    ticks = ticks[(ticks >= -span) & (ticks <= span)]
    if 0.0 not in ticks:
        ticks = np.sort(np.append(ticks, 0.0))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{10.0**float(t):g}" for t in ticks])
    cbar.ax.tick_params(labelsize=fs, length=6, width=1.5)
    cbar.set_label(u_label, fontsize=fs)

    fixed_str = "  |  ".join(f"{k}={v}" for k, v in fixed.items())
    ax.set_title(f"{pair}   [{fixed_str}]", fontsize=13)

    plt.tight_layout()
    subdir = "weibull" if use_weibull else "christensen"
    out_dir = PLOTS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pair}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    print(f"\nRows plotted: {len(df)}")
    print(f"x ({xcol}): {x_vals.min():.4g} → {x_vals.max():.4g}  ({len(x_vals)} pts)")
    print(f"y ({ycol}): {y_vals.min():.4g} → {y_vals.max():.4g}  ({len(y_vals)} pts)")
    u_finite = Z[np.isfinite(Z)]
    if u_finite.size:
        print(f"{u_col} range: {u_finite.min():.3g} → {u_finite.max():.3g}")

if __name__ == "__main__":
    main()
