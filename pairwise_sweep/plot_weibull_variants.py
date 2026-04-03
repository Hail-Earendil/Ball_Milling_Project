#!/usr/bin/env python3
"""
plot_weibull_variants.py — generate v_alpha Weibull heatmaps for each
reference (dp0, sigma_pc0) pair from the limestone crushing Table 1.

All variants are computed by rescaling the UW_max column already in the CSV:
    U_W_new = U_W_csv * (sigma_pc0_csv * dp0_csv^(3/m)) / (sigma_pc0_new * dp0_new^(3/m))

No LAMMPS re-runs needed.

Overlay: analytic Weibull fracture boundary from the energy balance
    ½ M_proj (v cos α)² = Vp · Epc(dp)
using fixed reference dp0=120 mm, sigma_pc0=5.579 MPa (Eq. 22 of Theory 2 doc).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# ── Weibull modulus ───────────────────────────────────────────────────
W_MOD = 2.73

# ── Reference pair used when generating the CSV ───────────────────────
CSV_W_D0   = 0.029
CSV_W_SIG0 = 12.858e6

# ── All limestone Table 1 reference pairs ─────────────────────────────
REFERENCE_PAIRS = [
    (0.0125, 19.162e6),
    (0.0165, 15.964e6),
    (0.0190, 14.880e6),
    (0.0210, 14.081e6),
    (0.0230, 13.312e6),
    (0.0250, 13.150e6),
    (0.0270, 12.988e6),
    (0.0290, 12.858e6),
    (0.0310, 12.622e6),
    (0.0330, 11.738e6),
    (0.0350, 11.241e6),
    (0.0370, 10.820e6),
    (0.0390, 10.111e6),
    (0.0590,  6.692e6),
    (0.1200,  5.579e6),
    (0.2400,  3.795e6),
]

# ── Analytic overlay: fixed reference pair ────────────────────────────
OVERLAY_DP0  = 0.1200       # m
OVERLAY_SIG0 = 5.579e6      # Pa

# ── Material / geometry constants ─────────────────────────────────────
R_BIG      = 0.025          # m  (grinding ball radius)
CHI_FIX    = 4.0            # default size ratio
RHO_MOVER  = 7800.0         # kg/m³ (steel)
E_ORE,  NU_ORE  = 7.782e10, 0.28
E_BALL, NU_BALL = 2.08e11,  0.27

# ── Plot settings (matching plot_sweep.py) ────────────────────────────
DEF = {"v_mps": 3.0, "a_deg": 0.0, "b_m": 0.0, "chi": 4.0, "mu": 0.47}
TOL = {"v_mps": 0.05, "a_deg": 0.5, "b_m": 1e-5, "chi": 0.1, "mu": 0.01}

CSV_PATH  = Path(__file__).parent.parent / "results" / "csv" / "sweep_results.csv"
PLOTS_DIR = Path(__file__).parent.parent / "results" / "plots" / "weibull" / "variants"


# ── Weibull-energy critical velocity (head-on) ────────────────────────
def _vc0_weibull(chi, dp0, sig0):
    """
    Head-on critical velocity from Weibull energy theory (Eq. 22).
        vc0 = sqrt(2 * Vp * Epc(dp) / M_proj)
    where Epc is from Eq. 22 and dp = 2*RS = 2*RB/chi.
    """
    RS     = R_BIG / chi
    dp     = 2.0 * RS
    R_star = 1.0 / (1.0/RS + 1.0/R_BIG)
    E_star = 1.0 / ((1.0 - NU_ORE**2)/E_ORE + (1.0 - NU_BALL**2)/E_BALL)

    # Eq. 22: Epc(dp) = (4.989/pi) * E*^(-2/3) * (RS/R*)^(1/3)
    #                   * sig0^(5/3) * dp0^(5/w) * dp^(-5/w)
    Epc = (4.989 / math.pi) \
          * E_star**(-2.0/3.0) \
          * (RS / R_star)**(1.0/3.0) \
          * sig0**(5.0/3.0) \
          * dp0**(5.0/W_MOD) \
          * dp**(-5.0/W_MOD)

    Vp     = (4.0/3.0) * math.pi * RS**3
    M_proj = (4.0/3.0) * math.pi * R_BIG**3 * RHO_MOVER

    return math.sqrt(2.0 * Vp * Epc / M_proj)


def _scale_factor(dp0_new, sig0_new):
    """Multiplier to convert CSV UW_max → UW_max for new reference pair."""
    exp = 3.0 / W_MOD
    csv_ff = CSV_W_SIG0 * (CSV_W_D0 ** exp)
    new_ff = sig0_new   * (dp0_new  ** exp)
    return csv_ff / new_ff


def plot_variant(x_vals, y_vals, Z_base, dp0, sig0, out_path):
    scale   = _scale_factor(dp0, sig0)
    Z_scaled = Z_base * scale

    U_plot = Z_scaled.copy()
    U_plot[np.isfinite(U_plot) & (U_plot <= 1e-15)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        Zlog = np.log10(U_plot)

    finite = Zlog[np.isfinite(Zlog)]
    if finite.size == 0:
        print(f"  [skip] all zeros for dp0={dp0*1000:.1f}mm")
        return
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

    # DEM U_W = 1 contour (blue) — add proxy artist for legend
    legend_handles = []
    import matplotlib.lines as mlines
    if np.any(np.isfinite(Zlog)):
        try:
            ax.contour(x_vals, y_vals, Zlog, levels=[0.0],
                       colors=["royalblue"], linewidths=[2.5])
            legend_handles.append(mlines.Line2D([], [], color="royalblue",
                linewidth=2.5, label=r"DEM Weibull Theory ($U_W = 1$)"))
        except Exception:
            pass

    # Analytic Weibull boundary: v_crit(α) = vc0 / |cos α|  (orange dashed)
    vc = _vc0_weibull(CHI_FIX, OVERLAY_DP0, OVERLAY_SIG0)
    v_max = float(x_vals.max())
    if vc <= v_max:
        v_arr = np.linspace(vc, v_max, 400)
        a_arr = np.degrees(np.arccos(np.clip(vc / v_arr, 0.0, 1.0)))
        ax.plot(v_arr,  a_arr, color="darkorange", linewidth=2.5, linestyle="--")
        ax.plot(v_arr, -a_arr, color="darkorange", linewidth=2.5, linestyle="--")
        legend_handles.append(mlines.Line2D([], [], color="darkorange",
            linewidth=2.5, linestyle="--",
            label="Analytic Weibull theory"))

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=fs - 2, loc="upper right")

    ax.set_xlabel(r"Impact velocity $v$ (m/s)")
    ax.set_ylabel(r"Impact angle $\alpha$ (°)")
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
    cbar.set_label(r"$U_W$", fontsize=fs)

    ax.set_title(
        rf"Weibull $v$–$\alpha$  |  $d_{{p0}}={dp0*1000:.2f}$ mm,  "
        rf"$\sigma_{{pc,0}}={sig0/1e6:.3f}$ MPa",
        fontsize=13)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if "UW_max" not in df.columns:
        print("UW_max column not found — re-run sweep first.")
        sys.exit(1)

    # Filter to v_alpha defaults (b=0, chi=4, mu=0.47)
    fixed = {k: v for k, v in DEF.items() if k not in ("v_mps", "a_deg")}
    for col, val in fixed.items():
        if col in df.columns:
            df = df[np.abs(df[col] - val) <= TOL[col]]

    if df.empty:
        print("No rows match defaults.")
        sys.exit(1)

    g = df.groupby(["v_mps", "a_deg"], as_index=False)["UW_max"].max()
    x_vals = np.sort(g["v_mps"].unique())
    y_vals = np.sort(g["a_deg"].unique())
    Xi = {x: i for i, x in enumerate(x_vals)}
    Yi = {y: i for i, y in enumerate(y_vals)}
    Z_base = np.full((len(y_vals), len(x_vals)), np.nan)
    for _, r in g.iterrows():
        Z_base[Yi[r["a_deg"]], Xi[r["v_mps"]]] = r["UW_max"]

    vc = _vc0_weibull(CHI_FIX, OVERLAY_DP0, OVERLAY_SIG0)
    print(f"Analytic vc0 (dp0={OVERLAY_DP0*1000:.0f}mm, "
          f"sig0={OVERLAY_SIG0/1e6:.3f}MPa, chi={CHI_FIX}): {vc:.3f} m/s")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(REFERENCE_PAIRS)} plots → {PLOTS_DIR}")

    for dp0, sig0 in REFERENCE_PAIRS:
        fname = f"v_alpha_weibull_dp{int(round(dp0*1000)):04d}mm.png"
        plot_variant(x_vals, y_vals, Z_base, dp0, sig0, PLOTS_DIR / fname)

    print("Done.")


if __name__ == "__main__":
    main()
