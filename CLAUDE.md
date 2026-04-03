# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DEM (Discrete Element Method) simulation study of ore fracture in ball milling. A three-sphere collision system is simulated in LAMMPS: a fixed large grinding ball (atom 3), a small brittle ore particle (atom 1), and an incoming large grinding ball (atom 2). Fracture is predicted using two criteria:

- **Christensen criterion** — virial stress tensor → utilisation index U; fracture when U ≥ 1
- **Weibull criterion** — contact forces → effective compression force → U_W = F_eff / F_{f,W}; fracture when U_W ≥ 1

The five sweep parameters: impact velocity `v`, impact angle `alpha`, impact parameter `b`, size ratio `chi = R/r`, friction coefficient `mu`.

## Key Commands

**Run a pairwise parameter sweep:**
```bash
bash pairwise_sweep/sweep.sh v_alpha results/sweep_v_alpha
```
Valid PAIR values: `v_alpha v_b v_chi v_mu alpha_b alpha_chi alpha_mu b_chi b_mu chi_mu`

Each sweep: generates geometry → runs LAMMPS in parallel → analyses U and U_W → deletes entire run directory → appends one row to `results/csv/sweep_results.csv`.

Override default fixed parameters via env vars before running:
```bash
DEF_ALPHA=20 bash pairwise_sweep/sweep.sh chi_mu results/sweep_chi_mu_a20
```
Available overrides: `DEF_V`, `DEF_ALPHA`, `DEF_B`, `DEF_CHI`, `DEF_MU`.

**Plot Christensen heatmaps:**
```bash
python3 pairwise_sweep/plot_sweep.py v_alpha
```
Saves to `results/plots/christensen/<pair>.png`.

**Plot Weibull heatmaps** (rescaled to dp0=120 mm, σ_pc0=5.579 MPa reference):
```bash
python3 pairwise_sweep/plot_sweep.py v_alpha --weibull
```
Saves to `results/plots/weibull/<pair>.png`.

**Plot Weibull v_alpha sensitivity across all 16 reference pairs** (appendix figure):
```bash
python3 pairwise_sweep/plot_weibull_variants.py
```
Saves 16 PNGs to `results/plots/weibull/variants/`.

**Run dump-interval sensitivity study** (timestep convergence):
```bash
python3 pairwise_sweep/dump_interval_study.py
```
Runs 3 LAMMPS sims at dump_every=1, subsamples at N=1…400 in Python, saves plot to `results/plots/dump_study/`. Reuse existing dumps unless `--rerun` is passed.

**Analyse a single run directory:**
```bash
python3 pairwise_sweep/analyse_run.py <run_dir> <v> <alpha_deg> <b_m> <chi> <mu> results/csv/sweep_results.csv
```

**Manual single LAMMPS run:**
```bash
cd <run_dir>
/Users/danielcui/Downloads/lammps-12Jun2025/build/lmp -in sync_sweep.in -v nsteps 50000 -v mu11 0.75 -log none
```

## Repository Structure

```
LAMMPS_model/
  sync_sweep.in      # sweep template: dump interval 100, uses ${nsteps} and ${mu11}
  dump_study.in      # dump-study template: configurable ${dump_every}
  3ps.dat            # overwritten per run by sweep.sh

pairwise_sweep/
  sweep.sh              # parallelised sweep launcher; deletes entire run dir after analysis
  analyse_run.py        # computes Christensen U and Weibull U_W, writes one CSV row
  plot_sweep.py         # heatmaps with analytic overlays; --weibull flag for Weibull mode
  plot_weibull_variants.py  # v_alpha Weibull heatmaps for all 16 limestone reference pairs
  dump_interval_study.py    # timestep sensitivity: run at de=1, subsample in Python

results/
  csv/
    sweep_results.csv           # all sweep data (U_max, UW_max per run)
  plots/
    christensen/                # Christensen heatmaps (one per pair)
    weibull/                    # Weibull heatmaps at dp0=120mm reference
      variants/                 # sensitivity: v_alpha for all 16 reference pairs
    dump_study/                 # timestep sensitivity plot + de1_runs/ (dumps kept)

equations/           # LaTeX derivation of Christensen U and Weibull U_W
```

## Simulation Pipeline Details

**3ps.dat geometry:** Atom 3 (fixed ball) at origin. Atom 1 (ore) at x = R_big + R_small. Atom 2 (mover) placed at GAP=0.010 m from ore surface, offset by `b`, approaching at angle `alpha`.

**nsteps (exact quadratic):** Solves `|C₂ + v·T − x_ore|² = (R+r)²` for the first contact time T. `nsteps = ⌊T/dt⌋ + 5000`. **Do not use a linear approximation** — it underestimates travel time for large `b` and the simulation ends before contact.

**Contact model:** `pair_style granular` with `hertz/material` + `mindlin`. Only `pair_coeff 1 1` and `2 2` are set; LAMMPS auto-mixes for 1-2. **Do not add an explicit `pair_coeff 1 2`** — it shifts the U=1 threshold. For mu sweeps, `mu11 = mu_eff² / 0.3` is passed via `-v mu11`; LAMMPS computes `mu_eff = sqrt(mu11 × 0.3)`. At default `mu_eff = 0.47`: `mu11 = 0.75`.

**Branch-length correction:** Virial stress is scaled by `λ = 2/(1+χ)` before computing U. At χ=4: λ=0.4.

**Weibull rescaling:** `analyse_run.py` computes U_W using reference pair (dp0=29 mm, σ_pc0=12.858 MPa). `plot_sweep.py --weibull` rescales to the chosen reference (dp0=120 mm, σ_pc0=5.579 MPa) via `scale = (σ_csv × dp0_csv^(3/m)) / (σ_new × dp0_new^(3/m))`, m=2.73.

**U=1 reference:** At α=0, b=0, χ=4, μ=0.47 → threshold v ≈ 2.5 m/s. Use to sanity-check any pipeline changes.

**Per-pair alpha overrides:** `chi_mu` and `v_mu` sweeps use α=20° (not 0°) to make the geometry-limiting boundary visible. This is set via `PAIR_DEF_OVERRIDE` in `plot_sweep.py` and must also be set via `DEF_ALPHA=20` when running those sweeps.

## Default Parameter Values

| Parameter | Default | Sweep range | Grid points |
|-----------|---------|-------------|-------------|
| v (m/s) | 3.0 | 0.5 → 5.0 | 46 (step 0.1) |
| alpha (°) | 0 | -90 → 90 | 46 (step 4°) |
| b (m) | 0 | 0 → R+r | 45 (44 intervals) |
| chi | 4 | 2 → 20 | 46 (step 0.4) |
| mu | 0.47 | 0 → 1 | 41 (step 0.025) |

b_max = R_big + R_small(chi). At b = b_max the mover grazes the ore with zero overlap.

## Analytic Fracture Boundaries (overlaid on heatmaps)

Both boundaries are implemented in `plot_sweep.py` and drawn on every heatmap.

### Geometry-limiting (solid black) — trapping condition, independent of v

Ore escapes if the Coulomb friction cone cannot prevent sideways ejection:
```
α_c = ±2·arctan(μ)                              [v_alpha, alpha_chi, alpha_mu]
b_c = (R+r)·sin(2·arctan(μ))                    [v_b, b_mu]
α_c(b) = ±[2·arctan(μ) ∓ arcsin(b/(R+r))]      [alpha_b — two arch curves]
```
At μ=0.47: α_c ≈ ±50.75°. For chi_mu and v_mu (at α=20°): μ_c = tan(10°) ≈ 0.176.

### Velocity-limiting (orange dashed) — analytic Christensen / Weibull energy threshold

Christensen (Theory 2): derived from σ_C and Hertz contact mechanics:
```
v_c(α) = v_c0(χ) / |cos α|
v_c(b) = v_c0(χ) / √(1 − (b/(R+r))²)
```
Weibull (Eq. 22): from energy balance ½M_proj(v cos α)² = V_p · E_pc(dp):
```
v_c0_W(χ) = sqrt(2 V_p E_pc / M_proj)
```
where E_pc uses the dp0=120mm, σ_pc0=5.579 MPa reference.

### Heatmap colour meaning
- **Black**: no contact (b or |α| beyond geometry limit, or nsteps too small)
- **Green**: contact, U < 1 (no fracture)
- **Red**: U > 1 (fracture predicted)
- **Blue contour**: DEM U=1 boundary
- **Orange dashed**: analytic theory boundary

## LAMMPS Reference

Granular contact: https://docs.lammps.org/pair_granular.html

## VMD Visualisation

```tcl
color Display Background white
menu tkcon on
display projection Orthographic
display depthcue off
color Display FPS black
color Axes Labels black
topo readlammpsdata 3ps.dat full
[atomselect top "all"] set radius {0.00625 0.025 0.025}
[atomselect top "all"] set type {2 1 1}
mol addfile traj1.dcd waitfor all 0
```
VMD frame = timestep / 100.
