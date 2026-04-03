#!/usr/bin/env bash
# ============================================================
# pairwise_sweep/sweep.sh
# Runs one pairwise parameter sweep, analyses each run,
# deletes dumps, appends results to a shared CSV.
#
# Usage:
#   bash sweep.sh <PAIR> <OUTPUT_ROOT>
#
# PAIR choices:
#   v_alpha  v_b  v_chi  v_mu
#   alpha_b  alpha_chi  alpha_mu
#   b_chi    b_mu
#   chi_mu
#
# Example:
#   bash sweep.sh v_alpha ../results/sweep_v_alpha
# ============================================================
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAMMPS_EXE="/Users/danielcui/Downloads/lammps-12Jun2025/build/lmp"
IN_TEMPLATE="$SCRIPT_DIR/../LAMMPS_model/sync_sweep.in"
ANALYSE_PY="$SCRIPT_DIR/analyse_run.py"
CSV_OUT="$SCRIPT_DIR/../results/csv/sweep_results.csv"

# ── Fixed geometry ────────────────────────────────────────────
R_BIG=0.025   # m
GAP=0.010     # m gap between mover and ore

# ── Default parameter values ──────────────────────────────────
: "${DEF_V:=3.0}"
: "${DEF_ALPHA:=0}"
: "${DEF_B:=0.0}"
: "${DEF_CHI:=4}"
: "${DEF_MU:=0.47}"

# ── Grid arrays ───────────────────────────────────────────────
# v: 0.5 to 5.0 in 0.1 steps → 46 values
V_VALS=($(seq 0.5 0.1 5.0))
# alpha: -90 to 90 in 4 deg steps → 46 values
ALPHA_VALS=($(seq -90 4 90))
# mu: 0 to 1 in 0.025 steps → 41 values
MU_VALS=($(seq 0 0.025 1.0))
# chi: 2 to 20 in 0.4 steps → 46 values
CHI_VALS=($(seq 2 0.4 20))
# b: 0..B_N (fractions of b_max); actual b computed per run
B_N=44
B_IDXS=($(seq 0 1 $B_N))   # 45 values: 0/44 .. 44/44 of b_max

# ── Parallelism ───────────────────────────────────────────────
: "${MAX_PARALLEL:=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}"
: "${NICE:=10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
job_count() { jobs -pr | wc -l | tr -d ' '; }
gate()      { while [[ $(job_count) -ge $MAX_PARALLEL ]]; do sleep 0.1; done; }

# ── Label helper (float → folder-safe string) ─────────────────
lbl() {
    awk -v x="$1" 'BEGIN{
        fmt = (x == int(x)) ? "%.0f" : "%.6g"
        printf fmt, x
    }' | sed 's/-/m/; s/\./p/g'
}

# ── Per-run launcher ──────────────────────────────────────────
run_sim() {
    local v="$1" alpha="$2" b="$3" chi="$4" mu="$5"

    local R_SMALL sum_r D rad C2x C2y vx vy nsteps mu11
    # mu_eff = sqrt(mu11 * 0.3)  →  mu11 = mu_eff^2 / 0.3
    mu11=$(awk -v m="$mu" 'BEGIN{printf "%.6f", m*m/0.3}')
    R_SMALL=$(awk -v R="$R_BIG" -v c="$chi"  'BEGIN{printf "%.8f", R/c}')
    sum_r=$(  awk -v R="$R_BIG" -v r="$R_SMALL" 'BEGIN{printf "%.8f", R+r}')
    D=$(      awk -v sr="$sum_r" -v g="$GAP"     'BEGIN{printf "%.8f", sr+g}')
    rad=$(    awk -v a="$alpha"  'BEGIN{printf "%.8f", a*atan2(0,-1)/180}')

    C2x=$(awk -v sr="$sum_r" -v D="$D" -v r="$rad" -v b="$b" \
              'BEGIN{printf "%.8f", sr + D*cos(r) + b*sin(r)}')
    C2y=$(awk -v D="$D" -v r="$rad" -v b="$b" \
              'BEGIN{printf "%.8f", -D*sin(r) + b*cos(r)}')
    vx=$( awk -v v="$v" -v r="$rad" 'BEGIN{printf "%.8f", -v*cos(r)}')
    vy=$( awk -v v="$v" -v r="$rad" 'BEGIN{printf "%.8f",  v*sin(r)}')

    nsteps=$(awk -v C2x="$C2x" -v C2y="$C2y" -v ore_x="$sum_r" \
                 -v vx="$vx" -v vy="$vy" -v sr="$sum_r" 'BEGIN{
        dx = C2x - ore_x
        dy = C2y
        A  = vx*vx + vy*vy
        B  = 2*(dx*vx + dy*vy)
        C  = dx*dx + dy*dy - sr*sr
        disc = B*B - 4*A*C
        if (disc < 0) { print 5000; exit }
        T = (-B - sqrt(disc)) / (2*A)
        steps = int(T / 1e-7) + 5000
        print (steps < 5000) ? 5000 : steps
    }')

    local run_dir
    run_dir="$OUTROOT/run_chi$(lbl "$chi")_v$(lbl "$v")_b$(lbl "$b")_a$(lbl "$alpha")_mu$(lbl "$mu")"
    mkdir -p "$run_dir"

    # Write 3ps.dat
    cat > "$run_dir/3ps.dat" <<EOF
LAMMPS data file

3 atoms
2 atom types

-0.12 0.12 xlo xhi
-0.12 0.12 ylo yhi
-0.12 0.12 zlo zhi

Atoms # id type diameter density x y z omegax omegay omegaz

1 2 $(awk -v r="$R_SMALL" 'BEGIN{printf "%.8f", 2*r}') 2600 $sum_r 0.0 0.0 0 0 0
2 1 $(awk -v R="$R_BIG"   'BEGIN{printf "%.8f", 2*R}') 7800 $C2x $C2y 0.0 0 0 0
3 1 $(awk -v R="$R_BIG"   'BEGIN{printf "%.8f", 2*R}') 7800 0.0 0.0 0.0 0 0 0

Velocities

1 0 0 0 0 0 0
2 $vx $vy 0 0 0 0
3 0 0 0 0 0 0
EOF

    cp "$IN_TEMPLATE" "$run_dir/sync_sweep.in"

    (
        cd "$run_dir" || exit 1
        nice -n "$NICE" "$LAMMPS_EXE" \
            -in sync_sweep.in \
            -v nsteps "$nsteps" \
            -v mu11 "$mu11" \
            -log none \
            > lammps.log 2>&1
        python3 "$ANALYSE_PY" . "$v" "$alpha" "$b" "$chi" "$mu" "$CSV_OUT"
        cd /tmp && rm -rf "$run_dir"
    ) &
}

# ── Argument parsing ──────────────────────────────────────────
if [[ $# -lt 2 ]]; then
    echo "Usage: $(basename "$0") <PAIR> <OUTPUT_ROOT>"
    echo "PAIR: v_alpha | v_b | v_chi | v_mu | alpha_b | alpha_chi | alpha_mu | b_chi | b_mu | chi_mu"
    exit 1
fi
PAIR="$1"
OUTROOT="$2"
mkdir -p "$OUTROOT"

echo "======== pairwise sweep: $PAIR ========"
echo "Output: $OUTROOT"
echo "CSV:    $CSV_OUT"
echo "Cores:  $MAX_PARALLEL"
echo "========================================"

# ── Sweep loops ───────────────────────────────────────────────
case "$PAIR" in

  v_alpha)
    for v in "${V_VALS[@]}"; do
        for alpha in "${ALPHA_VALS[@]}"; do
            gate; run_sim "$v" "$alpha" "$DEF_B" "$DEF_CHI" "$DEF_MU"
        done
    done ;;

  v_b)
    b_max=$(awk -v R="$R_BIG" -v c="$DEF_CHI" 'BEGIN{printf "%.8f", R + R/c}')
    for v in "${V_VALS[@]}"; do
        for bi in "${B_IDXS[@]}"; do
            b=$(awk -v i="$bi" -v bm="$b_max" -v n="$B_N" 'BEGIN{printf "%.8f", i*bm/n}')
            gate; run_sim "$v" "$DEF_ALPHA" "$b" "$DEF_CHI" "$DEF_MU"
        done
    done ;;

  v_chi)
    for v in "${V_VALS[@]}"; do
        for chi in "${CHI_VALS[@]}"; do
            gate; run_sim "$v" "$DEF_ALPHA" "$DEF_B" "$chi" "$DEF_MU"
        done
    done ;;

  v_mu)
    for v in "${V_VALS[@]}"; do
        for mu in "${MU_VALS[@]}"; do
            gate; run_sim "$v" "$DEF_ALPHA" "$DEF_B" "$DEF_CHI" "$mu"
        done
    done ;;

  alpha_b)
    b_max=$(awk -v R="$R_BIG" -v c="$DEF_CHI" 'BEGIN{printf "%.8f", R + R/c}')
    for alpha in "${ALPHA_VALS[@]}"; do
        for bi in "${B_IDXS[@]}"; do
            b=$(awk -v i="$bi" -v bm="$b_max" -v n="$B_N" 'BEGIN{printf "%.8f", i*bm/n}')
            gate; run_sim "$DEF_V" "$alpha" "$b" "$DEF_CHI" "$DEF_MU"
        done
    done ;;

  alpha_chi)
    for alpha in "${ALPHA_VALS[@]}"; do
        for chi in "${CHI_VALS[@]}"; do
            gate; run_sim "$DEF_V" "$alpha" "$DEF_B" "$chi" "$DEF_MU"
        done
    done ;;

  alpha_mu)
    for alpha in "${ALPHA_VALS[@]}"; do
        for mu in "${MU_VALS[@]}"; do
            gate; run_sim "$DEF_V" "$alpha" "$DEF_B" "$DEF_CHI" "$mu"
        done
    done ;;

  b_chi)
    for chi in "${CHI_VALS[@]}"; do
        b_max=$(awk -v R="$R_BIG" -v c="$chi" 'BEGIN{printf "%.8f", R + R/c}')
        for bi in "${B_IDXS[@]}"; do
            b=$(awk -v i="$bi" -v bm="$b_max" -v n="$B_N" 'BEGIN{printf "%.8f", i*bm/n}')
            gate; run_sim "$DEF_V" "$DEF_ALPHA" "$b" "$chi" "$DEF_MU"
        done
    done ;;

  b_mu)
    b_max=$(awk -v R="$R_BIG" -v c="$DEF_CHI" 'BEGIN{printf "%.8f", R + R/c}')
    for mu in "${MU_VALS[@]}"; do
        for bi in "${B_IDXS[@]}"; do
            b=$(awk -v i="$bi" -v bm="$b_max" -v n="$B_N" 'BEGIN{printf "%.8f", i*bm/n}')
            gate; run_sim "$DEF_V" "$DEF_ALPHA" "$b" "$DEF_CHI" "$mu"
        done
    done ;;

  chi_mu)
    for chi in "${CHI_VALS[@]}"; do
        for mu in "${MU_VALS[@]}"; do
            gate; run_sim "$DEF_V" "$DEF_ALPHA" "$DEF_B" "$chi" "$mu"
        done
    done ;;

  *)
    echo "Unknown pair: $PAIR"
    echo "Valid: v_alpha v_b v_chi v_mu alpha_b alpha_chi alpha_mu b_chi b_mu chi_mu"
    exit 1 ;;
esac

wait
echo "✅ Sweep '$PAIR' complete → $OUTROOT"
echo "   Results appended to: $CSV_OUT"
