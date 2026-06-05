#!/bin/bash
# Full GROMACS MD pipeline for a real protein (1N8Z = Trastuzumab Fab) on the
# iREMB PBS GPU queue. pdb2gmx -> box -> solvate -> ions -> EM -> NVT -> NPT ->
# production, then a trajectory is ready for analysis_server.analyze_trajectory.
#
# PREREQUISITE (upload via WinSCP to $PBS_O_WORKDIR, e.g. /scratch/<user>/1n8z):
#   - 1N8Z_fixed.pdb   (run prep_1n8z.py first — adds missing heavy atoms!)
#   - this script
#
# Why fixed: raw 1N8Z has incomplete side chains (e.g. LYS190 missing CG), which
# pdb2gmx cannot build. prep_1n8z.py (PDBFixer) completes them and keeps the Fab
# chains A,B. Alternatively use CHARMM-GUI to produce GROMACS inputs and run only
# the EM/NVT/NPT/MD section below.
#
# Usage:
#   python prep_1n8z.py 1N8Z.pdb 1N8Z_fixed.pdb --chains A B   # (local or cluster)
#   cd /scratch/<user>/1n8z && qsub pbs_1n8z_pipeline.sh
#   qstat -xfu <user>

#PBS -N md_1n8z
#PBS -q nvidiaq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -m abe
#PBS -r n
#PBS -V

cd "$PBS_O_WORKDIR"
module purge
module load GROMACS/2025.02 CUDA/12.4 OPENMPI/4.1.6.GCC5.8

set -e

PROD_NS=2                          # production length (ns); 2 ns = quick first run
NSTEPS=$(( PROD_NS * 500000 ))     # at dt = 0.002 ps
GPU="-nb gpu -pme cpu"             # -pme cpu avoids cuFFT issues (project convention)

# ---------------------------------------------------------------------------
# 0) input must be a PDBFixer-completed structure (missing heavy atoms added).
#    Run in a per-job subdir so concurrent/repeat submissions to the same folder
#    never clobber each other's intermediates (em.*, nvt.*, md.*, topol.top ...).
# ---------------------------------------------------------------------------
INPUT="$PBS_O_WORKDIR/${INPUT_PDB:-1N8Z_fixed.pdb}"
[ -f "$INPUT" ] || { echo "ERROR: $INPUT not found — run prep_1n8z.py first"; exit 1; }
RUNDIR="$PBS_O_WORKDIR/run_${PBS_JOBID%%.*}"
mkdir -p "$RUNDIR" && cd "$RUNDIR"
echo "Running in $RUNDIR"

# ---------------------------------------------------------------------------
# 1) topology (AMBER99SB-ILDN + TIP3P), 2) box, 3) solvate
# ---------------------------------------------------------------------------
gmx_mpi pdb2gmx -f "$INPUT" -ff amber99sb-ildn -water tip3p \
    -o processed.gro -p topol.top -i posre.itp -ignh
gmx_mpi editconf -f processed.gro -o box.gro -c -d 1.0 -bt cubic
gmx_mpi solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top

# ---------------------------------------------------------------------------
# 4) add ions: neutralise + 0.15 M NaCl
# ---------------------------------------------------------------------------
cat > ions.mdp <<'EOF'
integrator    = steep
nsteps        = 0
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 1.0
rcoulomb      = 1.0
EOF
gmx_mpi grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo SOL | gmx_mpi genion -s ions.tpr -o solv_ions.gro -p topol.top \
    -pname NA -nname CL -neutral -conc 0.15

# ---------------------------------------------------------------------------
# 5) energy minimisation
# ---------------------------------------------------------------------------
cat > em.mdp <<'EOF'
integrator    = steep
emtol         = 1000.0
emstep        = 0.01
nsteps        = 50000
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 1.0
rcoulomb      = 1.0
EOF
gmx_mpi grompp -f em.mdp -c solv_ions.gro -p topol.top -o em.tpr
gmx_mpi mdrun -deffnm em -ntomp 8 $GPU

# ---------------------------------------------------------------------------
# 6) NVT equilibration (position-restrained, 100 ps)
# ---------------------------------------------------------------------------
cat > nvt.mdp <<'EOF'
define        = -DPOSRES
integrator    = md
nsteps        = 50000
dt            = 0.002
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 1.0
rcoulomb      = 1.0
constraints   = h-bonds
tcoupl        = V-rescale
tc-grps       = Protein Non-Protein
tau-t         = 0.1 0.1
ref-t         = 300 300
pcoupl        = no
gen-vel       = yes
gen-temp      = 300
EOF
gmx_mpi grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 1
gmx_mpi mdrun -deffnm nvt -ntomp 8 $GPU

# ---------------------------------------------------------------------------
# 7) NPT equilibration (position-restrained, 100 ps)
# ---------------------------------------------------------------------------
cat > npt.mdp <<'EOF'
define        = -DPOSRES
integrator    = md
nsteps        = 50000
dt            = 0.002
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 1.0
rcoulomb      = 1.0
constraints   = h-bonds
tcoupl        = V-rescale
tc-grps       = Protein Non-Protein
tau-t         = 0.1 0.1
ref-t         = 300 300
pcoupl        = C-rescale
pcoupltype    = isotropic
tau-p         = 2.0
ref-p         = 1.0
compressibility = 4.5e-5
EOF
gmx_mpi grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt \
    -p topol.top -o npt.tpr -maxwarn 1
gmx_mpi mdrun -deffnm npt -ntomp 8 $GPU

# ---------------------------------------------------------------------------
# 8) production MD (no restraints)
# ---------------------------------------------------------------------------
cat > md.mdp <<EOF
integrator    = md
nsteps        = ${NSTEPS}
dt            = 0.002
nstxout-compressed = 5000
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 1.0
rcoulomb      = 1.0
constraints   = h-bonds
tcoupl        = V-rescale
tc-grps       = Protein Non-Protein
tau-t         = 0.1 0.1
ref-t         = 300 300
pcoupl        = Parrinello-Rahman
pcoupltype    = isotropic
tau-p         = 2.0
ref-p         = 1.0
compressibility = 4.5e-5
EOF
gmx_mpi grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1
gmx_mpi mdrun -deffnm md -ntomp 8 $GPU

echo "=== 1N8Z PIPELINE DONE (${PROD_NS} ns) ==="
echo "trajectory: md.xtc   topology: md.tpr   -> analyze_trajectory"
