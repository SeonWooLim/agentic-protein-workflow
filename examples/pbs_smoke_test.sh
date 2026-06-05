#!/bin/bash
# PBS Pro smoke test — verifies gmx_mpi (CUDA build) runs end-to-end on a GPU
# compute node of the iREMB pilot cluster. Self-contained: builds a tiny water
# box, runs grompp + a short GPU MD. No external input files needed.
#
# Usage (from a scratch dir, NOT home):
#   cd /scratch/<user>/gmx_smoke && qsub pbs_smoke_test.sh
#   qstat -xfu <user>        # watch state Q -> R -> F
#   cat gmx_smoke.o<jobid>    # check "SMOKE TEST DONE"
#
# Queue is nvidiaq (allocated + idle). For CPU-only test, see notes at bottom.

#PBS -N gmx_smoke
#PBS -q nvidiaq
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -m abe
#PBS -r n
#PBS -V

cd "$PBS_O_WORKDIR"
module purge
module load GROMACS/2025.02 CUDA/12.4 OPENMPI/4.1.6.GCC5.8

set -e

# 1) minimal topology (AMBER99 + TIP3P water); solvate appends the SOL count
cat > topol.top <<'EOF'
#include "amber99.ff/forcefield.itp"
#include "amber99.ff/tip3p.itp"

[ system ]
smoke water box

[ molecules ]
EOF

# 2) fill a 2.3 nm cubic box with water (spc216.gro ships with GROMACS)
gmx_mpi solvate -cs spc216.gro -box 2.3 2.3 2.3 -o conf.gro -p topol.top

# 3) a short NVT run
cat > md.mdp <<'EOF'
integrator    = md
nsteps        = 1000
dt            = 0.002
cutoff-scheme = Verlet
coulombtype   = PME
rvdw          = 0.9
rcoulomb      = 0.9
tcoupl        = V-rescale
tc-grps       = System
tau-t         = 0.1
ref-t         = 300
gen-vel       = yes
gen-temp      = 300
constraints   = h-bonds
EOF

# 4) grompp + GPU mdrun (-pme cpu avoids cuFFT issues, per project convention)
gmx_mpi grompp -f md.mdp -c conf.gro -p topol.top -o smoke.tpr -maxwarn 1
gmx_mpi mdrun -deffnm smoke -ntomp 8 -nb gpu -pme cpu

echo "=== SMOKE TEST DONE ==="
grep -iE 'On host|GPU' smoke.log | head

# --- Notes -------------------------------------------------------------------
# * gmx_mpi runs here as a single MPI rank directly. If OpenMPI complains, wrap
#   the mdrun line with the PBS launcher:  mpiexec -np 1 gmx_mpi mdrun ...
# * CPU-only test on workq: change to  #PBS -q workq  /  select=1:ncpus=8
#   and  gmx_mpi mdrun -deffnm smoke -ntomp 8 -nb cpu   (libcufft still loads
#   via the CUDA module, so the CUDA-built binary starts even without a GPU).
