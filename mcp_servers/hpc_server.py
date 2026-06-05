"""MCP server: pluggable MD job backend (on-prem HPC Slurm | Colab/local).

The SAME `submit_md_job` tool drives two interchangeable execution backends:

    backend="hpc"    -> Slurm `sbatch` on an on-prem cluster  (operation story)
    backend="colab"  -> local/Colab `gmx mdrun` subprocess     (one-click demo)

This abstraction is the point of the project: an agent orchestrates async MD
jobs without caring where they run, demonstrating both on-prem GPU operation
and reproducible cloud execution behind one interface. Cluster paths and
credentials come from environment variables and are never hard-coded.

Run standalone:
    python mcp_servers/hpc_server.py
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKDIR = Path(os.environ.get("APW_WORKDIR", "./work")).resolve()
STATE = WORKDIR / "jobs.json"

# --- on-prem cluster config (masked via env; see README) -------------------
GMX_SIF = os.environ.get("APW_GMX_SIF", "gmx_2024.sif")
SLURM_PARTITION = os.environ.get("APW_SLURM_PARTITION", "l40sq")
SLURM_NODE = os.environ.get("APW_SLURM_NODE", "")  # e.g. iREMB-C-08

# --- PBS Pro backend (iREMB pilot cluster — currently idle nodes) -----------
# <user> allocation only: workq (CPU), P100q / nvidiaq (GPU). Do NOT use others.
ALLOWED_PBS_QUEUES = {"workq", "P100q", "nvidiaq"}
PBS_QUEUE = os.environ.get("APW_PBS_QUEUE", "workq")
PBS_NCPUS = os.environ.get("APW_PBS_NCPUS", "8")
PBS_NGPUS = os.environ.get("APW_PBS_NGPUS", "0")            # 0 = CPU-only MD
# GROMACS comes from the system module (confirmed available on iREMB), not conda
PBS_GMX_MODULE = os.environ.get("APW_PBS_GMX_MODULE", "GROMACS/2025.02")
# binary name may be `gmx` or `gmx_mpi` depending on the build — verify with `ls`
PBS_GMX_BIN = os.environ.get("APW_PBS_GMX_BIN", "gmx")
# MPI builds (gmx_mpi) may need a launcher; on iREMB use PBS "mpiexec"; "" = direct
PBS_MPI_LAUNCH = os.environ.get("APW_PBS_MPI_LAUNCH", "")
# gmx_mpi is a CUDA + OpenMPI build → load CUDA + MPI runtime modules on the
# compute node, space-separated, e.g. "CUDA/12.4 OPENMPI/4.1.6.GCC5.8"
PBS_EXTRA_MODULES = os.environ.get("APW_PBS_EXTRA_MODULES", "")

# --- remote command channel (path B): run qsub/qstat ON the cluster via plink --
# e.g. APW_REMOTE_CMD='plink -batch -load iremb_pbs'  (reuses a saved PuTTY
# session + Pageant key — no new passwordless SSH needed). When set, scheduler
# commands execute on the cluster; input files are still staged via WinSCP.
REMOTE_CMD = os.environ.get("APW_REMOTE_CMD", "")
REMOTE_WORKDIR = os.environ.get("APW_REMOTE_WORKDIR", "/scratch/<user>")
# file staging over the same Pageant key (pscp from the PuTTY suite) — removes
# the last manual WinSCP step so stage+submit+poll all run from the laptop.
SCP_CMD = os.environ.get("APW_SCP_CMD", "")             # e.g. "pscp -batch -P 60026"
REMOTE_TARGET = os.environ.get("APW_REMOTE_TARGET", "")  # e.g. "<user>@<pbs-host>"
# Slurm cluster (iREMB-6) is a SEPARATE host → its own remote channel
SLURM_REMOTE_CMD = os.environ.get("APW_SLURM_REMOTE_CMD", "")  # e.g. "ssh -p 60026 -o BatchMode=yes <user>@<slurm-host>"
SLURM_REMOTE_WORKDIR = os.environ.get("APW_SLURM_REMOTE_WORKDIR", "/scratch/<user>")

mcp = FastMCP("hpc")


def _load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(jobs: dict) -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(jobs, indent=2))


def _remote_run(remote_cmd: str, command: str, *, input_text: str | None = None):
    """Run `command` locally (remote_cmd empty) or on a cluster via remote_cmd.

    `remote_cmd` is an ssh/plink wrapper (e.g. "ssh -p 60026 user@host"); each
    cluster passes its own (REMOTE_CMD for PBS, SLURM_REMOTE_CMD for Slurm).
    `command` is one shell string appended as the remote command.
    """
    if remote_cmd:
        argv = shlex.split(remote_cmd) + [command]
    else:
        argv = shlex.split(command)
    return subprocess.run(argv, input=input_text,
                          capture_output=True, text=True)


# --- backends ---------------------------------------------------------------
def _submit_slurm(deffnm: str, work: Path, run_tag: str) -> dict:
    """Submit an sbatch job locally or on iREMB-6 via APW_SLURM_REMOTE_CMD.

    iREMB-6 runs GROMACS in a Singularity container (APW_GMX_SIF), not a module.
    Each submission runs in its own run_<tag> subdir to avoid clobbering.
    """
    node_line = f"#SBATCH -w {SLURM_NODE}\n" if SLURM_NODE else ""
    base = SLURM_REMOTE_WORKDIR if SLURM_REMOTE_CMD else str(work)
    run_dir = f"{base}/run_{run_tag}"
    script_text = (
        "#!/bin/bash\n"
        "#SBATCH -N 1\n"
        "#SBATCH --ntasks-per-node=1\n"
        "#SBATCH --cpus-per-task=8\n"
        "#SBATCH --gres=gpu:1\n"
        f"#SBATCH --partition={SLURM_PARTITION}\n"
        f"{node_line}"
        f"#SBATCH --job-name {deffnm}\n"
        "#SBATCH --output=slurm-%x.%J.out\n\n"
        f"mkdir -p {run_dir}\n"
        f"cp {base}/{deffnm}.tpr {run_dir}/\n"
        f"cd {run_dir}\n"
        "module purge\n"
        "module load Singularity/4.3.4\n"
        "srun singularity exec --nv -B /scratch:/scratch "
        f"{GMX_SIF} bash -c "
        f'"gmx mdrun -deffnm {deffnm} -ntomp 8 -nb gpu -pme cpu"\n'
    )
    if SLURM_REMOTE_CMD:
        # submit on iREMB-6: pipe the script to `sbatch` over ssh stdin
        res = _remote_run(SLURM_REMOTE_CMD,
                          f"cd {SLURM_REMOTE_WORKDIR} && sbatch",
                          input_text=script_text)
    else:
        script = work / f"{deffnm}.sbatch"
        script.write_text(script_text)
        res = subprocess.run(["sbatch", str(script)], cwd=str(work),
                             capture_output=True, text=True)
    if res.returncode != 0:
        return {"ok": False, "error": res.stderr.strip()}
    # sbatch prints "Submitted batch job 12345"
    out = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    sched = out[-1].split()[-1] if out else res.stdout.strip()
    return {"ok": True, "scheduler_id": sched, "run_dir": run_dir}


def _submit_pbs(deffnm: str, work: Path, run_tag: str) -> dict:
    """Write a PBS Pro script and submit it with qsub (iREMB pilot cluster).

    GROMACS comes from the system module (APW_PBS_GMX_MODULE, e.g.
    GROMACS/2025.02) — not conda — so no env activation is needed. Targets idle
    CPU nodes by default; set APW_PBS_NGPUS>0 to request GPUs. The queue must be
    within the <user> allocation (ALLOWED_PBS_QUEUES). Each submission runs in
    its own run_<tag> subdir so concurrent/repeat runs never clobber outputs.
    """
    if PBS_QUEUE not in ALLOWED_PBS_QUEUES:
        return {"ok": False,
                "error": f"queue {PBS_QUEUE!r} not in allocation "
                         f"{sorted(ALLOWED_PBS_QUEUES)}"}
    ngpus = int(PBS_NGPUS)
    select = f"select=1:ncpus={PBS_NCPUS}"
    md_flags = f"-ntomp {PBS_NCPUS} -nb cpu"
    if ngpus > 0:
        select += f":ngpus={ngpus}"
        md_flags = f"-ntomp {PBS_NCPUS} -nb gpu -pme cpu"  # -pme cpu avoids cuFFT
    launch = f"{PBS_MPI_LAUNCH} " if PBS_MPI_LAUNCH else ""
    extra_lines = "".join(f"module load {m}\n" for m in PBS_EXTRA_MODULES.split())
    base = REMOTE_WORKDIR if REMOTE_CMD else str(work)
    run_dir = f"{base}/run_{run_tag}"
    script_text = (
        "#!/bin/bash\n"
        f"#PBS -N {deffnm}\n"
        f"#PBS -q {PBS_QUEUE}\n"
        f"#PBS -l {select}\n"
        "#PBS -m abe\n"
        "#PBS -r n\n"
        "#PBS -V\n\n"
        f"mkdir -p {run_dir}\n"
        f"cp {base}/{deffnm}.tpr {run_dir}/\n"
        f"cd {run_dir}\n"
        "module purge\n"
        f"module load {PBS_GMX_MODULE}\n"
        f"{extra_lines}"
        f"{launch}{PBS_GMX_BIN} mdrun -deffnm {deffnm} {md_flags}\n"
    )
    if REMOTE_CMD:
        # submit on the cluster: pipe the script to `qsub` over plink/ssh stdin
        res = _remote_run(REMOTE_CMD, f"cd {REMOTE_WORKDIR} && qsub",
                          input_text=script_text)
    else:
        script = work / f"{deffnm}.pbs"
        script.write_text(script_text)
        res = subprocess.run(["qsub", str(script)], cwd=str(work),
                             capture_output=True, text=True)
    if res.returncode != 0:
        return {"ok": False, "error": res.stderr.strip()}
    # qsub prints the job id, e.g. "12345.iremb"; take the last non-empty line
    # so a remote login banner (via plink) does not corrupt the id.
    lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip()]
    return {"ok": True, "scheduler_id": lines[-1] if lines else res.stdout.strip(),
            "run_dir": run_dir}


def _submit_colab(deffnm: str, work: Path, run_tag: str) -> dict:
    """Run gmx mdrun as a background subprocess (works on Colab or any host).

    Runs in its own run_<tag> subdir. `-cpi/-noappend` lets a session that hits
    Colab's wall-clock limit resume from the last checkpoint, which the agent
    drives via repeated check_job.
    """
    run_dir = work / f"run_{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(work / f"{deffnm}.tpr", run_dir / f"{deffnm}.tpr")
    log = run_dir / f"{deffnm}.log"
    cmd = f"gmx mdrun -deffnm {deffnm} -ntomp 2 -nb gpu -cpi {deffnm}.cpt -noappend"
    proc = subprocess.Popen(
        shlex.split(cmd), cwd=str(run_dir),
        stdout=log.open("w"), stderr=subprocess.STDOUT,
    )
    return {"ok": True, "scheduler_id": f"pid:{proc.pid}", "log": str(log),
            "run_dir": str(run_dir)}


_BACKENDS = {"slurm": _submit_slurm, "pbs": _submit_pbs, "colab": _submit_colab}
_BACKENDS["hpc"] = _submit_slurm  # friendly alias for the Slurm/L40s cluster


@mcp.tool()
def submit_md_job(deffnm: str, backend: str = "hpc",
                  workdir: str | None = None) -> dict:
    """Submit a GROMACS production MD job to the chosen backend.

    Args:
        deffnm: GROMACS -deffnm base name (the .tpr must already exist).
        backend: "slurm"/"hpc" (sbatch, L40s), "pbs" (qsub, idle iREMB nodes),
            or "colab" (local subprocess).
        workdir: directory holding <deffnm>.tpr (defaults to APW_WORKDIR).
    Returns:
        dict with an internal job_id and the unique run_dir for this submission.
        Each call runs in its own run_<job_id> subdir, so repeat or concurrent
        submissions never overwrite each other's GROMACS outputs.
    """
    if backend not in _BACKENDS:
        return {"ok": False, "error": f"unknown backend {backend!r}"}
    work = Path(workdir).resolve() if workdir else WORKDIR
    # when submitting remotely the .tpr lives on the cluster, not locally
    remote = (SLURM_REMOTE_CMD if backend in ("slurm", "hpc")
              else REMOTE_CMD if backend == "pbs" else "")
    if not remote and not (work / f"{deffnm}.tpr").exists():
        return {"ok": False, "error": f"missing {work / f'{deffnm}.tpr'} (run grompp first)"}

    job_id = uuid.uuid4().hex[:8]
    sub = _BACKENDS[backend](deffnm, work, job_id)
    if not sub.get("ok"):
        return sub

    jobs = _load()
    jobs[job_id] = {"backend": backend, "deffnm": deffnm,
                    "work": str(work), **sub}
    _save(jobs)
    return {"ok": True, "job_id": job_id, "backend": backend,
            "scheduler_id": sub["scheduler_id"], "run_dir": sub.get("run_dir")}


@mcp.tool()
def stage_files(paths: list[str], dest_subdir: str = "") -> dict:
    """Upload local files to the cluster workdir via pscp (path B file staging).

    Reuses the same Pageant key as the plink command channel, so the whole
    workflow — stage, submit, poll — runs from the laptop with no manual WinSCP.
    Requires APW_SCP_CMD (e.g. "pscp -batch -P 60026") and APW_REMOTE_TARGET
    (e.g. "<user>@<pbs-host>").
    """
    if not SCP_CMD or not REMOTE_TARGET:
        return {"ok": False, "error": "set APW_SCP_CMD and APW_REMOTE_TARGET"}
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        return {"ok": False, "error": f"missing local files: {missing}"}
    dest = REMOTE_WORKDIR + (f"/{dest_subdir}" if dest_subdir else "")
    if REMOTE_CMD:  # make sure the destination directory exists
        _remote_run(REMOTE_CMD, f"mkdir -p {dest}")
    argv = (shlex.split(SCP_CMD) + [str(p) for p in paths]
            + [f"{REMOTE_TARGET}:{dest}/"])
    res = subprocess.run(argv, capture_output=True, text=True)
    if res.returncode != 0:
        return {"ok": False, "error": res.stderr.strip() or res.stdout.strip()}
    return {"ok": True, "uploaded": paths, "dest": dest}


@mcp.tool()
def check_job(job_id: str) -> dict:
    """Poll a submitted job's status across either backend."""
    job = _load().get(job_id)
    if not job:
        return {"ok": False, "error": f"unknown job_id {job_id!r}"}
    backend, sid = job["backend"], job["scheduler_id"]

    if backend in ("slurm", "hpc"):
        # routed through APW_SLURM_REMOTE_CMD (iREMB-6) when set
        q = _remote_run(SLURM_REMOTE_CMD, f"squeue -j {sid} -h -o %T")
        state = q.stdout.strip() or "COMPLETED_OR_GONE"  # gone from queue = done
        return {"ok": True, "job_id": job_id, "state": state}

    if backend == "pbs":
        # qstat -x includes finished jobs; -f gives the job_state field.
        # Routed through plink/ssh when APW_REMOTE_CMD is set (path B).
        q = _remote_run(REMOTE_CMD, f"qstat -x -f {sid}")
        if q.returncode != 0:
            return {"ok": True, "job_id": job_id, "state": "GONE"}
        state = "UNKNOWN"
        for line in q.stdout.splitlines():
            if "job_state" in line:
                code = line.split("=")[-1].strip()
                state = {"Q": "QUEUED", "R": "RUNNING",
                         "E": "EXITING", "F": "FINISHED"}.get(code, code)
                break
        return {"ok": True, "job_id": job_id, "state": state}

    # colab/local backend: check the PID
    pid = int(sid.split(":")[1])
    alive = Path(f"/proc/{pid}").exists()
    return {"ok": True, "job_id": job_id,
            "state": "RUNNING" if alive else "FINISHED", "log": job.get("log")}


if __name__ == "__main__":
    mcp.run()
