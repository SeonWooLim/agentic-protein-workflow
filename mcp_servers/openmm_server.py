"""MCP server: OpenMM in-process MD engine (lightweight, no job scheduler).

A second MD *engine* alongside the GROMACS/scheduler backends in hpc_server.py.
OpenMM runs entirely inside this Python process via its API — no sbatch/qsub —
so an agent can run a short MD directly, which is ideal for Colab/local demos.
The compute Platform (CUDA > OpenCL > CPU) is auto-detected, so the same call
uses a GPU when present and transparently falls back to CPU otherwise.

Public PDB input only (no patent-encumbered data).

Run standalone:
    python mcp_servers/openmm_server.py
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKDIR = Path(os.environ.get("APW_WORKDIR", "./work")).resolve()
mcp = FastMCP("openmm")


def _pick_platform():
    """Return the fastest available OpenMM Platform (CUDA > OpenCL > CPU)."""
    from openmm import Platform
    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            return Platform.getPlatformByName(name)
        except Exception:  # noqa: BLE001
            continue
    return None


@mcp.tool()
def list_platforms() -> dict:
    """List the OpenMM compute platforms available on this host (GPU vs CPU)."""
    from openmm import Platform
    names = [Platform.getPlatform(i).getName()
             for i in range(Platform.getNumPlatforms())]
    chosen = _pick_platform()
    return {"ok": True, "available": names,
            "selected": chosen.getName() if chosen else None}


@mcp.tool()
def run_openmm_md(pdb_path: str, nsteps: int = 25000,
                  forcefield: str = "amber14-all.xml",
                  water: str = "amber14/tip3pfb.xml",
                  solvate: bool = False,
                  report_interval: int = 1000) -> dict:
    """Run a short in-process OpenMM MD on a public PDB; write a DCD trajectory.

    The DCD + topology PDB it writes feed analysis_server.analyze_trajectory,
    so OpenMM and GROMACS runs land in the same downstream analysis path.

    Args:
        pdb_path: input PDB (e.g. a Trastuzumab Fab from structure_server).
        nsteps: number of 4 fs integration steps (25000 = 100 ps).
        forcefield/water: OpenMM ForceField XML names.
        solvate: add a TIP3P water box + 0.15 M ions when True.
        report_interval: steps between trajectory/state frames.
    Returns:
        dict with the trajectory path, the platform actually used, and the
        final potential energy.
    """
    from openmm import LangevinMiddleIntegrator, unit
    from openmm.app import (DCDReporter, ForceField, HBonds, Modeller, PDBFile,
                            PME, Simulation, StateDataReporter)

    src = Path(pdb_path)
    if not src.exists():
        return {"ok": False, "error": f"not found: {pdb_path}"}

    pdb = PDBFile(str(src))
    ff = ForceField(forcefield, water)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    if solvate:
        modeller.addSolvent(ff, padding=1.0 * unit.nanometer,
                            ionicStrength=0.15 * unit.molar)

    system = ff.createSystem(
        modeller.topology, nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(
        310 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds)

    platform = _pick_platform()
    if platform is None:
        return {"ok": False, "error": "no OpenMM platform available"}

    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()

    out = WORKDIR / "openmm"
    out.mkdir(parents=True, exist_ok=True)
    top = out / f"{src.stem}_system.pdb"
    with top.open("w") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    dcd = out / f"{src.stem}.dcd"
    sim.reporters.append(DCDReporter(str(dcd), report_interval))
    sim.reporters.append(StateDataReporter(
        str(out / f"{src.stem}.log"), report_interval,
        step=True, potentialEnergy=True, temperature=True))
    sim.step(nsteps)

    energy = sim.context.getState(getEnergy=True).getPotentialEnergy()
    return {"ok": True, "engine": "openmm",
            "platform": platform.getName(),
            "topology": str(top), "trajectory": str(dcd),
            "n_steps": nsteps,
            "final_potential_kJ_mol":
                energy.value_in_unit(unit.kilojoule_per_mole)}


if __name__ == "__main__":
    mcp.run()
