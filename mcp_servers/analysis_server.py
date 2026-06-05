"""MCP server: trajectory analysis & report generation.

Computes standard MD observables (RMSD, RMSF, Rg) with MDAnalysis and renders
a Markdown report with figures. SASA can be delegated to `gmx sasa` (optional;
left as a TODO hook). Outputs follow the project convention: CSV + PNG + md.

Run standalone:
    python mcp_servers/analysis_server.py
"""
from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKDIR = Path(os.environ.get("APW_WORKDIR", "./work")).resolve()
mcp = FastMCP("analysis")


@mcp.tool()
def analyze_trajectory(topology: str, trajectory: str,
                       selection: str = "protein and name CA") -> dict:
    """Compute RMSD / RMSF / Rg for a trajectory and write CSV + PNG.

    Args:
        topology: .tpr/.gro/.pdb path.
        trajectory: .xtc/.trr path.
        selection: MDAnalysis atom selection used for all three metrics.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    u = mda.Universe(topology, trajectory)
    ca = u.select_atoms(selection)

    # RMSD vs the first frame
    R = rms.RMSD(u, u, select=selection).run()
    time_ps = R.results.rmsd[:, 1]
    rmsd = R.results.rmsd[:, 2]                       # Angstrom

    # RMSF per residue (requires an aligned trajectory)
    align.AlignTraj(u, u, select=selection, in_memory=True).run()
    rmsf = rms.RMSF(ca).run().results.rmsf

    # Radius of gyration over time
    rg = np.array([ca.radius_of_gyration() for _ in u.trajectory])

    out = WORKDIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_ps": time_ps, "rmsd_A": rmsd, "rg_A": rg}).to_csv(
        out / "timeseries.csv", index=False)
    pd.DataFrame({"resid": ca.resids, "rmsf_A": rmsf}).to_csv(
        out / "rmsf.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(time_ps, rmsd); ax[0].set(xlabel="time (ps)", ylabel="RMSD (Å)")
    ax[1].plot(ca.resids, rmsf); ax[1].set(xlabel="residue", ylabel="RMSF (Å)")
    fig.tight_layout(); fig.savefig(out / "summary.png", dpi=150)

    return {"ok": True, "out_dir": str(out),
            "rmsd_mean_A": float(rmsd.mean()),
            "rmsf_max_A": float(rmsf.max()),
            "figure": str(out / "summary.png")}


@mcp.tool()
def generate_report(out_dir: str, title: str = "MD analysis report") -> dict:
    """Assemble a Markdown report from analysis artifacts in `out_dir`."""
    out = Path(out_dir)
    figs = sorted(out.glob("*.png"))
    lines = [f"# {title}", ""]
    if (out / "timeseries.csv").exists():
        lines += ["## Time series", "`timeseries.csv` written.", ""]
    for fig in figs:
        lines += [f"![{fig.stem}]({fig.name})", ""]
    md = out / "report.md"
    md.write_text("\n".join(lines))
    return {"ok": True, "report": str(md), "figures": [f.name for f in figs]}


if __name__ == "__main__":
    mcp.run()
