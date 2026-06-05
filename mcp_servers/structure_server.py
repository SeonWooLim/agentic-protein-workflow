"""MCP server: protein structure acquisition & sequence extraction.

Exposes tools an agent can call to fetch a *public* PDB structure and pull
per-chain sequences. RCSB public data only — no patent-encumbered inputs.

Run standalone:
    python mcp_servers/structure_server.py
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

from mcp.server.fastmcp import FastMCP

WORKDIR = Path(os.environ.get("APW_WORKDIR", "./work")).resolve()
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

mcp = FastMCP("structure")


@mcp.tool()
def fetch_structure(pdb_id: str) -> dict:
    """Download a public PDB structure from RCSB.

    Args:
        pdb_id: 4-character PDB accession (e.g. "1N8Z" = Trastuzumab Fab).
    Returns:
        dict with the local path and a basic atom-record count.
    """
    pdb_id = pdb_id.strip().upper()
    if len(pdb_id) != 4:
        return {"ok": False, "error": f"invalid PDB id: {pdb_id!r}"}

    WORKDIR.mkdir(parents=True, exist_ok=True)
    out = WORKDIR / f"{pdb_id}.pdb"
    if not out.exists():
        try:
            urllib.request.urlretrieve(RCSB_URL.format(pdb_id=pdb_id), out)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"download failed: {exc}"}

    n_atoms = sum(1 for line in out.read_text().splitlines()
                  if line.startswith(("ATOM", "HETATM")))
    return {"ok": True, "pdb_id": pdb_id, "path": str(out),
            "n_atom_records": n_atoms}


@mcp.tool()
def extract_sequence(pdb_path: str) -> dict:
    """Extract per-chain SEQRES sequences from a PDB file (BioPython SeqIO).

    Writes a FASTA next to the input and returns a {chain_id: sequence} map.
    For an antibody Fab this yields the heavy (H) and light (L) chains that
    feed downstream design / aggregation prediction.
    """
    from Bio import SeqIO  # lazy import: server still starts without Bio

    path = Path(pdb_path)
    if not path.exists():
        return {"ok": False, "error": f"not found: {pdb_path}"}

    chains: dict[str, str] = {}
    for record in SeqIO.parse(str(path), "pdb-seqres"):
        chain_id = record.annotations.get("chain", record.id.split(":")[-1])
        chains[chain_id] = str(record.seq)

    fasta = path.with_suffix(".fasta")
    with fasta.open("w") as fh:
        for cid, seq in chains.items():
            fh.write(f">{path.stem}_{cid}\n{seq}\n")

    return {"ok": True, "chains": chains, "fasta": str(fasta)}


if __name__ == "__main__":
    mcp.run()
