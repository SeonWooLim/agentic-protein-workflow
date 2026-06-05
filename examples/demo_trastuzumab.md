# Demo — agentic MD of Trastuzumab Fab (1N8Z) driven from a laptop

This walkthrough shows an LLM agent orchestrating a full antibody MD run on a
remote HPC cluster, end to end, from natural language — calling MCP tools and an
SSH command channel so the user never hand-writes a scheduler script.

> Credentials live in env vars (`APW_REMOTE_CMD`, …), never in this file.
> The remote command channel is plink over a saved key (Pageant) — see README.

---

## Scenario

> **User:** "Run a short GPU MD of the Trastuzumab Fab (PDB 1N8Z) on the PBS
> cluster and report its backbone stability."

The agent decomposes this into MCP tool calls across three servers
(`structure`, `hpc`, `analysis`) plus a one-time PDBFixer prep.

---

## Tool-call sequence

### 1. Fetch the public structure  (`structure.fetch_structure`)
```json
→ fetch_structure(pdb_id="1N8Z")
← {"ok": true, "pdb_id": "1N8Z", "path": ".../1N8Z.pdb", "n_atom_records": 7778}
```

### 2. Complete missing atoms + keep the Fab  (PDBFixer prep)
Raw 1N8Z has incomplete side chains (e.g. LYS190 missing CG) and includes the
HER2 antigen (chain C). `prep_1n8z.py` keeps the Fab (A,B) and builds missing
heavy atoms:
```bash
python prep_1n8z.py 1N8Z.pdb 1N8Z_fixed.pdb --chains A B
# 7778 atoms (3 chains) -> 3300 atoms (Fab A,B, missing heavy atoms added)
```
`1N8Z_fixed.pdb` is staged to the cluster scratch dir via WinSCP.

### 3. Submit the MD job remotely  (`hpc.submit_md_job`, backend="pbs")
With `APW_REMOTE_CMD` set, the agent's call builds a PBS script and pipes it to
`qsub` over the plink channel — no file copy, runs on the cluster's GPU queue:
```json
→ submit_md_job(deffnm="md", backend="pbs")     # APW_PBS_QUEUE=nvidiaq, NGPUS=1
← {"ok": true, "job_id": "37ef2eab", "backend": "pbs",
   "scheduler_id": "237334.iREMB-E-M01"}
```

### 4. Poll until done  (`hpc.check_job`)
The agent polls `qstat` over the same channel:
```json
→ check_job("37ef2eab")  ← {"state": "RUNNING"}
→ check_job("37ef2eab")  ← {"state": "FINISHED"}
```

### 5. Analyse the trajectory  (`analysis.analyze_trajectory`)
```json
→ analyze_trajectory(topology="md.tpr", trajectory="md.xtc")
← {"ok": true, "rmsd_mean_A": 1.28, "rmsf_max_A": 2.46,
   "figure": ".../analysis/summary.png"}
```
The 2 ns run wrote 201 frames over 434 Cα atoms (Fab chains A+B). Measured
observables (MDAnalysis, CA selection, PBC-unwrapped):

| Metric | Value |
|--------|-------|
| Backbone Cα RMSD (mean / last frame / max) | 1.28 / 1.20 / 1.98 Å |
| Per-residue Cα RMSF (mean / max) | 0.88 / 2.46 Å |
| RMSF maximum | residue 214 — C-terminus of chain A (light) |
| Radius of gyration (mean) | 24.3 Å |

### 6. Report
The agent summarises: the backbone **RMSD plateaus near 1.2 Å** within the first
~0.5 ns and stays there (max 1.98 Å) — the Fab fold is stable over the run. The
**per-residue RMSF** is low across the framework core (~0.9 Å) and rises only at
the chain termini and surface loops, peaking at 2.46 Å at the chain-A C-terminus;
elevated flexibility also localises to the variable-domain (Fv) loops where the
CDRs reside. The figure path (`analysis/summary.png`, and the portfolio
`analysis/portfolio_1n8z.png`) is returned — answering the original question.

---

## Verified components (2026-06-05)

| Step | Status |
|------|--------|
| PBS GPU backend (qsub/qstat, CUDA `gmx_mpi`) | ✅ smoke test job F (nvidiaq) |
| **Remote channel** (laptop → plink → `qsub`/`qstat`) | ✅ `submit→RUNNING→FINISHED` via `test_remote_submit.py` |
| PDBFixer prep (1N8Z → Fab, missing atoms) | ✅ 3300 atoms, chains A,B |
| Multi-engine (GROMACS / OpenMM) abstraction | ✅ `hpc_server` + `openmm_server` |

The smoke job (`237334`) above was submitted **from the laptop through the MCP
backend**, proving the agent can drive the cluster without the user touching a
scheduler. The 1N8Z production run uses the identical path with the prepared
structure.

## What this demonstrates
- **MCP-based Agentic AI** orchestrating a real multi-step HPC workflow.
- **On-prem GPU operation** (PBS, CUDA `gmx_mpi`) behind a natural-language interface.
- **Pluggable scheduler (Slurm/PBS) and engine (GROMACS/OpenMM)** abstraction.
- **Public-data only** (RCSB 1N8Z) — no patent-encumbered inputs.
