"""Prepare a crystal PDB for MD: keep selected chains, add missing heavy atoms,
drop heterogens. Fixes the common pdb2gmx failure on incomplete side chains
(e.g. 1N8Z LYS190 is missing its CG). Uses PDBFixer (OpenMM ecosystem).

Install:  conda install -c conda-forge pdbfixer openmm
Usage:    python prep_1n8z.py 1N8Z.pdb 1N8Z_fixed.pdb --chains A B

Default keeps chains A,B = Trastuzumab Fab (antibody); chain C is the HER2
antigen — keep it too (omit --chains) for an antibody-antigen complex.
Hydrogens are NOT added here; pdb2gmx adds them (run with -ignh).
"""
import argparse

from pdbfixer import PDBFixer
from openmm.app import PDBFile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--chains", nargs="*", default=["A", "B"],
                    help="chain IDs to keep (default: A B = Fab)")
    args = ap.parse_args()

    fixer = PDBFixer(filename=args.inp)

    # 1) keep only the requested chains
    keep = set(args.chains)
    remove = [i for i, c in enumerate(fixer.topology.chains()) if c.id not in keep]
    if remove:
        fixer.removeChains(remove)  # positional = chainIndices (version-agnostic)

    # 2) find missing residues, but drop those at chain termini (disordered
    #    tails) so we don't build long unresolved loops
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    # 3) standardise residues, strip heterogens, build missing heavy atoms
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with open(args.out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    print(f"wrote {args.out} (chains kept: {sorted(keep)})")


if __name__ == "__main__":
    main()
