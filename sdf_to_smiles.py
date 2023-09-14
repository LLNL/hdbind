import sys
from rdkit import Chem

def converter(file_name):
    sppl = Chem.SDMolSupplier(file_name)
    sppl = [Chem.MolFromMol2File(file_name)]
    outname = file_name.replace(".sdf", ".txt")
    out_file = open(outname, "w")
    for mol in sppl:
        if mol is not None:# some compounds cannot be loaded.
            smi = Chem.MolToSmiles(mol, sanitize=False)
            name = mol.GetProp("_Name")
            out_file.write(f"{smi}\t{name}\n")
    out_file.close()
if __name__ == "__main__":
    converter(sys.argv[1])
