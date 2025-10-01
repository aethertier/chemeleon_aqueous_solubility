from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover


def clean_smiles(smiles: str, remove_hs: bool = True, strip_stereochem: bool = False, strip_salts: bool = True) -> str:
    """Applies preprocessing to SMILES strings, seeking the 'parent' SMILES

    Note that this is different from simply _neutralizing_ the input SMILES - we attempt to get the parent molecule, analogous to a molecular skeleton.
    This is adapted in part from https://rdkit.org/docs/Cookbook.html#neutralizing-molecules

    Args:
        smiles (str): input SMILES
        remove_hs (bool, optional): Removes hydrogens. Defaults to True.
        strip_stereochem (bool, optional): Remove R/S and cis/trans stereochemistry. Defaults to False.
        strip_salts (bool, optional): Remove salt ions. Defaults to True.

    Returns:
        str: cleaned SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    if strip_stereochem:
        Chem.RemoveStereochemistry(mol)
    if strip_salts:
        remover = SaltRemover()  # use default saltremover
        mol = remover.StripMol(mol)  # strip salts

    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return Chem.MolToSmiles(mol)  # this also canonicalizes the input
