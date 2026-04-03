# inference.py
# 
# this file contains the class CheMeleonAqueous which can be instantiated
# and called to predict aqueous solubility (in logS) for a list of SMILES
# strings and/or RDKit Mols. you may wish to simply copy or download this file directly for use,
# or adapt the code for your own purposes. No other files are required for it
# to work, though you must `pip install 'chemprop>=2.2.1'` for this to run.
#
# run `python inference.py` for a quick usage demo, otherwise you
# should `import` the CheMeleonAqueous class into your other code and use
# it there (following the example at the bottom of this file) to generate
# your predictions
#
# this file was adapted from the CheMeleon repository:
# https://github.com/JacksonBurns/chemeleon/blob/237fa44d42fa503cecc095cf0aadf3a9eef52a95/chemeleon_fingerprint.py
from pathlib import Path

import sys
import argparse
import torch
from chemprop import featurizers
from chemprop.data import BatchMolGraph
from chemprop.models import load_model
from rdkit.Chem import MolFromSmiles, Mol
import pandas as pd
import numpy as np


class CheMeleonAqueous:
    def __init__(self, device: str | torch.device | None = None, model_path: str | Path | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        if model_path is None:
            model_path = Path(__file__).parent / 'pretrained' / "chemeleon_aqueous.pt"
        self.model = load_model(model_path)
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph([self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model(bmg).numpy(force=True)


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(prog='chemeleon-aqsol', description=(
        'Applies the pretrained model to predict kinetic aqueous solubility on '
        'new compounds (uses chemeleon_aqueous.pt).'
    ))
    parser.add_argument('infile', type=Path)
    parser.add_argument('-c', '--smiles-col', default='SMILES')
    parser.add_argument('-o', '--outfile', type=Path, default=None)

    args = parser.parse_args(arguments)
    if args.outfile is None:
        args.outfile = args.infile.parent / (args.infile.stem + '_out.csv')
    return args


def main(arguments = None):

    # Parse arguments
    args = parse_args(arguments)

    # Read data
    df = pd.read_csv(args.infile)
    smiles = df[args.smiles_col].to_list()

    # Run inference
    chemaqsol = CheMeleonAqueous()
    y_pred = chemaqsol(smiles)

    df['chemeleon-aqsol logS(M)'] = y_pred
    df.to_csv(args.outfile, index=False)
    

if __name__ == "__main__":
    main()
