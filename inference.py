# inference.py
# 
# this file contains the class CheMeleonFingerprint which can be instantiated
# and called to generate the CheMeleon learned embeddings for a list of SMILES
# strings and/or RDKit Mols. you may wish to simply copy or download this file directly for use,
# or adapt the code for your own purposes. No other files are required for it
# to work, though you must `pip install 'chemprop>=2.2.0'` for this to run.
#
# run `python chemeleon_fingerprint.py` for a quick usage demo, otherwise you
# should `import` the CheMeleonFingerprint class into your other code and use
# it there (following the example at the bottom of this file) to generate
# your learned fingerprints
#
# this file was adapted from the CheMeleon repository:
# https://github.com/JacksonBurns/chemeleon/blob/237fa44d42fa503cecc095cf0aadf3a9eef52a95/chemeleon_fingerprint.py
from pathlib import Path

import torch
from chemprop import featurizers
from chemprop.data import BatchMolGraph
from chemprop.models import load_model
from rdkit.Chem import MolFromSmiles, Mol
import numpy as np


class CheMeleonAqueous:
    def __init__(self, device: str | torch.device | None = None, model_path: str | Path | None = None):
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        if model_path is None:
            model_path = Path(__file__).parent / "chemeleon_aqueous.pt"
        self.model = load_model(model_path)
        self.model.eval()
        if device is not None:
            self.model.to(device=device)

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        bmg = BatchMolGraph([self.featurizer(MolFromSmiles(m) if isinstance(m, str) else m) for m in molecules])
        bmg.to(device=self.model.device)
        with torch.no_grad():
            return self.model(bmg).numpy(force=True)


if __name__ == "__main__":
    chemeleon_aqueous = CheMeleonAqueous()
    chemeleon_aqueous([
        "O=C(C)Oc1ccccc1C(=O)O",  # aspirin
        MolFromSmiles("CC(=O)Nc1ccc(O)cc1"),  # acetaminophen
    ])