# chemeleon_aqueous_solubility

This repo shows how to train a CheMeleon model to predict aqueous solubility.
In particular we use pre-training to improve performance and implement a custom Mixture of Experts Feed-forward Neural Network to decrease training time as well as improve overall performance.

If you want to use the pre-trained model for your own data, see [`inference.py`](./inference.py).

To run this demo you will need to install (with Python 3.11 or 3.12) `'chemprop[hpopt]>=2.2.1'` (i.e., `pip install 'chemprop[hpopt]>=2.2.1'`).

## Description of Files

 - [`opt.py`](./opt.py): Main driver script - this shows how to train a single CheMeleon model for aqueous solubility prediction, as well as how to run hyperparameter optimization for the MOE model.
 - [`finetune.py`](./finetune.py): Takes a pre-trained model from `opt.py` and finetunes it on the testing dataset from `Polaris`.
 - [`inference.py`](./inference.py): Demonstrates using the pretrained or finetuned model for inference on new compounds without any further training or finetuning.
 - [`moe_ffn.py`](./moe_ffn.py): Implements the 'Adaptive Mixtures of Local Experts' model (aka Mixture of Experts) from [Jacobs _et al._](https://doi.org/10.1162/neco.1991.3.1.79) 1991 paper.
 - [`baybekov_ksol.csv`](./baybekov_ksol.csv): SMILES CSV version of the dataset curated in [this paper](https://doi.org/10.1002/minf.202300216) and available at [this link](https://doi.org/10.57745/ZWS0WC) in its original format.
 - [`clean_smiles.py`](./clean_smiles.py): This basic SMILES preprocessing script just helps to improve the performance of ML models (I've found empirically) - see the docstring therein for more details.
 - [`analyze_experts.ipynb`](./analyze_experts.ipynb): This notebook isn't fully built out, but was a helpful debugging tool during initial development to see if the mixture of experts model was actually working. Might prove interesting with more analysis!

## License

All of this code is made available under the [MIT License](./LICENSE) - please use it in your own workflows, and (if you want) send me a message if it turns out to be helpful!

Some of this code (particularly for unit conversions) has been adapted from [this repo](https://github.com/PatWalters/practical_cheminformatics_posts/blob/fefdb70fcabd05e603ac1fe473e82cab3bd9b2a8/solubility_2025/evaluate_solubility_model.ipynb) under the terms of the MIT license.
In other places where code has been re-used or modified, it is cited inline.
