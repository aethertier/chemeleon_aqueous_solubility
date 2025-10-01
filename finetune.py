import polaris as po
import pickle
from chemprop.models import MPNN
from lightning import Trainer
import torch
from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from chemprop.data import MoleculeDatapoint, build_dataloader, MoleculeDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit.Chem.Descriptors import MolWt
import numpy as np
from astartes import train_test_split
from rdkit.Chem import MolFromSmiles
    

# model predictions are in log10 of molar solubility (logS), convert to units of assay which are log10 of ug/mL
def logS_to_logugmL(smiles, logS):
    mol = MolFromSmiles(smiles)
    wt = MolWt(mol)
    S = 10**logS
    return np.log10(S * wt * 1000)


if __name__ == "__main__":
    foundation_model = Path(r"output/2025-10-01_13-49-22/checkpoints/epoch=8-step=5139.ckpt")

    # Load the benchmark from the Hub
    benchmark = po.load_benchmark("polaris/adme-fang-solu-1")
    smiles_col = list(benchmark.input_cols)[0]
    target_col = list(benchmark.target_cols)[0]

    # Get the train and test data-loaders
    train, test = benchmark.get_train_test_split()
    train_df = train.as_dataframe()
    test_df = test.as_dataframe()

    model = MPNN.load_from_checkpoint(foundation_model)
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    # start by checking the zero-shot performance of the foundation model
    test_data = [
        MoleculeDatapoint.from_smi(smi)
        for smi in test_df[smiles_col]
    ]
    test_dataset = MoleculeDataset(test_data, featurizer)
    test_dataloader = build_dataloader(test_dataset, num_workers=1, shuffle=False)
    trainer = Trainer(enable_progress_bar=False, logger=False)
    predictions = torch.cat(trainer.predict(model, dataloaders=test_dataloader)).numpy(force=True).flatten()
    predictions = [logS_to_logugmL(smi, pred) for smi, pred in zip(test_df[smiles_col], predictions)]

    # Evaluate your predictions
    results = benchmark.evaluate(predictions)
    result_str = ""
    result_str += "Zero-shot results on test set:\n"
    result_str += str(results.results) + "\n"

    # now finetune the model on the training set
    target = train_df[[target_col]].values
    # change units to match the original model training (logS)
    target = np.array([np.log10(10**y / (MolWt(MolFromSmiles(smi)) * 1000)) for smi, y in zip(train_df[smiles_col], target)]).reshape(-1, 1)
    model = MPNN.load_from_checkpoint(foundation_model)
    model.to(torch.device("cpu"))

    train_idxs, val_idxs = train_test_split(
        np.arange(train_df.shape[0]),
        train_size=0.70,
        test_size=0.30,
        random_state=42,
    )
    train_data = [
        MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(
            train_df[smiles_col].iloc[train_idxs], target[train_idxs]
        )
    ]
    val_data = [
        MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(
            train_df[smiles_col].iloc[val_idxs], target[val_idxs]
        )
    ]
    train_dataset = MoleculeDataset(train_data, featurizer)
    val_dataset = MoleculeDataset(val_data, featurizer)
    # rescale the targets using the training set statistics
    target_scaler = model.predictor.output_transform.to_standard_scaler()
    train_dataset.normalize_targets(target_scaler)
    val_dataset.normalize_targets(target_scaler)
    train_dataloader = build_dataloader(train_dataset, num_workers=1)
    val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False)
    outdir = foundation_model.parent.resolve() / "biogen_finetune"
    tensorboard_logger = TensorBoardLogger(
        outdir,
        name="tensorboard_logs",
        default_hp_metric=False,
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=False,
            patience=10,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=outdir / "checkpoints",
        ),
    ]
    trainer = Trainer(
        max_epochs=50,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Reloading best fine-tuned model from checkpoint file: {ckpt_path}")
    model = MPNN.load_from_checkpoint(ckpt_path)

    trainer = Trainer(enable_progress_bar=False, logger=False)
    predictions = torch.cat(trainer.predict(model, dataloaders=test_dataloader)).numpy(force=True).flatten()
    predictions = [logS_to_logugmL(smi, pred) for smi, pred in zip(test_df[smiles_col], predictions)]

    results = benchmark.evaluate(predictions)
    result_str += "Fine-tune results on test set:\n"
    result_str += str(results.results) + "\n"

    print(result_str)

    # we'll write the results to disk so I can upload them to polaris later
    with open("polaris_results.pkl", "wb") as f:
        pickle.dump(results, f)
