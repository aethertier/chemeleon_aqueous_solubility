from pathlib import Path
from urllib.request import urlretrieve
from datetime import datetime

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from astartes import train_test_split
import pandas as pd
import numpy as np

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing
from chemprop.models import MPNN
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.transforms import UnscaleTransform

from .clean_smiles import clean_smiles
from .moe_ffn import ExpertMixtureRegressionFFN


def define_by_run(trial):
    num_experts = trial.suggest_int("num_experts", 2, 6)
    expert_hidden_dim = trial.suggest_categorical("expert_hidden_dim", [128, 256, 512, 1024])
    expert_n_layers = trial.suggest_int("expert_n_layers", 1, 3)
    gate_hidden_dim = trial.suggest_categorical("gate_hidden_dim", [512, 1024, 2048])
    gate_n_layers = trial.suggest_int("gate_n_layers", 1, 3)
    return {
        "num_experts": num_experts,
        "expert_hidden_dim": expert_hidden_dim,
        "expert_n_layers": expert_n_layers,
        "gate_hidden_dim": gate_hidden_dim,
        "gate_n_layers": gate_n_layers,
    }


def train_one(
    *,
    num_experts: int,
    expert_hidden_dim: int,
    expert_n_layers: int,
    gate_hidden_dim: int,
    gate_n_layers: int,
):
    outdir = Path("output") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir.mkdir(parents=True, exist_ok=True)
    # https://github.com/JacksonBurns/chemeleon/blob/237fa44d42fa503cecc095cf0aadf3a9eef52a95/chemeleon_fingerprint.py#L27C9-L39C55
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    agg = MeanAggregation()
    ckpt_dir = Path().home() / ".chemprop"
    ckpt_dir.mkdir(exist_ok=True)
    mp_path = ckpt_dir / "chemeleon_mp.pt"
    if not mp_path.exists():
        urlretrieve(
            r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
            mp_path,
        )
    chemeleon_mp = torch.load(mp_path, weights_only=True)
    mp = BondMessagePassing(**chemeleon_mp['hyper_parameters'])
    mp.load_state_dict(chemeleon_mp['state_dict'])

    train_df = pd.read_csv("baybekov_ksol.csv")

    smiles_col = "SMILES"
    train_df[smiles_col] = train_df[smiles_col].apply(clean_smiles)
    target = train_df[["logS"]].values

    train_idxs, val_idxs = train_test_split(
        np.arange(train_df.shape[0]),
        train_size=0.80,
        test_size=0.20,
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
    target_scaler = train_dataset.normalize_targets()
    val_dataset.normalize_targets(target_scaler)
    output_transform = UnscaleTransform.from_standard_scaler(target_scaler)
    train_dataloader = build_dataloader(train_dataset, num_workers=1)
    val_dataloader = build_dataloader(val_dataset, num_workers=1, shuffle=False)
    fnn = ExpertMixtureRegressionFFN(
        n_experts=num_experts,
        n_tasks=1,
        input_dim=mp.output_dim,
        hidden_dim=expert_hidden_dim,
        n_layers=expert_n_layers,
        gate_hidden_dim=gate_hidden_dim,
        gate_n_layers=gate_n_layers,
        output_transform=output_transform,
    )

    model = MPNN(
        mp,
        agg,
        fnn,
        batch_norm=False,
        init_lr=1e-5,
        max_lr=1e-4,
        final_lr=1e-6,
        warmup_epochs=5,
    )
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
            patience=5,
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=outdir / "checkpoints",
        ),
    ]
    trainer = Trainer(
        max_epochs=30,
        logger=tensorboard_logger,
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print(f"Reloading best model from checkpoint file: {ckpt_path}")
    model = MPNN.load_from_checkpoint(ckpt_path)
    result = trainer.validate(model, val_dataloader)[0]["val_loss"]
    return result

if __name__ == "__main__":
    # run hyperparameter optimization
    # import optuna

    # study = optuna.create_study(direction="minimize")
    # study.optimize(lambda trial: train_one(**define_by_run(trial)), n_trials=8)
    # print("Best hyperparameters: ", study.best_params)
    # # dump all results to csv
    # study.trials_dataframe().to_csv("optuna_study_results.csv")

    train_one(
        num_experts=4,
        expert_hidden_dim=128,
        expert_n_layers=3,
        gate_hidden_dim=512,
        gate_n_layers=1,
    )
