from chemprop.nn import Predictor, PredictorRegistry
from chemprop.utils import Factory
from chemprop.nn.predictors import MLP
from chemprop.nn.metrics import MSE, ChempropMetric
from chemprop.conf import DEFAULT_HIDDEN_DIM
from chemprop.nn.transforms import UnscaleTransform

from lightning.pytorch.core.mixins import HyperparametersMixin

from torch import nn, Tensor
import torch.nn.functional as F
import torch


@PredictorRegistry.register("regression-moe")
class ExpertMixtureRegressionFFN(Predictor, HyperparametersMixin):
    n_targets = 1
    _T_default_criterion = MSE
    _T_default_metric = MSE

    def __init__(
        self,
        n_experts: int = 2,
        n_tasks: int = 1,
        input_dim: int = DEFAULT_HIDDEN_DIM,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str | nn.Module = "relu",
        gate_hidden_dim: int | None = None,
        gate_n_layers: int = 1,
        criterion: ChempropMetric | None = None,
        task_weights: Tensor | None = None,
        threshold: float | None = None,
        output_transform: UnscaleTransform | None = None,
    ):
        super().__init__()
        ignore_list = ["criterion", "output_transform", "activation"]
        self.save_hyperparameters(ignore=ignore_list)
        self.hparams["criterion"] = criterion
        self.hparams["output_transform"] = output_transform
        self.hparams["activation"] = activation
        self.hparams["cls"] = self.__class__

        self.n_experts = n_experts

        # Experts
        self.experts = nn.ModuleList([
            MLP.build(input_dim, n_tasks * self.n_targets, hidden_dim, n_layers, dropout, activation)
            for _ in range(n_experts)
        ])

        # Gating network
        gate_hidden_dim = gate_hidden_dim or hidden_dim
        self.gate = MLP.build(
            input_dim,
            n_experts,
            gate_hidden_dim,
            gate_n_layers,
            dropout,
            activation,
        )

        # Criterion
        task_weights = torch.ones(n_tasks) if task_weights is None else task_weights
        self.criterion = criterion or Factory.build(
            self._T_default_criterion, task_weights=task_weights, threshold=threshold
        )

        self.output_transform = output_transform if output_transform is not None else nn.Identity()

        # For analysis
        self._last_gate_weights: Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.experts[0].input_dim

    @property
    def output_dim(self) -> int:
        return self.experts[0].output_dim

    @property
    def n_tasks(self) -> int:
        return self.output_dim // self.n_targets

    def forward(self, Z: Tensor) -> Tensor:
        expert_outputs = torch.stack([expert(Z) for expert in self.experts], dim=1)  # [B, E, O]
        gate_logits = self.gate(Z)                                                   # [B, E]
        gate_weights = F.softmax(gate_logits, dim=-1)                                # [B, E]

        # Save for later inspection
        self._last_gate_weights = gate_weights.detach()

        Y = torch.einsum("be,bed->bd", gate_weights, expert_outputs)
        return self.output_transform(Y)

    train_step = forward

    def encode(self, Z: Tensor, i: int) -> Tensor:
        return self.experts[0][:i](Z)
