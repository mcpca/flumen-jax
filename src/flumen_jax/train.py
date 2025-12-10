import jax
from jax import numpy as jnp

import equinox
import optax

import sys
from typing import cast, Iterator

from torch.utils.data import DataLoader

from .model import (
    Flumen,
    BatchedOutput,
    BatchedState,
    BatchedRNNInput,
    BatchedTimeIncrement,
    BatchLengths,
)

Inputs = tuple[
    BatchedState, BatchedRNNInput, BatchedTimeIncrement, BatchLengths
]


def evaluate(dataloader: DataLoader, model: Flumen) -> float:
    total_loss = 0.0
    for y, inputs in torch2jax(dataloader):
        total_loss += compute_loss(model, inputs, y).item()
    return total_loss / len(dataloader.dataset)  # type: ignore


def torch2jax(dataloader: DataLoader) -> Iterator[tuple[BatchedOutput, Inputs]]:
    for y, x0, rnn_input, tau, lengths in dataloader:
        yield (
            y.numpy(),
            (
                x0.numpy(),
                rnn_input.numpy(),
                tau.numpy(),
                lengths.numpy(),
            ),
        )


@equinox.filter_jit
def compute_loss(model: Flumen, inputs: Inputs, y: jax.Array):
    x, rnn_input, tau, batch_lens = inputs
    y_pred = model(x, rnn_input, tau, batch_lens)
    loss_val = jnp.sum(jnp.square(y - y_pred))

    return loss_val


@equinox.filter_jit
def train_step(
    model: Flumen,
    inputs: Inputs,
    y: BatchedOutput,
    optimiser: optax.GradientTransformation,
    state: optax.OptState,
) -> tuple[Flumen, optax.OptState, jax.Array]:
    loss, grad = equinox.filter_value_and_grad(compute_loss)(model, inputs, y)
    update, new_state = optimiser.update(grad, state, cast(optax.Params, model))
    model = equinox.apply_updates(model, update)

    return model, new_state, loss


class MetricMonitor:
    patience: int
    atol: float
    rtol: float

    def __init__(self, patience: int, rtol: float, atol: float):
        self._best = float("inf")
        self._counter = 0
        self.patience = patience
        self.rtol = rtol
        self.atol = atol
        self._is_best = False

    def update(self, metric: float) -> bool:
        if self.better(metric):
            self._best = metric
            self._is_best = True
            self._counter = 0
            return False

        self._is_best = False
        self._counter += 1

        if self._counter > self.patience:
            self._counter = 0
            return True

        return False

    def better(self, metric: float) -> bool:
        return metric + self.atol < (1 - self.rtol) * self._best

    @property
    def is_best(self):
        return self._is_best

    @property
    def best_metric(self):
        return self._best


def reduce_learning_rate(state: optax.OptState, factor: float, eps: float):
    curr_lr = state.hyperparams["learning_rate"]  # type:ignore
    new_lr = curr_lr / factor

    if curr_lr - new_lr > eps:
        state.hyperparams["learning_rate"] = new_lr  # type: ignore
        print(
            f"Learning rate reduced to {state.hyperparams['learning_rate']:.2e}",  # type: ignore
            file=sys.stderr,
        )
