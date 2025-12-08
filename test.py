import torch

import jax
from jax import random as jrd
from jax import numpy as jnp

import equinox
import optax

from model import (
    Flumen,
    BatchedOutput,
    BatchedState,
    BatchedRNNInput,
    BatchedTimeIncrement,
    BatchLengths,
)

from flumen import TrajectoryDataset
from flumen.utils import get_batch_inputs
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import pickle
from pathlib import Path
from typing import cast, Iterator

import matplotlib.pyplot as plt


TRAIN_CONFIG = {
    "batch_size": 128,
    "feature_dim": 16,
    "encoder_hsz": 20,
    "decoder_hsz": 20,
    "learning_rate": 7e-4,
    "n_epochs": 100,
}

Inputs = tuple[
    BatchedState, BatchedRNNInput, BatchedTimeIncrement, BatchLengths
]


def torch2jax(dataloader: DataLoader) -> Iterator[tuple[BatchedOutput, Inputs]]:
    for y, x0, rnn_input, tau, lengths in dataloader:
        yield (
            jnp.array(y.numpy()),
            (
                jnp.array(x0.numpy()),
                jnp.array(rnn_input.numpy()),
                jnp.array(tau.numpy()),
                jnp.array(lengths.numpy()),
            ),
        )


def print_header():
    header_msg = (
        f"{'Epoch':>5} :: {'Loss (Train)':>16} :: "
        f"{'Loss (Val)':>16} :: {'Loss (Test)':>16} :: {'Best (Val)':>16}"
    )

    print(header_msg)
    print("=" * len(header_msg))


def print_losses(
    epoch: int,
    train: float,
    val: float,
    test: float,
    best_val_yet: float,
):
    print(
        f"{epoch + 1:>5d} :: {train:>16e} :: {val:>16e} :: "
        f"{test:>16e} :: {best_val_yet:>16e}"
    )


def main():
    ap = ArgumentParser()
    ap.add_argument("load_path", type=str, help="Path to trajectory dataset")

    args = ap.parse_args()
    data_path = Path(args.load_path)

    with data_path.open("rb") as f:
        data = pickle.load(f)

    train_data = TrajectoryDataset(data["train"])
    val_data = TrajectoryDataset(data["val"])
    test_data = TrajectoryDataset(data["test"])

    bs = TRAIN_CONFIG["batch_size"]
    train_dl = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=bs, shuffle=True)

    model_args = {
        "state_dim": train_data.state_dim,
        "control_dim": train_data.control_dim,
        "output_dim": train_data.output_dim,
        "feature_dim": TRAIN_CONFIG["feature_dim"],
        "encoder_hsz": TRAIN_CONFIG["encoder_hsz"],
        "decoder_hsz": TRAIN_CONFIG["decoder_hsz"],
    }

    model = make_model(model_args)
    optim = optax.adam(TRAIN_CONFIG["learning_rate"])

    state = optim.init(equinox.filter(model, equinox.is_inexact_array))

    print_header()
    print_losses(
        0,
        evaluate(train_dl, model),
        evaluate(val_dl, model),
        evaluate(test_dl, model),
        0.0,
    )

    for epoch in range(TRAIN_CONFIG["n_epochs"]):
        for y, inputs in torch2jax(train_dl):
            model, state, _ = train_step(
                model,
                inputs,
                y,
                optim,
                state,
            )

        print_losses(
            epoch + 1,
            evaluate(train_dl, model),
            evaluate(val_dl, model),
            evaluate(test_dl, model),
            0.0,
        )

    x0 = torch.tensor([1.0, 1.0])
    u = torch.randn((31, 1))
    times = torch.linspace(0.0, 15.0, 100).unsqueeze(-1)
    x0, rnn_input, tau, lengths = get_batch_inputs(
        x0, times, u, train_data.delta
    )
    x0 = jnp.array(x0.numpy())
    rnn_input = jnp.array(rnn_input.numpy())
    tau = jnp.array(tau.numpy())
    lengths = jnp.array(lengths.numpy())

    y = model(x0, rnn_input, tau, lengths)

    plt.plot(times.numpy(), y)
    plt.show()


def evaluate(dataloader: DataLoader, model: Flumen) -> float:
    total_loss = 0.0
    for y, inputs in torch2jax(dataloader):
        total_loss += compute_loss(model, inputs, y).item()
    return total_loss / len(dataloader)


@equinox.filter_jit
def compute_loss(model: Flumen, inputs: Inputs, y: jax.Array):
    x, rnn_input, tau, batch_lens = inputs
    y_pred = model(x, rnn_input, tau, batch_lens)
    loss_val = jnp.mean(jnp.square(y - y_pred))

    return loss_val


@equinox.filter_jit
def train_step(
    model: Flumen,
    inputs: Inputs,
    y: jax.Array,
    optimiser: optax.GradientTransformation,
    state: optax.OptState,
) -> tuple[Flumen, optax.OptState, jax.Array]:
    loss, grad = equinox.filter_value_and_grad(compute_loss)(model, inputs, y)
    update, new_state = optimiser.update(grad, state, cast(optax.Params, model))
    model = equinox.apply_updates(model, update)

    return model, new_state, loss


def make_model(args: dict[str, int]) -> Flumen:
    key = jrd.key(345098145)

    model = Flumen(
        args["state_dim"],
        args["control_dim"],
        args["output_dim"],
        args["feature_dim"],
        args["encoder_hsz"],
        args["decoder_hsz"],
        key=key,
    )

    return model


if __name__ == "__main__":
    main()
