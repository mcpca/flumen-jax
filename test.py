import torch

import jax
from jax import random as jrd
from jax import numpy as jnp

import equinox
import optax

from model import Flumen

from flumen import TrajectoryDataset
from flumen.utils import get_batch_inputs
from torch.utils.data import DataLoader

from argparse import ArgumentParser
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


TRAIN_CONFIG = {
    "batch_size": 128,
    "feature_dim": 16,
    "encoder_hsz": 20,
    "decoder_hsz": 20,
    "learning_rate": 7e-4,
    "n_epochs": 100,
}


def torch2jax(dataloader):
    for example in dataloader:
        yield (tuple(jnp.array(v.numpy()) for v in example))


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

    for epoch in range(TRAIN_CONFIG["n_epochs"]):
        train_loss = 0.0
        for x0, y, rnn_input, tau, lens in torch2jax(train_dl):
            model, state, batch_loss = train_step(
                model, (x0, rnn_input, tau, lens), y, optim, state
            )
            train_loss += batch_loss
        print(f"{epoch + 1}:: {train_loss / len(train_dl)}")

    x0 = torch.tensor([1.0, 1.0])
    u = torch.randn((31, 1))
    times = torch.linspace(0.0, 15.0, 100).unsqueeze(-1)
    x0, rnn_input, tau, lengths = get_batch_inputs(
        x0, times, u, train_data.delta, pack_inputs=False
    )
    x0 = jnp.array(x0.numpy())
    rnn_input = jnp.array(rnn_input.numpy())
    tau = jnp.array(tau.numpy())
    lengths = jnp.array(lengths.numpy())

    y = model(x0, rnn_input, tau, lengths)

    plt.plot(times.numpy(), y)
    plt.show()


@equinox.filter_value_and_grad
def compute_loss(model: Flumen, inputs, y):
    x, rnn_input, tau, batch_lens = inputs

    y_pred = model(x, rnn_input, tau, batch_lens)

    loss_val = jnp.mean(jnp.square(y - y_pred))

    return loss_val


@equinox.filter_jit
def train_step(
    model: equinox.Module,
    inputs,
    y,
    optimiser: optax.GradientTransformation,
    state: optax.OptState,
) -> tuple[equinox.Module, optax.OptState, jax.Array]:
    loss, grad = compute_loss(model, inputs, y)
    update, new_state = optimiser.update(grad, state, model)
    model = equinox.apply_updates(model, update)

    return model, new_state, loss


def make_model(args: dict) -> Flumen:
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
