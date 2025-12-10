import torch
from torch.utils.data import DataLoader

from jax import random as jrd
from jax import numpy as jnp

import equinox
import optax

from flumen_jax import Flumen
from flumen_jax.train import (
    torch2jax,
    evaluate,
    train_step,
    MetricMonitor,
    reduce_learning_rate,
)

from flumen import TrajectoryDataset
from flumen.utils import get_batch_inputs

from argparse import ArgumentParser
import pickle
from pathlib import Path
import sys
from time import time

import matplotlib.pyplot as plt


TRAIN_CONFIG = {
    "batch_size": 128,
    "feature_dim": 16,
    "encoder_hsz": 20,
    "decoder_hsz": 20,
    "learning_rate": 7e-4,
    "n_epochs": 200,
    "sched_factor": 2,
    "sched_patience": 10,
    "sched_rtol": 1e-4,
    "sched_eps": 1e-8,
    "es_patience": 20,
    "es_atol": 5e-5,
    "torch_seed": 3520756,
}

torch.manual_seed(seed=TRAIN_CONFIG["torch_seed"])


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
        f"{epoch + 1:>5d} :: {train:>16.5e} :: {val:>16.5e} :: "
        f"{test:>16.5e} :: {best_val_yet:>16.5e}"
    )


@optax.inject_hyperparams
def adam(learning_rate):
    return optax.adam(learning_rate)


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

    optim = adam(TRAIN_CONFIG["learning_rate"])
    state = optim.init(equinox.filter(model, equinox.is_inexact_array))

    lr_monitor = MetricMonitor(
        patience=TRAIN_CONFIG["sched_patience"],
        rtol=TRAIN_CONFIG["sched_rtol"],
        atol=0.0,
    )

    early_stop = MetricMonitor(
        patience=TRAIN_CONFIG["es_patience"],
        atol=TRAIN_CONFIG["es_atol"],
        rtol=0.0,
    )

    val_loss = evaluate(val_dl, model)
    lr_monitor.update(val_loss)
    early_stop.update(val_loss)

    print_header()
    print_losses(
        0,
        evaluate(train_dl, model),
        val_loss,
        evaluate(test_dl, model),
        early_stop.best_metric,
    )

    train_time = time()
    for epoch in range(TRAIN_CONFIG["n_epochs"]):
        for y, inputs in torch2jax(train_dl):
            model, state, _ = train_step(
                model,
                inputs,
                y,
                optim,
                state,
            )

        val_loss = evaluate(val_dl, model)
        stop = early_stop.update(val_loss)

        print_losses(
            epoch + 1,
            evaluate(train_dl, model),
            val_loss,
            evaluate(test_dl, model),
            early_stop.best_metric,
        )

        if stop:
            print("Early stop.", file=sys.stderr)
            break

        update_lr = lr_monitor.update(val_loss)
        if update_lr:
            reduce_learning_rate(
                state,
                factor=TRAIN_CONFIG["sched_factor"],
                eps=TRAIN_CONFIG["sched_eps"],
            )
    train_time = time() - train_time
    print(f"Training took {train_time} sec.")

    x0 = torch.tensor([1.0, 1.0])
    u = torch.randn((31, 1))
    times = torch.linspace(0.0, 15.0, 100).unsqueeze(-1)
    x0, rnn_input, tau, lengths = get_batch_inputs(
        x0, times, u, train_data.delta
    )
    x0 = x0.numpy()
    rnn_input = rnn_input.numpy()
    tau = tau.numpy()
    lengths = lengths.numpy()

    y = model(x0, rnn_input, tau, lengths)

    plt.plot(times.numpy(), y)
    plt.show()


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
