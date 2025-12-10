import torch
from torch.utils.data import DataLoader

from jax import random as jrd

import equinox
import optax

import yaml

from flumen_jax import Flumen
from flumen_jax.train import (
    torch2jax,
    evaluate,
    train_step,
    MetricMonitor,
    reduce_learning_rate,
)

from flumen import TrajectoryDataset

from argparse import ArgumentParser
import pickle
from pathlib import Path
import sys
from time import time
from typing import TypedDict
import datetime
import re


class TrainConfig(TypedDict):
    batch_size: int
    feature_dim: int
    encoder_hsz: int
    decoder_hsz: int
    learning_rate: float
    n_epochs: int
    sched_factor: int
    sched_patience: int
    sched_rtol: float
    sched_eps: float
    es_patience: int
    es_atol: float
    torch_seed: int


TRAIN_CONFIG: TrainConfig = {
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


def get_timestamp() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%y%m%d_%H%M")

    return ts


def prepare_model_saving(names: list[str], outdir: Path):
    first_name = names[0]
    timestamp = get_timestamp()
    full_name = "_".join([timestamp] + names)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

    model_save_dir = Path(outdir / f"{first_name}/{full_name}")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to directory {model_save_dir}", file=sys.stderr)

    return model_save_dir


def main():
    ap = ArgumentParser()
    ap.add_argument("load_path", type=str, help="Path to trajectory dataset")
    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")
    ap.add_argument("--outdir", type=str, default="./outputs")

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

    model_metadata = {
        "args": model_args,
        "framework": "equinox",
        "data_path": data_path.absolute().as_posix(),
        "data_settings": data["settings"],
        "data_args": data["args"],
    }

    model_save_dir = prepare_model_saving(args.name, Path(args.outdir))

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", "w") as f:
        yaml.dump(model_metadata, f)

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

        if early_stop.best_metric:
            equinox.tree_serialise_leaves(model_save_dir / "leaves.eqx", model)

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
    train_time = int(time() - train_time)
    print(f"Training took {train_time} sec.")


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
