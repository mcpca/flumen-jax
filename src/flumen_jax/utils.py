import datetime
import re
import sys
from pathlib import Path
from typing import TypedDict

import optax
from jax import random as jrd

from flumen_jax import Flumen


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
    model_key_seed: int


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


def get_timestamp() -> str:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    ts = now.strftime("%y%m%d_%H%M")

    return ts


def prepare_model_saving(names: list[str]) -> tuple[str, str, str]:
    first_name = names[0]
    timestamp = get_timestamp()
    full_name = "_".join([timestamp] + names)
    full_name = re.sub("[^a-zA-Z0-9_-]", "_", full_name)

    return first_name, full_name, timestamp


def make_model_dir(outdir: Path, first_name: str, full_name: str) -> Path:
    model_save_dir = Path(outdir / f"{first_name}/{full_name}")
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to directory {model_save_dir}", file=sys.stderr)

    return model_save_dir


@optax.inject_hyperparams
def adam(learning_rate):
    return optax.adam(learning_rate)


def make_model(args: dict[str, int], key_seed: int) -> Flumen:
    key = jrd.key(key_seed)

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
