import datetime
import re
import sys
from pathlib import Path
from typing import NotRequired, TypedDict

import equinox
import optax
from jaxtyping import PRNGKeyArray

from flumen_jax import Flumen
from flumen_jax.typing import Output


class TrainConfig(TypedDict):
    batch_size: int
    feature_dim: int
    encoder_hsz: int
    encoder_depth: int
    decoder_hsz: int
    decoder_depth: int
    learning_rate: float
    n_epochs: int
    sched_factor: int
    sched_patience: int
    sched_rtol: float
    sched_eps: float
    es_patience: int
    es_atol: float
    init_last_layer_bias: bool
    numpy_seed: NotRequired[int]
    model_key_seed: NotRequired[int]


def print_header():
    header_msg = (
        f"{'Epoch':>5} :: {'Loss (Train)':>16} :: "
        f"{'Loss (Val)':>16} :: {'Best (Val)':>16}"
    )

    print(header_msg)
    print("=" * len(header_msg))


def print_losses(
    epoch: int,
    train: float,
    val: float,
    best_val_yet: float,
):
    print(
        f"{epoch:>5d} :: {train:>16.5e} :: {val:>16.5e} :: "
        f"{best_val_yet:>16.5e}"
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


def make_model(args: dict[str, int], key: PRNGKeyArray) -> Flumen:
    model = Flumen(**args, key=key)
    return model


def init_last_layer_bias(model: Flumen, val: Output, sum=True) -> Flumen:
    if sum:
        val = val + model.decoder.layers[-1].bias
    model = equinox.tree_at(lambda m: m.decoder.layers[-1].bias, model, val)
    return model
