import os
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import time
from typing import cast

import equinox
import jax
import numpy as np
import yaml
from flumen import TrajectoryDataset
from jax import random as jrd
from jaxtyping import PRNGKeyArray

import wandb
from flumen_jax.dataloader import NumPyDataset, NumPyLoader
from flumen_jax.train import (
    MetricMonitor,
    evaluate,
    reduce_learning_rate,
    train_step,
)
from flumen_jax.utils import (
    TrainConfig,
    adam,
    make_model,
    make_model_dir,
    prepare_model_saving,
    print_header,
    print_losses,
)

TRAIN_CONFIG: TrainConfig = {
    "batch_size": 128,
    "feature_dim": 24,
    "encoder_hsz": 128,
    "decoder_hsz": 128,
    "learning_rate": 1e-3,
    "n_epochs": 500,
    "sched_factor": 2,
    "sched_patience": 10,
    "sched_rtol": 1e-4,
    "sched_eps": 1e-8,
    "es_patience": 20,
    "es_atol": 5e-5,
}

DEFAULT_JAX_SEED = 0
DEFAULT_NUMPY_KEY_SEED = 3520756


def handle_seeds() -> tuple[PRNGKeyArray, int, int, int, int | None]:
    model_key_seed_str = os.environ.get("FLUMEN_JAX_SEED")
    if not model_key_seed_str:
        print("No model key seed found, using default", file=sys.stderr)
        model_key_seed = DEFAULT_JAX_SEED
    else:
        model_key_seed = int(model_key_seed_str)

    model_key = jrd.key(model_key_seed)

    array_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if array_id_str:
        array_id = int(array_id_str)
        if array_id > 1:
            *_, model_key = jrd.split(model_key, array_id)
    else:
        array_id = None

    numpy_seed_str = os.environ.get("FLUMEN_NUMPY_KEY_SEED")
    if not numpy_seed_str:
        print("No NumPy key seed found, using default", file=sys.stderr)
        numpy_key_seed = DEFAULT_NUMPY_KEY_SEED
    else:
        numpy_key_seed = int(numpy_seed_str)

    numpy_key = jrd.key(numpy_key_seed)

    if array_id and array_id > 1:
        *_, numpy_key = jrd.split(numpy_key, array_id)

    numpy_seed = int(
        jrd.randint(numpy_key, (1,), minval=0, maxval=32768).item()
    )

    return model_key, model_key_seed, numpy_key_seed, numpy_seed, array_id


def main():
    print("JAX device list:", jax.devices(), file=sys.stderr)

    ap = ArgumentParser()
    ap.add_argument("load_path", type=str, help="Path to trajectory dataset")
    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")
    ap.add_argument(
        "--model_log_rate",
        type=int,
        default=15,
        help="Model will not be logged to W&B more often than every model_log_rate epochs.",
    )
    ap.add_argument("--outdir", type=str, default="./outputs")

    args = ap.parse_args()
    data_path = Path(args.load_path)

    with data_path.open("rb") as f:
        data = pickle.load(f)

    first_name, full_name, timestamp = prepare_model_saving(args.name)

    run = wandb.init(
        project="flumen-jax", config=cast(dict, TRAIN_CONFIG), name=full_name
    )
    model_save_dir = make_model_dir(
        Path(args.outdir), first_name, full_name + "_" + run.id
    )
    model_name = f"flumen_jax-{timestamp}-{data_path.stem}-{run.id}"

    model_key, model_key_seed, numpy_key_seed, numpy_seed, array_id = (
        handle_seeds()
    )
    wandb.config["model_key_seed"] = model_key_seed
    wandb.config["numpy_key_seed"] = numpy_key_seed
    wandb.config["numpy_seed"] = numpy_key_seed
    if array_id:
        wandb.config["array_id"] = array_id

    np.random.seed(numpy_seed)

    train_data = NumPyDataset(TrajectoryDataset(data["train"]))
    val_data = NumPyDataset(TrajectoryDataset(data["val"]))

    bs = TRAIN_CONFIG["batch_size"]
    train_dl = NumPyLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = NumPyLoader(val_data, batch_size=bs, shuffle=False)

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

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", "w") as f:
        yaml.dump(model_metadata, f)

    model = make_model(model_args, model_key)

    optim = adam(run.config["learning_rate"])
    state = optim.init(equinox.filter(model, equinox.is_inexact_array))

    lr_monitor = MetricMonitor(
        patience=run.config["sched_patience"],
        rtol=run.config["sched_rtol"],
        atol=0.0,
    )

    early_stop = MetricMonitor(
        patience=run.config["es_patience"],
        atol=run.config["es_atol"],
        rtol=0.0,
    )

    train_loss = evaluate(train_dl, model)
    val_loss = evaluate(val_dl, model)

    lr_monitor.update(val_loss)
    early_stop.update(val_loss)

    print_header()
    print_losses(
        0,
        train_loss,
        val_loss,
        early_stop.best_metric,
    )

    last_log_epoch = 0
    train_time = time()
    for epoch in range(run.config["n_epochs"]):
        for y, inputs in train_dl:
            model, state, _ = train_step(
                model,
                inputs,
                y,
                optim,
                state,
            )

        train_loss = evaluate(train_dl, model)
        val_loss = evaluate(val_dl, model)
        stop = early_stop.update(val_loss)

        print_losses(
            epoch + 1,
            train_loss,
            val_loss,
            early_stop.best_metric,
        )

        if early_stop.is_best:
            equinox.tree_serialise_leaves(model_save_dir / "leaves.eqx", model)

            if epoch >= last_log_epoch + args.model_log_rate:
                run.log_model(model_save_dir.as_posix(), name=model_name)
                last_log_epoch = epoch

            run.summary["best_train"] = train_loss
            run.summary["best_val"] = val_loss
            run.summary["best_epoch"] = epoch + 1

        wandb.log(
            {
                "time": time() - train_time,
                "epoch": epoch + 1,
                "lr": state.hyperparams["learning_rate"],  # type: ignore
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if stop:
            print("Early stop.", file=sys.stderr)
            break

        update_lr = lr_monitor.update(val_loss)
        if update_lr:
            reduce_learning_rate(
                state,
                factor=run.config["sched_factor"],
                eps=run.config["sched_eps"],
            )

    train_time = int(time() - train_time)
    run.summary["train_time"] = train_time
    print(f"Training took {train_time} sec.")

    # Log best model
    run.log_model(model_save_dir.as_posix(), name=model_name, aliases=["best"])

    # Load best model and compute test loss
    model = equinox.tree_deserialise_leaves(
        model_save_dir / "leaves.eqx", model
    )

    test_data = NumPyDataset(TrajectoryDataset(data["test"]))
    test_dl = NumPyLoader(test_data, batch_size=bs, shuffle=False)
    test_loss = evaluate(test_dl, model)
    run.summary["best_test"] = test_loss
    print(f"Test loss: {test_loss:.5e}")


if __name__ == "__main__":
    main()
