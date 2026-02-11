import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import time

import equinox
import jax
import numpy as np
import yaml
from flumen import TrajectoryDataset
from jax import random as jrd

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
    init_last_layer_bias,
)

TRAIN_CONFIG: TrainConfig = {
    "batch_size": 128,
    "feature_dim": 16,
    "encoder_hsz": 16,
    "encoder_depth": 2,
    "decoder_hsz": 16,
    "decoder_depth": 2,
    "learning_rate": 1e-3,
    "n_epochs": 500,
    "sched_factor": 2,
    "sched_patience": 10,
    "sched_rtol": 1e-4,
    "sched_eps": 1e-8,
    "es_patience": 20,
    "es_atol": 5e-5,
    "init_last_layer_bias": True,
    "numpy_seed": 3520756,
    "model_key_seed": 345098145,
}

np.random.seed(TRAIN_CONFIG["numpy_seed"])


def main():
    print("JAX device list:", jax.devices(), file=sys.stderr)

    ap = ArgumentParser()
    ap.add_argument("load_path", type=str, help="Path to trajectory dataset")
    ap.add_argument("name", type=str, nargs="+", help="Name of the experiment.")
    ap.add_argument("--outdir", type=str, default="./outputs")

    args = ap.parse_args()
    data_path = Path(args.load_path)

    with data_path.open("rb") as f:
        data = pickle.load(f)

    train_data = NumPyDataset(TrajectoryDataset(data["train"]))
    val_data = NumPyDataset(TrajectoryDataset(data["val"]))

    bs = TRAIN_CONFIG["batch_size"]
    train_dl = NumPyLoader(train_data, batch_size=bs, shuffle=True)
    val_dl = NumPyLoader(
        val_data, batch_size=bs, shuffle=False, skip_last=False
    )

    model_args = {
        "state_dim": train_data.state_dim,
        "control_dim": train_data.control_dim,
        "output_dim": train_data.output_dim,
        "feature_dim": TRAIN_CONFIG["feature_dim"],
        "encoder_hsz": TRAIN_CONFIG["encoder_hsz"],
        "decoder_hsz": TRAIN_CONFIG["decoder_hsz"],
        "encoder_depth": TRAIN_CONFIG["encoder_depth"],
        "decoder_depth": TRAIN_CONFIG["decoder_depth"],
    }

    model_metadata = {
        "args": model_args,
        "framework": "equinox",
        "data_path": data_path.absolute().as_posix(),
        "data_settings": data["settings"],
        "data_args": data["args"],
    }

    first_name, full_name, _ = prepare_model_saving(args.name)
    model_save_dir = make_model_dir(Path(args.outdir), first_name, full_name)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", "w") as f:
        yaml.dump(model_metadata, f)

    key = jrd.key(TRAIN_CONFIG.get("model_key_seed", 0))
    model = make_model(model_args, key)

    y_train_var = np.var(train_data.y, axis=0)
    print(
        f"Trace of output variance in training data: {np.sum(y_train_var):.2f}",
        file=sys.stderr,
    )

    if TRAIN_CONFIG["init_last_layer_bias"]:
        y_train_mean = np.mean(train_data.y, axis=0)
        model = init_last_layer_bias(model, y_train_mean, sum=True)

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

    flat_model, model_treedef = jax.tree_util.tree_flatten(model)
    flat_state, state_treedef = jax.tree_util.tree_flatten(state)

    val_loss = evaluate(val_dl, flat_model, model_treedef)
    lr_monitor.update(val_loss)
    early_stop.update(val_loss)

    print_header()
    print_losses(
        0,
        evaluate(train_dl, flat_model, model_treedef),
        val_loss,
        early_stop.best_metric,
    )

    train_time = time()
    for epoch in range(TRAIN_CONFIG["n_epochs"]):
        train_loss = 0.0
        for y, inputs in train_dl:
            flat_model, flat_state, loss = train_step(
                flat_model,
                model_treedef,
                inputs,
                y,
                optim,
                flat_state,
                state_treedef,
            )
            train_loss += loss.item()
        train_loss /= len(train_dl)

        val_loss = evaluate(val_dl, flat_model, model_treedef)
        stop = early_stop.update(val_loss)

        print_losses(
            epoch + 1,
            train_loss,
            val_loss,
            early_stop.best_metric,
        )

        if early_stop.is_best:
            model = jax.tree_util.tree_unflatten(model_treedef, flat_model)
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

    model = equinox.tree_deserialise_leaves(
        model_save_dir / "leaves.eqx", model
    )
    test_data = NumPyDataset(TrajectoryDataset(data["test"]))
    test_dl = NumPyLoader(
        test_data, batch_size=bs, shuffle=False, skip_last=False
    )
    test_loss = evaluate(test_dl, *jax.tree_util.tree_flatten(model))
    print(f"Test loss: {test_loss:.5e}")


if __name__ == "__main__":
    main()
