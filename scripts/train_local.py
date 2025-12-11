import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import time

import equinox
import jax
from jax import random as jrd
import torch
import yaml
from flumen import TrajectoryDataset
from torch.utils.data import DataLoader

from flumen_jax.train import (
    MetricMonitor,
    evaluate,
    reduce_learning_rate,
    torch2jax,
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
    "feature_dim": 64,
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
    "torch_seed": 3520756,
    "model_key_seed": 345098145,
}

torch.manual_seed(seed=TRAIN_CONFIG["torch_seed"])


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

    first_name, full_name, _ = prepare_model_saving(args.name)
    model_save_dir = make_model_dir(Path(args.outdir), first_name, full_name)

    # Save local copy of metadata
    with open(model_save_dir / "metadata.yaml", "w") as f:
        yaml.dump(model_metadata, f)

    key = jrd.key(TRAIN_CONFIG.get("model_key_seed", 0))
    model = make_model(model_args, key)

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


if __name__ == "__main__":
    main()
