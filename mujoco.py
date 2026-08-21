"""ODE MuJoCo forecasting runner used by ODEvsSDE.py.

Install this file as ``benchmark_forecasting/mujoco.py``. Input corruption is
deliberately absent here: this runner trains once on the baseline dataset,
whilst ODEvsSDE.py applies test-only corruption after training.
"""

import os
import random
from random import SystemRandom

import numpy as np
import torch
from tensorboardX import SummaryWriter

import common
import datasets
from parse import parse_args


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
args = parse_args()


def main(
    manual_seed=args.seed,
    intensity=args.intensity,
    device="cuda",
    max_epochs=args.epoch,
    missing_rate=args.missing_rate,
    pos_weight=10,
    *,
    model_name=args.model,
    hidden_channels=args.h_channels,
    hidden_hidden_channels=args.hh_channels,
    num_hidden_layers=args.layers,
    ode_hidden_hidden_channels=args.ode_hidden_hidden_channels,
    lr=args.lr,
    c1=args.c1,
    c2=args.c2,
    weight_decay=args.weight_decay,
    dry_run=False,
    method=args.method,
    step_mode=args.step_mode,
    time_seq=args.time_seq,
    y_seq=args.y_seq,
    dataset_name=args.dataset_name,
    batch_size=args.batch_size,
    **kwargs,
):
    project_dir = os.path.dirname(os.path.abspath(__file__))

    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.random.manual_seed(manual_seed)

    dataset = datasets.get_dataset(dataset_name)
    num_input_features, num_output_features = datasets.feature_dimensions(dataset)

    time_augment = intensity
    (
        times,
        train_dataloader,
        val_dataloader,
        test_dataloader,
    ) = dataset.get_data(
        batch_size=batch_size,
        missing_rate=missing_rate,
        append_time=time_augment,
        time_seq=time_seq,
        y_seq=y_seq,
        loader_seed=manual_seed,
    )

    output_time = y_seq
    experiment_id = int(SystemRandom().random() * 100000)

    auxiliary_dir = os.path.join(project_dir, "h_0")
    os.makedirs(auxiliary_dir, exist_ok=True)
    auxiliary_file = os.path.join(
        auxiliary_dir,
        f"{dataset_name}_{experiment_id}.npy",
    )

    input_channels = int(time_augment) + num_input_features
    folder_name = dataset_name.capitalize()
    test_name = "step_" + "_".join(
        str(value) for value in vars(args).values()
    ) + "_" + str(experiment_id)
    result_folder = os.path.join(project_dir, "tensorboard")
    writer = SummaryWriter(
        f"{result_folder}/runs/{folder_name}/{test_name}"
    )

    make_model = common.make_model(
        model_name,
        input_channels,
        num_output_features,
        hidden_channels,
        hidden_hidden_channels,
        ode_hidden_hidden_channels,
        num_hidden_layers,
        auxiliary_file,
        use_intensity=intensity,
        initial=True,
        output_time=output_time,
    )

    name = None if dry_run else dataset_name.capitalize()
    solver_kwargs = dict(kwargs)
    solver_kwargs["method"] = method

    return common.main_forecasting(
        name=name,
        model_name=model_name,
        times=times,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        make_model=make_model,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        writer=writer,
        file=auxiliary_file,
        kwargs=solver_kwargs,
        step_mode=step_mode,
        c1=c1,
        c2=c2,
        pos_weight=torch.tensor(pos_weight),
    )


if __name__ == "__main__":
    main(method=args.method)
