""" Adapted from https://github.com/mir-group/nequip
"""

""" Train a network."""

# This is a weird hack to avoid Intel MKL issues on the cluster when this is called as a subprocess of a process that has itself initialized PyTorch.
# Since numpy gets imported later anyway for dataset stuff, this shouldn't affect performance.
import torch

import logging
from typing import Tuple
from functools import partial
import warnings
warnings.filterwarnings("ignore")
from geqtrain.utils.test import assert_AtomicData_equivariant
from geqtrain.scripts._logger import set_up_script_logger
from geqtrain.utils._global_options import _set_global_options
from geqtrain.utils import Config, load_file
from geqtrain.model import model_from_config
from pathlib import Path
from os.path import isdir
import logging
import argparse
import shutil
from geqtrain.train import (
    setup_distributed_training,
    cleanup_distributed_training,
    configure_dist_training,
    instanciate_train_val_dsets,
    load_trainer_and_model,
)

from source.diffusion_utils.sample_with_diffusion import ddpm_sampling, ddim_sampling


def parse_command_line(args=None) -> Tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser(
        description="Train (or restart training of) a model."
    )
    parser.add_argument(
        "config", help="YAML file configuring the model, dataset, and other options"
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Device on which to run the training. could be either 'cpu' or 'cuda[:n]'",
        default=None,
    )
    parser.add_argument(
        "--equivariance-test",
        help="test the model's equivariance before training on first frame of the validation dataset",
        action="store_true",
    )
    parser.add_argument(
        "--grad-anomaly-mode",
        help="enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production training!",
        action="store_true",
    )
    parser.add_argument(
        "-ws", # if wd is present then use_dt
        "--world-size",
        help="Number of available GPUs for Distributed Training",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-ma",
        "--master-addr",
        help="set MASTER_ADDR environment variable for Distributed Training",
        default='localhost',
    )
    parser.add_argument(
        "-mp",
        "--master-port",
        help="set MASTER_PORT environment variable for Distributed Training",
        default='rand',
    )
    args = parser.parse_args(args=args)

    # Check consistency
    if args.world_size is not None:
        if args.device is not None:
            raise argparse.ArgumentError("Cannot specify device when using Distributed Training")
        if args.equivariance_test:
            raise argparse.ArgumentError("You can run Equivariance Test on single CPU/GPU only")

    config = Config.from_file(args.config)
    flags = ("device", "equivariance_test", "grad_anomaly_mode")
    config.update({flag: getattr(args, flag) for flag in flags if getattr(args, flag) is not None})
    config.update({"use_dt": args.world_size is not None})
    return args, config


def check_for_config_updates(config):
    """
    Compares the provided configuration with a previously saved configuration and updates
    modifiable parameters. Ensures consistency between the current configuration and the
    saved trainer state.

    Args:
        config (dict): The current configuration dictionary containing training parameters.

    Returns:
        tuple: A tuple containing:
            - config (Config): Updated configuration object excluding certain keys.
            - progress_config (Config): Configuration object including progress-related keys.

    Raises:
        ValueError: If attempting to restart a fine-tuning run or if there are mismatches
                    in non-modifiable parameters between the current and saved configurations.

    Notes:
        - The function checks for differences between the current configuration and the
          saved configuration (`trainer.pth` file). If a parameter is modifiable, it updates
          the saved configuration. Otherwise, it raises an error for mismatches.
        - The `modifiable_params` list defines which parameters can be updated during a
          restart.
        - Special handling is applied for parameters like `early_stop`, `filepath`,
          `dataset_list`, and `validation_dataset_list`.

    Assertions:
        - Ensures that the `filepath` in the current and saved configurations resolve to
          the same absolute path. This is critical to avoid inconsistencies in file paths
          that could lead to incorrect behavior.
        - Ensures that `dataset_list` and `validation_dataset_list` are lists and contain
          exactly one dataset. This restriction simplifies the handling of datasets and
          avoids unexpected behavior during training.
    """
    # compare old_config to config and update stop condition related arguments
    restart_file = f"{config['root']}/{config['run_name']}/trainer.pth"
    old_config = load_file(
        supported_formats=dict(torch=["pt", "pth"]),
        filename=restart_file,
        enforced_format="torch",
    )
    if old_config.get("fine_tune", False):
        raise ValueError("Cannot restart training of a fine-tuning run")

    modifiable_params = ["max_epochs", "loss_coeffs", "learning_rate", "device", "metrics_components",
                        "noise", "use_dt", "wandb", "batch_size", "validation_batch_size", "train_dloader_n_workers", "heads",
                        "val_dloader_n_workers", "dloader_prefetch_factor", "dataset_num_workers", "inmemory", "transforms",
                        "report_init_validation", "metrics_key", "max_gradient_norm", "dropout_edges", "optimizer_params", "head_wds", "seed",
                    ] # todo: "num_types" should be added here after moving binning functionality away from dataset creation

    for k,v in config.items():
        if v != old_config.get(k, ""):
            if k in modifiable_params:
                logging.info(f'Update "{k}" from {old_config[k]} to {v}')
                old_config[k] = v
            elif k.startswith("early_stop"):
                logging.info(f'Update "{k}" from {old_config[k]} to {v}')
                old_config[k] = v
            elif k == 'filepath':
                # assert Path(config[k]).resolve() == Path(old_config[k]).resolve()
                old_config[k] = v
            elif k in ['dataset_list', 'validation_dataset_list']:
                assert isinstance(v, list), "dataset_list/validation_dataset_list must be of type list"
                assert isinstance(old_config[k], list), "dataset_list/validation_dataset_list must be of type list"
                assert len(v) == 1, "for now only 1 dataset under dataset_list/validation_dataset_list is allowed"
                assert len(old_config[k]), "for now only 1 dataset under dataset_list/validation_dataset_list is allowed"
                new_dset_and_kwargs = v[0]
                old_dset_and_kwargs = old_config[k][0]
                for dlist_k in new_dset_and_kwargs.keys():
                    if dlist_k in modifiable_params:
                        continue
                    if new_dset_and_kwargs[dlist_k] != old_dset_and_kwargs[dlist_k]:
                        raise ValueError(f'Key "{k}" is different in config and the result trainer.pth file. Please double check')
            elif isinstance(v, type(old_config.get(k, ""))):
                raise ValueError(f'Key "{k}" is different in config and the result trainer.pth file. Please double check')

    config          = Config(old_config, exclude_keys=["state_dict", "progress"])
    progress_config = Config(old_config)
    return config, progress_config

def restart(rank, world_size, config: dict, train_dataset, validation_dataset, progress_config: dict):
    try:
        # Necessary for mp.spawn
        assert isinstance(config, dict), f"config must be of type Dict. It is of type {type(config)}"
        config = Config.from_dict(config)
        assert isinstance(progress_config, dict), f"progress_config must be of type Dict. It is of type {type(progress_config)}"
        progress_config = Config.from_dict(progress_config)

        if config.use_dt:
            setup_distributed_training(rank, world_size)

        trainer, model = load_trainer_and_model(rank, world_size, progress_config, is_restart=True)

        trainer.warmup_epochs = 0
        trainer.lr_scheduler_name = 'none'
        trainer.dataset_train = None
        trainer.dataset_val = None

        trainer.init_model(model=model)
        trainer.update_kwargs(config)

        # trainer.dataset_val[0]['node_types'].squeeze().to(device)
        trainer.dataset_val = [{'node_types':torch.tensor([6, 6, 8, 7, 6, 6, 6, 8, 7, 6], dtype=torch.int64)}]

        # ddpm_sampling(trainer) # ddpm_sampling(trainer, t_init:int = 0, condition_class:int = 0, guidance_scale:float = 7.0)
        n_samples = 50
        list_labels = [0,1,2,3,4,5,6,7]
        sampler = 'ddim'
        for label in list_labels:
            if sampler == 'ddim':
                ddim_sampling(
                    trainer,
                    # method="quadratic",
                    n_steps = 50,
                    # t_init:int = 0,
                    condition_class=label,
                    guidance_scale=7.0,
                    forced_log_dir='/home/nobilm@usi.ch/mydiff/data/alanine/alanine_generated_per_label',
                    iter_epochs=1,
                    n_samples=n_samples,
                )
            elif sampler == 'ddpm':
                ddpm_sampling(
                    trainer,
                    condition_class=label,
                    # guidance_scale:float = 7.0,
                    forced_log_dir='/home/nobilm@usi.ch/mydiff/data/alanine/alanine_generated_per_label',
                    n_samples=n_samples,
                    iter_epochs=1,
                )

    except KeyboardInterrupt:
        logging.info("Process manually stopped!")
    except Exception as e:
        logging.error(e)
        raise e
    finally:
        try:
            if config.get("use_dt", False):
                cleanup_distributed_training(rank)
        except:
            pass
    return

def main(args=None):
    args, config = parse_command_line(args)
    found_restart_file = isdir(f"{config.root}/{config.run_name}")
    if found_restart_file and not (config.append):
        raise RuntimeError(
            f"Training instance exists at {config.root}/{config.run_name}; "
            "either set append to True or use a different root or runname"
        )

    config: Config  # Explicitly annotate the type of `config` to help the linter
    if found_restart_file:
        config, progress_config = check_for_config_updates(config)
        logging.info("--- Restart ---")
        func = partial(restart, progress_config=progress_config.as_dict())
    else:
        raise ValueError('This code is for restart only')

    _set_global_options(config)
    train_dataset, validation_dataset = None, None
    func(rank=0, world_size=1, config=config.as_dict(), train_dataset=train_dataset, validation_dataset=validation_dataset)



if __name__ == "__main__":
    main()
