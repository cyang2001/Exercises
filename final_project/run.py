"""Project entry point for the Kaggle spectrogram challenge."""

import importlib
import os
import subprocess
import sys

import dotenv
import hydra
import wandb
from omegaconf import DictConfig

from src.utils.logger import get_logger


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(os.path.join(DIR_PATH, ".env"), override=True)


@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run the configured project pipeline.

    Args:
        cfg: Hydra configuration for the current execution.
    """
    logger = get_logger(__name__)

    if cfg.mode.get("test_mode", False):
        logger.info("Test mode is enabled.")
        test_dir = cfg.mode.get("test_dir", "tests")
        test_file = cfg.mode.get("test_file", "")

        if test_file:
            test_path = os.path.join(test_dir, test_file)
            cmd = f"python -m unittest {test_path}"
            logger.info("Running tests for file: %s", test_path)
        else:
            cmd = f"python -m unittest discover {test_dir}"
            logger.info("Running all tests in directory: %s", test_dir)

        exit_code = subprocess.call(cmd, shell=True)
        logger.info("Test run finished with exit code %s", exit_code)
        return

    dispatch_dict = cfg.get("pipeline_dispatch", {})
    pipeline_name = cfg.get("mode", {}).get("pipeline_name")

    if not pipeline_name:
        logger.error("Please specify mode.pipeline_name in the configuration.")
        sys.exit(1)

    if pipeline_name not in dispatch_dict:
        logger.error(
            "No matching entry in pipeline_dispatch for pipeline_name=%s",
            pipeline_name,
        )
        sys.exit(1)

    module_path, func_name = dispatch_dict[pipeline_name].split(":")
    module = importlib.import_module(module_path)
    target_func = getattr(module, func_name)

    logger.info("Dispatching to %s", dispatch_dict[pipeline_name])
    target_func(cfg)


if __name__ == "__main__":
    main()
    wandb.finish()
