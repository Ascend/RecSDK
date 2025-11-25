#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import logging
from pathlib import Path

# ================== Log Configuration ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ================== Path Configuration ==================
CONFIG_FILE = "config.json"
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(exist_ok=True)


def read_config(config_path: Path) -> dict:
    """Read config.json and return configuration dictionary"""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def download_and_install(
    config: dict, repo_url: str, target_dir: Path, commit_id: str, patch_path: str
) -> bool:
    if target_dir.exists():
        logger.info(f"Target directory already exists: {target_dir}")
        return True

    logger.info(f"Cloning repository: {repo_url}")
    try:
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", commit_id],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Clone successful! Saved to: {target_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed! Error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False

    # Check and apply patch
    if patch_path:
        patch_file = Path.absolute(Path(patch_path))
        if not apply_patch(patch_file, target_dir):
            return False
    else:
        logger.info("No patch path specified, skipping patch step.")

    if not install_depend(config, target_dir):
        logger.error("Dependencies installed failed!")
        return False
    return True


def apply_patch(patch_path: Path, target_dir: Path) -> bool:
    """Apply patch using git apply"""
    if not patch_path.exists():
        logger.error(f"Patch file does not exist: {patch_path}")
        return False

    logger.info(f"Applying patch: {patch_path}")
    try:
        subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=str(target_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"Patch applied successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Patch application failed! Error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False


def get_args():
    parser = argparse.ArgumentParser(description="Script for cloning repositories and applying patches")
    parser.add_argument("config", type=str, help="Configuration file")
    args = parser.parse_args()
    return args


def install_depend(config: dict, taget_dir: Path) -> bool:
    try:
        if config.get("pip_install_requirements"):
            logger.info("Installing requirements.txt dependencies...")
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=str(taget_dir),
                check=True,
            )
        if config.get("pip_install_self"):
            logger.info("Installing current directory dependencies...")
            subprocess.run(
                ["pip", "install", "-e", "."], cwd=str(taget_dir), check=True
            )
        if config.get("extra_cmd"):
            cmds = config.get("extra_cmd")
            for cmd in cmds:
                logger.info(f"Executing extra command: {cmd}")
                subprocess.run(cmd.split(" "), cwd=str(taget_dir), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"pip failed, error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True


def set_env(config: dict):
    os.environ["MODEL_MODE"] = config.get("mode")
    os.environ["MODEL_EPOCH"] = str(config.get("epoch"))
    os.environ["MODEL_PROFILING_FLAG"] = str(config.get("profiling_flag"))
    os.environ["MODEL_COMPILE_FLAG"] = str(config.get("compile_flag"))
    os.environ["MODEL_ACLGRAPH_FLAG"] = str(config.get("aclgraph_flag"))
    os.environ["MODEL_DATA_TYPE"] = config.get("data_type")
    os.environ["MODEL_NAME"] = config.get("name")
    os.environ["MODEL_E2E_FLAG"] = str(config.get("e2e_flag"))


def run_model(config: dict, taget_dir: Path) -> bool:
    set_env(config)
    logger.info(f"Running model command: {config.get('run_cmd')}")
    try:
        subprocess.run(config.get("run_cmd"), cwd=str(taget_dir), check=True)
        logger.info("Model run successful!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Model run failed, error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True


def main():
    args = get_args()
    logger.info("Starting benchmark process...")

    # 1. Read configuration file
    config = read_config(Path.absolute(Path("configs") / args.config))
    logger.info(f"Successfully read configuration file: {args.config}")

    repo_url = config.get("url")
    patch_path = config.get("patch_path")
    commit_id = config.get("commit_id", "")
    if not repo_url:
        logger.error("Configuration missing 'url' field")
        raise ValueError("Configuration missing 'url' field")
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    target_dir = MODELS_DIR / repo_name

    # 2. Execute git clone and apply patch
    if not download_and_install(config, repo_url, target_dir, commit_id, patch_path):
        logger.error("Clone failed, process terminated.")
        return

    # 3. Run model
    if not run_model(config, target_dir):
        logger.error("Model run failed, process terminated.")
        return

    logger.info("Benchmark process completed successfully!")


if __name__ == "__main__":
    main()
