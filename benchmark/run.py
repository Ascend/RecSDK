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
MODELS_DIR = Path("models")
MODELS_CONFIGS_DIR= MODELS_DIR / "configs"

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
    parser.add_argument(
        "--cpu",
        action="store_true",
        default=None,
        help="Force use CPU only (overrides config file setting)"
    )
    args = parser.parse_args()
    return args


def install_depend(config: dict, target_dir: Path) -> bool:
    try:        
        if config.get("pre_cmd"):
            cmds = config.get("pre_cmd")
            for cmd in cmds:
                logger.info(f"Executing pre command: {cmd}")
                subprocess.run(cmd.split(" "), cwd=str(target_dir), check=True)
        if config.get("pip_install_requirements"):
            logger.info("Installing requirements.txt dependencies...")
            subprocess.run(
                ["pip", "install", "-r", "requirements.txt"],
                cwd=str(target_dir),
                check=True,
            )
        if config.get("pip_install_self"):
            logger.info("Installing current directory dependencies...")
            subprocess.run(
                ["pip", "install", "-e", "."], cwd=str(target_dir), check=True
            )
        if config.get("extra_cmd"):
            if "TorchEasyRec" in str(target_dir):
                import glob
                proto_files = glob.glob("models/TorchEasyRec/tzrec/protos/*.proto")
                cmd = ["protoc", "--proto_path=models/TorchEasyRec/", "--python_out=models/TorchEasyRec/"] + proto_files
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
                proto_files = glob.glob("models/TorchEasyRec/tzrec/protos/models/*.proto")
                cmd = ["protoc", "--proto_path=models/TorchEasyRec/", "--python_out=models/TorchEasyRec/"] + proto_files
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
            else:
                cmds = config.get("extra_cmd")
                for cmd in cmds:
                    if cmd == "":
                        continue
                    logger.info(f"Executing extra command: {cmd}")
                    subprocess.run(cmd.split(" "), cwd=str(target_dir), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"pip failed, error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True


def set_env(config: dict, cpu_only: bool):
    os.environ["MODEL_MODE"] = config.get("mode")
    os.environ["MODEL_EPOCH"] = str(config.get("epoch"))
    os.environ["MODEL_PROFILING_FLAG"] = str(config.get("profiling_flag"))
    os.environ["MODEL_COMPILE_FLAG"] = str(config.get("compile_flag"))
    os.environ["MODEL_ACLGRAPH_FLAG"] = str(config.get("aclgraph_flag"))
    os.environ["MODEL_DATA_TYPE"] = config.get("data_type")
    os.environ["MODEL_NAME"] = config.get("name")
    os.environ["MODEL_E2E_FLAG"] = str(config.get("e2e_flag"))
    os.environ["MODEL_CPU_ONLY"] = str(cpu_only)
    lib_fbgemm_npu_api_so_path = config.get("lib_fbgemm_npu_api_so_path")
    if lib_fbgemm_npu_api_so_path:
        logger.info(f"LIB_FBGEMM_NPU_API_SO_PATH: {lib_fbgemm_npu_api_so_path}")
        os.environ["LIB_FBGEMM_NPU_API_SO_PATH"] = config.get("lib_fbgemm_npu_api_so_path")

    os.environ["COMPARE_ACCURACY_FLAG"] = str(config.get("compare_accuracy_flag"))
    os.environ["SAVE_TENSOR_FLAG"] = str(config.get("save_tensor_flag"))

def run_model(config: dict, target_dir: Path, cpu_only: bool) -> bool:
    set_env(config, cpu_only)
    logger.info(f"Running model command: {config.get('run_cmd')}")
    try:
        subprocess.run(config.get("run_cmd"), cwd=str(target_dir), check=True)
        logger.info("Model run successful!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Model run failed, error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True

def download_file(url: str, destination: str, target_dir: Path) -> bool:
    try:
        cmd = [
            'wget',
            '--no-check-certificate',
            '-O',
            destination,
            url
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=str(target_dir), check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"downlaod data, error message:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True

def extract_tar(tar_file: Path, extract_dir: Path) -> bool: 
    import tarfile
    """解压tar.gz文件"""
    logger.info(f"解压文件: {tar_file} -> {extract_dir}")
    try:
        with tarfile.open(tar_file, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
        logger.info(f"解压完成: {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"解压失败: {e}")
        return False

def download_data_file(target_dir: Path) -> bool:
    data_config = read_config(Path.absolute(Path("data") / "TorchEasyRecData.json"))
    if not data_config:
        logger.error("数据配置文件不存在")
        return False
    data_path = Path(target_dir) / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    for key, url in data_config['data_files'].items():
        tar_file = Path(target_dir) / f"{key}.tar.gz"
        if not tar_file.exists():
            logger.info(f"开始下载文件: {url} -> {tar_file}")
            if not download_file(url, f"{key}.tar.gz", target_dir):
                logger.error(f"下载文件失败: {url}")
                return False
            if not extract_tar(tar_file, data_path):
                logger.error(f"解压文件失败: {tar_file}")
                return False
    return True


def build_initial_tree(target_dir: Path) -> bool:
    tree_dir = target_dir / "data/init_tree"
    if tree_dir.exists():
        logger.info(f"初始树已存在: {tree_dir}")
        return True

    logger.info("开始创建初始树")
    cmd = (
        "python -m tzrec.tools.tdm.init_tree "
        "--item_input_path data/taobao_ad_feature_transformed_fill/*.parquet "
        "--item_id_field adgroup_id "
        "--cate_id_field cate_id "
        "--attr_fields cate_id,campaign_id,customer,brand,price "
        "--node_edge_output_file data/init_tree "
        "--tree_output_dir data/init_tree"
    )

    try:
        subprocess.run(cmd, cwd=str(target_dir), shell=True, check=True)
        logger.info("创建初始树成功")
    except subprocess.CalledProcessError as e:
        logger.error(f"创建初始树失败, error message:\n{e.stderr}")

        return False
    except Exception as e:
        logger.error(f"Unknown error: {e}")
        return False
    return True


def load_config(config: dict, target_dir: Path) -> bool:
    config_path = target_dir / "model_configs"
    os.makedirs(str(config_path),  exist_ok=True)
    config_file_name = config.get("name").lower() + "_taobao.config"
    config_path = config_path / config_file_name
    if config_path.exists():
        return True
    try:
        if config.get("model_config_set"):
            cmds = config.get("model_config_set")
            for cmd in cmds:
                logger.info(f"Executing extra command: {cmd.split(' ')}")
                subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"modify config file failed, error message:\n{e.stderr}")
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
    if config.get("A5_flag"):
        commit_id = config.get("commit_id_for_A5")
        patch_path = config.get("patch_path_for_A5")
    else:
        commit_id = config.get("commit_id")
        patch_path = config.get("patch_path")
    if not repo_url:
        logger.error("Configuration missing 'url' field")
        raise ValueError("Configuration missing 'url' field")
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    target_dir = MODELS_DIR / repo_name

    # 2. Execute git clone and apply patch
    if not download_and_install(config, repo_url, target_dir, commit_id, patch_path):
        logger.error("Clone failed, process terminated.")
        return False
    if "TorchEasyRec" in repo_url:
        if not download_data_file(target_dir):
            logger.error("Download data file failed, process terminated.")
            return
        if not build_initial_tree(target_dir):
            logger.error("Build initial tree failed, process terminated.")
            return
        if not load_config(config, target_dir):
            logger.error("Load config failed, process terminated.")
            return

    # 3. Run model
    if not run_model(config, target_dir, args.cpu):
        logger.error("Model run failed, process terminated.")
        return

    logger.info("Benchmark process completed successfully!")


if __name__ == "__main__":
    main()
