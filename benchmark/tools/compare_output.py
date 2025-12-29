import torch
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime


def setup_logging(log_dir: str = None):
    """
    配置日志系统，同时输出到控制台和文件

    Args:
        log_dir: 日志文件保存目录,如果为None则不保存到文件
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"compare_output_{timestamp}.log"

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 控制台输出
            logging.StreamHandler(sys.stdout),
            # 文件输出
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    logging.info(f"[INFO] 日志文件已保存到: {log_file}")


def do_compare(actual_tensor, expected_tensor):
    """
    计算各种误差指标
    """
    results = {}

    # 平均绝对误差
    abs_error = torch.abs(actual_tensor - expected_tensor)
    results["mean_abs_error"] = abs_error.mean().item()

    # 平均相对误差
    eps = 1e-18
    rel_error = abs_error / (torch.abs(actual_tensor) + eps)
    results["mean_rel_error"] = rel_error.mean().item()

    # 均方误差
    mse = torch.mean((actual_tensor - expected_tensor) ** 2)
    rmse = torch.sqrt(mse)
    results["rmse"] = rmse.item()

    logging.info(f"平均绝对误差: {results['mean_abs_error']:.6e}")
    logging.info(f"平均相对误差: {results['mean_rel_error']:.6e}")
    logging.info(f"均方根误差(RMSE): {results['rmse']:.6e}")


def compare_detail(actual_tensor, expected_tensor):
    if isinstance(actual_tensor, torch.Tensor) and isinstance(expected_tensor, torch.Tensor):
        do_compare(actual_tensor, expected_tensor)
    elif isinstance(actual_tensor, dict) and isinstance(expected_tensor, dict):
        for key in actual_tensor.keys():
            if key in expected_tensor:
                if isinstance(actual_tensor[key], torch.Tensor) and isinstance(expected_tensor[key], torch.Tensor):
                    do_compare(actual_tensor[key], expected_tensor[key])
    else:
        logging.error(f"[ERROR] 输入的类型不正确: {type(actual_tensor)}")


def find_matching_file(
    expected_dir: Path, model_name: str, mode: str, step: int
) -> Optional[Path]:
    """
    在expected目录中查找对应的文件

    Args:
        expected_dir: expected目录路径
        model_name: 模型名称
        mode: 模式（eager或inductor）
        step: 步数

    Returns:
        找到的文件路径，如果不存在则返回None
    """
    filename = f"predictions_{model_name}_{mode}_{step}.pt"
    file_path = expected_dir / model_name / filename

    if file_path.exists():
        return file_path
    return None


def compare_tensors(
    actual_tensor, expected_tensor, rtol=1e-4, atol=1e-4
) -> Tuple[bool, str]:
    """
    比较两个tensor的精度

    Args:
        actual_tensor: 实际输出
        expected_tensor: 期望输出
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度

    Returns:
        (是否匹配, 错误信息)
    """
    try:
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=rtol, atol=atol, equal_nan=True
        )
        return True, ""
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"比较过程中出错: {str(e)}"


def parse_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """
    解析文件名，提取模型名、模式和步数

    文件名格式：predictions_{model_name}_{mode}_{step}.pt

    Args:
        filename: 文件名

    Returns:
        (model_name, mode, step) 或 None（如果格式不匹配）
    """
    if not filename.startswith("predictions_") or not filename.endswith(".pt"):
        return None

    parts = filename.split("_")
    if len(parts) != 4:
        return None

    model_name = parts[1]
    mode = parts[2]
    step = int(parts[-1].split(".")[0])

    return (model_name, mode, step)


def compare_outputs(actual_dir: str, expected_dir: str, rtol=1e-4, atol=1e-4):
    """
    比较actual和expected目录下的所有输出文件

    Args:
        actual_dir: actual目录路径
        expected_dir: expected目录路径
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
    """
    actual_path = Path(actual_dir)
    expected_path = Path(expected_dir)

    if not actual_path.exists():
        logging.error(f"[ERROR] Actual目录不存在: {actual_dir}")
        return

    if not expected_path.exists():
        logging.error(f"[ERROR] Expected目录不存在: {expected_dir}")
        return

    # 遍历actual目录下的所有模型文件夹
    for model_dir in actual_path.iterdir():
        total_files = 0
        matched_files = 0
        failed_files = 0
        missing_files = 0
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        logging.info(f"[INFO] 处理模型: {model_name}")

        max_files = 1
        # 遍历该模型目录下的所有文件
        for i, file_path in enumerate(model_dir.iterdir()):
            if not file_path.is_file() or not file_path.suffix == ".pt":
                continue

            if i >= max_files:
                break

            filename = file_path.name
            parsed = parse_filename(filename)

            if parsed is None:
                logging.warning(f"[WARNING] 无法解析文件名格式: {filename}")
                continue

            file_model_name, mode, step = parsed

            # 验证模型名是否匹配
            if file_model_name != model_name:
                logging.warning(
                    f"[WARNING] 文件名中的模型名({file_model_name})与目录名({model_name})不匹配: {filename}"
                )
                continue

            total_files += 1

            # 在expected目录中查找对应文件
            expected_file = find_matching_file(expected_path, model_name, mode, step)

            if expected_file is None:
                logging.error(
                    f"[ERROR] 在expected目录中未找到对应文件: "
                    f"模型={model_name}, 模式={mode}, 步数={step}"
                )
                missing_files += 1
                continue

            # 加载并比较文件
            try:
                actual_output = torch.load(
                    file_path, map_location="cpu", weights_only=False
                )
                expected_output = torch.load(
                    expected_file, map_location="cpu", weights_only=False
                )

                if model_name == "ple":
                    match_flag = True
                    for name in ["logits_ctr", "probs_ctr", "logits_cvr", "probs_cvr"]:
                        is_match, error_msg = compare_tensors(actual_output[name], expected_output[name], rtol, atol)
                        match_flag = match_flag and is_match
                        if is_match:
                            compare_detail(actual_output, expected_output)
                            logging.debug(
                                f"[DEBUG] 精度对齐: {model_name}/{filename}"
                                f"模式={mode}, 步数={step}"
                            )
                        else:
                            compare_detail(actual_output, expected_output)
                            logging.error(
                                f"[ERROR] 精度非对齐: {model_name}/{filename} 模式={mode}, 步数={step}\n错误信息: {error_msg}")
                    if match_flag:
                        matched_files += 1
                    else:
                        failed_files += 1
                else:
                    is_match, error_msg = compare_tensors(
                        actual_output, expected_output, rtol, atol
                    )

                    if is_match:
                        compare_detail(actual_output, expected_output)
                        logging.debug(
                            f"[SUCCESS] 精度对齐: {model_name}/{filename}"
                            f"模式={mode}, 步数={step}"
                        )
                        matched_files += 1
                    else:
                        compare_detail(actual_output, expected_output)
                        logging.error(
                            f"[ERROR] 精度非对齐: {model_name}/{filename}"
                            f"模式={mode}, 步数={step}\n错误信息: {error_msg}"
                        )
                        failed_files += 1

            except Exception as e:
                logging.error(
                    f"[ERROR] 处理文件时出错: {model_name}/{filename}\n"
                    f"错误信息: {str(e)}"
                )
                failed_files += 1

        # 打印统计信息
        logging.info(
            f"比较结果：{model_name} 总文件数: {total_files} "
            f"精度对齐文件数: {matched_files} "
            f"精度不对齐文件数: {failed_files} "
            f"缺失文件数: {missing_files}"
        )

        if failed_files > 0 or missing_files > 0:
            logging.error(f"[ERROR] {model_name} 存在不匹配或缺失的文件，请检查！")


def main():
    parser = argparse.ArgumentParser(
        description="比较actual和expected目录下的模型输出文件"
    )
    parser.add_argument(
        "--actual_output",
        type=str,
        default="../models/save_results",
        help="actual输出目录路径",
    )
    parser.add_argument(
        "--expected_output",
        type=str,
        default="../models/save_results_gpu",
        help="expected输出目录路径",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-4, help="相对误差容忍度 (默认: 1e-4)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4, help="绝对误差容忍度 (默认: 1e-4)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="日志文件保存目录 (默认: ./logs)"
    )

    args = parser.parse_args()

    # 配置日志系统
    setup_logging(args.log_dir)

    logging.info(f"[INFO] Actual输出目录: {args.actual_output}")
    logging.info(f"[INFO] Expected输出目录: {args.expected_output}")
    logging.info(f"[INFO] 相对误差容忍度: {args.rtol}")
    logging.info(f"[INFO] 绝对误差容忍度: {args.atol}")

    compare_outputs(
        args.actual_output, args.expected_output, rtol=args.rtol, atol=args.atol
    )


if __name__ == "__main__":
    main()
