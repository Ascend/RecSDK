"""Export fixed-validation metrics and optional prediction tensors for RecSDK."""

import argparse
import json
from pathlib import Path


def append_result(path, text):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def export_validation(report, destination, name, cases_path):
    result = {"batch_size": report["validation"]["batch_size"],
              "samples": report["validation"]["validation_samples"],
              "mean_loss": report["validation"]["mean_loss"], **report["metrics"]}
    if cases_path is not None:
        import torch

        detail = json.loads(cases_path.read_text(encoding="utf-8"))
        if detail["fingerprints"] != report["fingerprints"] or detail["inputs"] != report["inputs"]:
            raise ValueError("prediction cases and validation summary come from different runs")
        cases = detail["cases"]
        # SID codes are small integers. Float tensors also support the framework's error statistics.
        predictions = {key: torch.tensor([case[key] for case in cases], dtype=torch.float32)
                       for key in ("target", "greedy", "beam")}
        model_dir = destination / name
        model_dir.mkdir(exist_ok=True)
        torch.save(predictions, model_dir / f"predictions_{name}_EAGER_0.pt")
    append_result(destination / "precision_result.txt", f"{report['platform'].upper()}_{name}_EAGER_{result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("cannot export an unsuccessful run")
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"OxygenREC-{report['precision']}"
    if report["protocol"] != "oxygenrec_v2_full_fixed_validation_v1":
        raise ValueError(f"unsupported OxygenREC protocol {report['protocol']}")
    export_validation(report, args.output, name, args.cases)
    print(f"Benchmark results: {args.output}")


if __name__ == "__main__":
    main()
