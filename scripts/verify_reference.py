"""Check ONNX against LeRobot's original saved preprocessor/model/postprocessor."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from export_act import INPUT_NAMES, error_metrics, load_policy

from lerobot.policies.factory import make_pre_post_processors

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    policy = load_policy(ROOT / "checkpoint")
    pre, post = make_pre_post_processors(
        policy.config,
        pretrained_path=str(ROOT / "checkpoint"),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(ROOT / "onnx" / "act.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    report = {
        "reference": "original LeRobot saved processor pipeline + ACTPolicy.predict_action_chunk",
        "samples": {},
    }
    with torch.inference_mode():
        for path in sorted((ROOT / "samples").glob("*.npz")):
            sample = np.load(path)
            batch = {name: torch.from_numpy(sample[name]) for name in INPUT_NAMES}
            reference = post(policy.predict_action_chunk(pre(batch))).numpy()
            actual = session.run(None, {name: sample[name] for name in INPUT_NAMES})[0]
            report["samples"][path.stem] = error_metrics(reference, actual)
            np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
    report["max_abs"] = max(row["max_abs"] for row in report["samples"].values())
    (ROOT / "reports" / "reference_pipeline_validation.json").write_text(json.dumps(report, indent=2))
    print("Original LeRobot pipeline vs ONNX max abs error:", report["max_abs"])


if __name__ == "__main__":
    main()
