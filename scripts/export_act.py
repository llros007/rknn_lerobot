"""Export the local ACT checkpoint with its saved normalization, then verify ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify
from safetensors.torch import load_file
from torch import nn

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ["observation.state", "observation.images.front", "observation.images.wrist"]


class ACTExport(nn.Module):
    """Raw physical state and RGB [0,1] in; physical-unit action chunk out.

    The trained weights are untouched. The training-only VAE is omitted by eval
    tracing; ACT's deterministic inference latent is zero.
    """

    def __init__(self, policy: ACTPolicy, checkpoint: Path):
        super().__init__()
        self.model = policy.model
        stats = load_file(str(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
        post = load_file(str(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        for i, name in enumerate(INPUT_NAMES):
            self.register_buffer(f"mean_{i}", stats[f"{name}.mean"])
            self.register_buffer(f"std_{i}", stats[f"{name}.std"])
        self.register_buffer("action_mean", post["action.mean"])
        self.register_buffer("action_std", post["action.std"])

    def forward(self, state, front, wrist):
        values = [state, front, wrist]
        normalized = [
            (value - getattr(self, f"mean_{i}")) / (getattr(self, f"std_{i}") + 1e-8)
            for i, value in enumerate(values)
        ]
        batch = dict(zip(INPUT_NAMES, normalized, strict=True))
        batch["observation.images"] = normalized[1:]
        actions = self.model(batch)[0]
        return actions * self.action_std + self.action_mean


def error_metrics(reference, actual):
    difference = np.abs(np.asarray(reference) - np.asarray(actual))
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "per_action_dim_max_abs": difference.reshape(-1, difference.shape[-1]).max(0).tolist(),
    }


def load_policy(checkpoint: Path):
    config = ACTConfig.from_pretrained(checkpoint)
    config.device = "cpu"
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=config).cpu().eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoint")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local LeRobot dataset; otherwise use its Hugging Face repository",
    )
    parser.add_argument("--dataset-repo-id", default="aractingi/franka_sim_pick_lift_6")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.mha.set_fastpath_enabled(False)
    policy = load_policy(args.checkpoint)
    wrapper = ACTExport(policy, args.checkpoint).eval()

    # Use actual decoded demonstration images; closed-loop simulation is separate.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, video_backend="torchcodec")
    samples = []
    for index in [0, 3, 10, 20, 37, 38, 48, 60, 100, 200, 400, 700]:
        item = dataset[index]
        values = [item[key].unsqueeze(0).cpu().float() for key in INPUT_NAMES]
        samples.append((f"dataset_{index:04d}", values))

    for directory in ["onnx", "samples", "reports"]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    for name, values in samples:
        with torch.no_grad():
            expected = wrapper(*values).numpy()
        np.savez(
            ROOT / "samples" / f"{name}.npz",
            **{key: value.numpy() for key, value in zip(INPUT_NAMES, values, strict=True)},
            action=expected,
        )

    raw_path = ROOT / "onnx" / "act_raw.onnx"
    final_path = ROOT / "onnx" / "act.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(samples[0][1]),
            str(raw_path),
            opset_version=13,
            input_names=INPUT_NAMES,
            output_names=["action"],
            do_constant_folding=True,
            dynamo=False,
        )
    model = onnx.load(raw_path)
    onnx.checker.check_model(model)
    model, valid = simplify(model)
    if not valid:
        raise RuntimeError("ONNX simplification failed validation")
    onnx.checker.check_model(model)
    onnx.save(model, final_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(final_path), sess_options=options, providers=["CPUExecutionProvider"])
    report = {"samples": {}, "opset": 13, "inputs": {}, "output": [1, 50, 4]}
    for name, values in samples:
        feed = {key: value.numpy() for key, value in zip(INPUT_NAMES, values, strict=True)}
        expected = np.load(ROOT / "samples" / f"{name}.npz")["action"]
        actual = session.run(None, feed)[0]
        metrics = error_metrics(expected, actual)
        report["samples"][name] = metrics
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    for key, value in zip(INPUT_NAMES, samples[0][1], strict=True):
        report["inputs"][key] = list(value.shape)
    report["max_abs"] = max(m["max_abs"] for m in report["samples"].values())
    (ROOT / "reports" / "onnx_validation.json").write_text(json.dumps(report, indent=2))
    with (args.checkpoint / "model.safetensors").open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    manifest = {
        "policy": "act",
        "source_checkpoint_step": 50000,
        "checkpoint_sha256": checkpoint_sha256,
        "input_order": INPUT_NAMES,
        "input_shapes": report["inputs"],
        "input_dtype": "float32",
        "image_input": "RGB NCHW float32 in [0,1]; normalization is embedded; do not normalize twice",
        "state_input": "physical state: 7 qpos + 7 qvel + gripper command + 3 TCP coordinates",
        "output": {"name": "action", "shape": [1, 50, 4], "units": "XYZ delta meters; gripper in [0,2]"},
        "normalization_embedded": True,
        "latent": "zero (deterministic inference)",
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "temporal_ensemble_coeff": policy.config.temporal_ensemble_coeff,
        "simulation": {"env": "gym_hil/PandaPickCube-v0", "ee_step_size": {"x": 1.0, "y": 1.0, "z": 1.0}},
        "onnx": "onnx/act.onnx",
        "reference": {
            "url": "https://gitcode.com/openeuler/IB_Robot",
            "commit": "fb21cf1ff784a2592a6cbd7f521289e34db2093c",
            "file": "src/model_utils/model_utils/export_onnx_rknn.py",
        },
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
