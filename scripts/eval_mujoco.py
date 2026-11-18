"""Closed-loop MuJoCo evaluation of exported ONNX or RKNNLite policy."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from runtime import INPUT_NAMES, ActionChunkPolicy, RKNNPolicy, prepare_observation

ROOT = Path(__file__).resolve().parents[1]


def evaluate(
    infer,
    episodes=3,
    video=None,
    video_width=1920,
    video_height=1080,
    camera_width=640,
    camera_height=480,
):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gym_hil  # noqa: F401
    import gymnasium as gym
    from gym_hil.mujoco_gym_env import GymRenderingSpec
    from PIL import Image

    env = gym.make(
        "gym_hil/PandaPickCube-v0",
        image_obs=True,
        render_mode="rgb_array",
        render_spec=GymRenderingSpec(width=camera_width, height=camera_height),
        ee_step_size={"x": 1.0, "y": 1.0, "z": 1.0},
    )
    policy = ActionChunkPolicy(infer, n_action_steps=10)
    results = []
    recorder = None
    try:
        if video is not None:
            import mujoco

            model = env.unwrapped._model
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, video_width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, video_height)
            recorder = mujoco.Renderer(model, height=video_height, width=video_width)
            Path(video).parent.mkdir(parents=True, exist_ok=True)
        for episode in range(episodes):
            obs, _ = env.reset(seed=episode)
            if video is not None and episode == 0:
                for name in ("front", "wrist"):
                    Image.fromarray(obs["pixels"][name]).save(
                        Path(video).with_name(f"{Path(video).stem}_{name}.png")
                    )
            policy.reset()
            frames = []
            success = False
            trace = []

            def record_frame(episode=episode, frames=frames):
                if recorder is not None and episode == 0:
                    recorder.update_scene(env.unwrapped._data, camera="front")
                    frames.append(recorder.render().copy())

            record_frame()
            for step in range(100):
                # The deployed checkpoint has a fixed 128x128 input contract.
                policy_images = [
                    np.asarray(
                        Image.fromarray(obs["pixels"][name]).resize((128, 128), Image.Resampling.BILINEAR)
                    )
                    for name in ("front", "wrist")
                ]
                inputs = prepare_observation(obs["agent_pos"], *policy_images)
                action = policy.select_action(inputs)
                obs, reward, terminated, truncated, info = env.step(action)
                record_frame()
                success |= bool(info.get("succeed", False))
                trace.append({"step": step, "action": action.tolist(), "reward": float(reward)})
                if terminated or truncated:
                    break
            result = {
                "episode": episode,
                "success": success,
                "steps": step + 1,
                "block_height_m": float(env.unwrapped._data.sensor("block_pos").data[2]),
                "trace": trace,
            }
            results.append(result)
            print({key: value for key, value in result.items() if key != "trace"}, flush=True)
            if frames:
                import imageio.v2 as imageio

                imageio.mimsave(
                    str(video),
                    frames,
                    fps=10,
                    codec="libx264",
                    macro_block_size=1,
                    ffmpeg_params=["-crf", "16", "-preset", "slow", "-movflags", "+faststart"],
                )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
    return {
        "episodes": results,
        "successes": sum(row["success"] for row in results),
        "total": len(results),
        "random_block_position": False,
        "camera_resolution": {"width": camera_width, "height": camera_height},
        "policy_image_shape": [1, 3, 128, 128],
        "camera_to_policy": "RGB bilinear resize to 128x128; no crop",
        "note": "Default fixed block position; not a generalization benchmark",
        "video": {
            "path": str(video),
            "width": video_width,
            "height": video_height,
            "fps": 10,
            "independent_of_policy_input": True,
        }
        if video is not None
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["onnx", "rknn-lite", "http"], default="http")
    parser.add_argument(
        "--url", default="http://127.0.0.1:18088", help="Board HTTP service through SSH tunnel"
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--video-width", type=int, default=1920)
    parser.add_argument("--video-height", type=int, default=1080)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    args = parser.parse_args()
    runtime = None
    if args.backend == "onnx":
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(args.model or ROOT / "onnx" / "act.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        def infer(inputs):
            return session.run(None, dict(zip(INPUT_NAMES, inputs, strict=True)))[0]
    elif args.backend == "http":
        from client import infer as http_infer

        def infer(inputs):
            payload = io.BytesIO()
            np.savez(payload, **dict(zip(INPUT_NAMES, inputs, strict=True)))
            return np.asarray(http_infer(args.url, payload.getvalue())["action"], dtype=np.float32)

    else:
        runtime = RKNNPolicy(args.model or ROOT / "models" / "act_franka_rk3588_fp16.rknn")
        infer = runtime._infer
    try:
        result = evaluate(
            infer,
            args.episodes,
            args.video or ROOT / "reports" / f"{args.backend}_rollout.mp4",
            args.video_width,
            args.video_height,
            args.camera_width,
            args.camera_height,
        )
        report = args.report or ROOT / "reports" / f"{args.backend}_mujoco.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        result["backend"] = args.backend
        result["inference_url"] = args.url if args.backend == "http" else None
        report.write_text(json.dumps(result, indent=2))
        print(f"Successes: {result['successes']}/{result['total']}", flush=True)
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
