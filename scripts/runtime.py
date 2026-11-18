"""NumPy-only ACT action queue and RKNNLite adapter for board deployment."""

from collections import deque
from pathlib import Path

import numpy as np

INPUT_NAMES = ["observation.state", "observation.images.front", "observation.images.wrist"]


def prepare_observation(state, front_rgb, wrist_rgb):
    """Images must be RGB uint8 HWC; resize upstream to the trained 128x128."""
    state = np.asarray(state, dtype=np.float32).reshape(1, 18)
    values = [state]
    for image in [front_rgb, wrist_rgb]:
        image = np.asarray(image)
        if image.shape != (128, 128, 3) or image.dtype != np.uint8:
            raise ValueError(f"Expected RGB uint8 (128,128,3), got {image.shape} {image.dtype}")
        values.append(np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0)
    return values


class ActionChunkPolicy:
    """Execute the first ten actions of each fifty-action model prediction."""

    def __init__(self, infer, n_action_steps=10):
        if not 1 <= n_action_steps <= 50:
            raise ValueError("n_action_steps must be between 1 and 50")
        self.infer = infer
        self.n_action_steps = n_action_steps
        self.actions = deque()

    def reset(self):
        self.actions.clear()

    def select_action(self, inputs):
        if not self.actions:
            chunk = np.asarray(self.infer(inputs), dtype=np.float32).reshape(1, 50, 4)
            if not np.isfinite(chunk).all():
                raise RuntimeError("Model returned nonfinite actions")
            self.actions.extend(chunk[0, : self.n_action_steps].copy())
        return self.actions.popleft()


class RKNNPolicy(ActionChunkPolicy):
    """Load the compiled RKNN on a Rockchip board with RKNN-Toolkit-Lite2."""

    def __init__(self, model_path: str | Path, n_action_steps=10):
        from rknnlite.api import RKNNLite

        self.runtime = RKNNLite()
        ret = self.runtime.load_rknn(str(model_path))
        if ret != 0:
            self.runtime.release()
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = self.runtime.init_runtime()
        if ret != 0:
            self.runtime.release()
            raise RuntimeError(f"init_runtime failed: {ret}")
        super().__init__(self._infer, n_action_steps)

    def _infer(self, inputs):
        outputs = self.runtime.inference(inputs=inputs, data_format=["nchw"] * 3)
        if outputs is None or len(outputs) != 1:
            raise RuntimeError("RKNNLite inference failed")
        return outputs[0]

    def close(self):
        self.runtime.release()
