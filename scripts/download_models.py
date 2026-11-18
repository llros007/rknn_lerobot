"""Download the released PyTorch/RKNN files from Hugging Face and verify hashes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    info = json.loads((ROOT / "model_info.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["rknn", "pytorch", "all"], default="rknn")
    parser.add_argument("--repo-id", default=info["model_repo"])
    parser.add_argument("--revision", default=info["model_revision"])
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    sums = Path(hf_hub_download(args.repo_id, "SHA256SUMS", revision=args.revision))
    selected = {"rknn", "pytorch"} if args.format == "all" else {args.format}
    for line in sums.read_text().splitlines():
        expected, remote = line.split(maxsplit=1)
        path = Path(remote)
        if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
            raise ValueError(f"Invalid model path: {remote}")
        if path.parts[0] not in selected:
            continue
        cached = Path(hf_hub_download(args.repo_id, remote, revision=args.revision))
        actual = hashlib.sha256(cached.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: {remote}")
        destination = (
            args.output_dir
            / ("checkpoint" if path.parts[0] == "pytorch" else "models")
            / Path(*path.parts[1:])
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, destination)
        print(f"Verified: {destination}")


if __name__ == "__main__":
    main()
