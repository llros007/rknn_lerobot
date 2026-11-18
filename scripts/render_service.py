"""Print a systemd unit for this checkout; does not install or start it."""

import argparse
import getpass
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=getpass.getuser())
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if "\n" in str(root) or "\n" in args.user or " " in str(root):
        parser.error("Use a repository path without spaces or newlines")
    print(f"""[Unit]
Description=RK3588S NPU ACT inference service
After=network.target
RequiresMountsFor={root}

[Service]
Type=simple
User={args.user}
SupplementaryGroups=render video
WorkingDirectory={root}
Environment=LD_LIBRARY_PATH={root}/lib
Environment=PYTHONUNBUFFERED=1
Environment=OMP_NUM_THREADS=4
ExecStart={root}/.venv-board/bin/python {root}/scripts/serve.py --host 127.0.0.1 --port 8088
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
""")


if __name__ == "__main__":
    main()
