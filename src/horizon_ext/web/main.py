"""CLI entrypoint for the local Horizon Brief web dashboard."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Horizon Brief web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--config", default="data/config.json")
    parser.add_argument("--data-dir", default="data")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = create_app(config_path=args.config, data_dir=args.data_dir)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
