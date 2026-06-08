"""Compatibility entrypoint for the full dplm training workflow."""

try:
    from .workflows.train import *  # noqa: F401,F403
    from .workflows.train import main
except ImportError:  # pragma: no cover - direct script-path compatibility
    from workflows.train import *  # noqa: F401,F403
    from workflows.train import main


if __name__ == "__main__":
    main()
