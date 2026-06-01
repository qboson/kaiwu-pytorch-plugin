"""Compatibility entrypoint for the DPLM guided rerun workflow."""

try:
    from .workflows.rerun import *  # noqa: F401,F403
    from .workflows.rerun import main
except ImportError:  # pragma: no cover - direct script-path compatibility
    from workflows.rerun import *  # noqa: F401,F403
    from workflows.rerun import main


if __name__ == "__main__":
    main()
