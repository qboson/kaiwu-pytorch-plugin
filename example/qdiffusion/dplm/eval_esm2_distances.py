"""Compatibility entrypoint for the dplm ESM2 evaluation workflow."""

try:
    from .workflows.esm2_eval import *  # noqa: F401,F403
    from .workflows.esm2_eval import main
except ImportError:  # pragma: no cover - direct script-path compatibility
    from workflows.esm2_eval import *  # noqa: F401,F403
    from workflows.esm2_eval import main


if __name__ == "__main__":
    main()
