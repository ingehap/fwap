"""Allow ``python -m fwap`` to invoke the CLI entry point.

See :func:`fwap.cli.main` for the actual argument parser and demo
dispatch. Running ``python -m fwap`` is equivalent to running the
installed ``fwap`` console script.
"""

from fwap.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
