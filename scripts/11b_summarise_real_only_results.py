"""Write real-node-only directed-dependence summaries."""

from __future__ import annotations

import runpy


def main() -> None:
    # Keep one implementation source of truth while exposing the requested script
    # name for the project-control workflow.
    runpy.run_path("scripts/11_make_preliminary_summary.py", run_name="__main__")


if __name__ == "__main__":
    main()
