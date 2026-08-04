"""Entry: python3 -m device_profile.phase2 [--diag]"""

from __future__ import annotations

import sys


def _entry(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("--diag", "diag", "2.2", "--phase22"):
        from .diag import main as diag_main

        return diag_main(args[1:])
    from .harness import main as harness_main

    return harness_main(args)


raise SystemExit(_entry())
