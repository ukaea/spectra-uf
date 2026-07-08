#!/usr/bin/env python3
"""
Wrapper script to run pylint consistently in CI and locally.
"""

import sys
from pylint.lint import Run

def main():
    # Default to linting the "spectrauf" package if no args given
    targets = sys.argv[1:] or ["spectrauf"]

    pylint_opts = [
        "--output-format=colorized",   # nice for local runs
        "--reports=n",                 # disable full report
        "--score=y",                   # show overall score
    ] + targets

    # Run pylint (use exit=False for modern versions)
    results = Run(pylint_opts, exit=False)

    # Exit with pylint's status code (non‑zero if issues found)
    sys.exit(results.linter.msg_status)

if __name__ == "__main__":
    main()
