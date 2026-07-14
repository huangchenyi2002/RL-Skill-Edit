from __future__ import annotations

import json

from .cli import parse_args, run


def main() -> int:
    args = parse_args()
    result = run(args.config, seed=args.seed, test_only=args.test_only)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
