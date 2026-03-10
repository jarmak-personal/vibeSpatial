from __future__ import annotations

import json

from vibespatial.testing import verify_degeneracy_corpus


def main() -> int:
    print(json.dumps(verify_degeneracy_corpus(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
