from __future__ import annotations

from pathlib import Path


def main() -> int:
    removed = 0
    for directory, patterns in {
        Path("results"): ("*.csv", "*.md"),
        Path("figures"): ("*.png",),
    }.items():
        directory.mkdir(exist_ok=True)
        for pattern in patterns:
            for path in directory.glob(pattern):
                path.unlink()
                removed += 1

    print(f"Removed {removed} generated result file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
