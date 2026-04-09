#!/usr/bin/env python3
"""
Manual semantic version bump.

Usage:
    python bump.py patch     # 1.0.0 → 1.0.1
    python bump.py minor     # 1.0.1 → 1.1.0
    python bump.py major     # 1.1.0 → 2.0.0
    python bump.py           # shows current version
"""

import sys
from versioning import bump, get_full_label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(get_full_label())
        sys.exit(0)

    part = sys.argv[1].lower()
    try:
        new_version = bump(part)
        print(f"Version bumped → {new_version}")
        print(get_full_label())
    except ValueError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
