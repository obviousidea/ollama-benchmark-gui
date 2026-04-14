#!/usr/bin/env python3
"""
Display current version.

Usage:
    python bump.py    # shows current version label
"""

import sys
from versioning import get_full_label

if __name__ == "__main__":
    print(get_full_label())
