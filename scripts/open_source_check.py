#!/usr/bin/env python3
"""Check the selected release source tree without deleting local files."""
from release_inventory import check_inventory, release_files


def main():
    issues = check_inventory()
    if issues:
        for issue in issues:
            print(issue)
        return 1
    print(f"open_source_check: {len(release_files())} selected source files passed")
    print("Excluded private/output directories are not scanned as release content.")
    print("Review staged Git files and reachable Git history separately before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
