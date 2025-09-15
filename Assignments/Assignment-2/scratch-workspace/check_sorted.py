#!/usr/bin/env python3
import csv
import sys

def is_sorted(arr):
    """Check if arr is sorted in non-decreasing order."""
    return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

def check_csv_sorted(csv_file):
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row:
                continue

            try:
                length = int(row[0])
                arr = list(map(int, row[1:]))
            except ValueError:
                print(f"Line {line_num}: Invalid integers, skipping")
                continue

            if len(arr) != length:
                print(f"Line {line_num}: Length mismatch "
                      f"(expected {length}, got {len(arr)})")
                continue

            if is_sorted(arr):
                print(f"Line {line_num}: Sorted ✓")
            else:
                print(f"Line {line_num}: Not sorted ✗")
                break

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <csv_file>")
        sys.exit(1)

    check_csv_sorted(sys.argv[1])
