#!/usr/bin/env python3
import csv
import random
import sys

def generate_sequences(L, N, output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        for _ in range(N):
            length = random.randint(1, L)  # actual length ≤ L
            # For testing, restricting values to range 0–255
            sequence = [random.randint(0, 255) for _ in range(length)]
            writer.writerow([length] + sequence)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python GenSequence.py <L> <N> <file_name.csv>")
        sys.exit(1)

    L = int(sys.argv[1])
    N = int(sys.argv[2])
    output_csv = sys.argv[3]

    generate_sequences(L, N, output_csv)
    print(f"Generated {N} sequences (max length {L}, values 0–255) in {output_csv}")

