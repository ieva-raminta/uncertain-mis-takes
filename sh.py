import os
import sys
def main():
    if len(sys.argv) != 3:
        print("Usage: python read_sweep.py <sweep_file> <output_file>")
        sys.exit(1)

    sweep_file = sys.argv[1]

    if not os.path.isfile(sweep_file):
        print(f"Error: {sweep_file} does not exist.")
        sys.exit(1)

    with open(sweep_file, 'r') as file:
        for line in file:
            if "solution" in line:
                if "finetuned_explanations" in line and ("cross_entropy" in line or "cross_label" in line) and "combined" in line and "--dropout_value 0.5" not in line: 
                    with open(sys.argv[2], 'a') as output_file:
                        output_file.write(line)

if __name__ == "__main__":
    main()
