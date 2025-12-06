import csv
import random
import re

# reading from gemini generated file and cleaning it for train.csv 
# and then from train.csv creating valid.csv
INPUT_FILE = '../../data/l.csv'
CLEAN_FILE = 'train.csv'
VALID_FILE = 'valid.csv'
VALID_PER = 0.5

def clean_file():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(CLEAN_FILE, 'w', newline='', encoding='utf-8') as outfile:
        # Use more lenient CSV reading to handle malformed rows
        reader = csv.reader(infile, quoting=csv.QUOTE_ALL, skipinitialspace=True)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        # Write headers
        try:
            headers = next(reader)
            writer.writerow(headers)
        except StopIteration:
            print("Empty file")
            return

        # Write cleaned rows
        skipped = 0
        written = 0
        for i, row in enumerate(reader, start=2):
            try:
                # Skip rows that don't have exactly 3 columns
                if len(row) != 3:
                    skipped += 1
                    print(f"Skipping malformed row {i}: expected 3 fields, got {len(row)}")
                    continue
                
                # Clean each field: replace newlines and multiple spaces with single space
                cleaned_row = [re.sub(r'\s+', ' ', field.strip()) if field else field for field in row]
                writer.writerow(cleaned_row)
                written += 1
            except Exception as e:
                skipped += 1
                print(f"Error processing row {i}: {e}")
        
        print(f"Written {written} rows, skipped {skipped} malformed rows")


def create_valid_subset():
    with open(CLEAN_FILE, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    header = lines[0]
    rows = lines[1:]

    subset_size = max(1, int(len(rows) * VALID_PER))
    selected = random.sample(rows, subset_size)

    with open(VALID_FILE, 'w', encoding='utf-8') as outfile:
        outfile.write(header) 
        outfile.writelines(selected)


if __name__ == '__main__':
    clean_file()
    create_valid_subset()
    print("Clean and valid CSV files generated successfully.")