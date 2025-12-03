import csv
import random

# reading from gemini generated file and cleaning it for train.csv 
# and then from train.csv creating valid.csv
INPUT_FILE = 'translated_chat_with_mbti.csv'
CLEAN_FILE = 'train.csv'
VALID_FILE = 'valid.csv'
VALID_PER = 0.5

def clean_file():
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(CLEAN_FILE, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

        # Write headers without quotes
        headers = next(reader)
        outfile.write(','.join(headers) + '\n')

        # Write cleaned rows with quoted values
        for row in reader:
            writer.writerow(row)


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