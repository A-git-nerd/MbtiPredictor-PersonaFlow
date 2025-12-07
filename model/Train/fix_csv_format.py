import csv
import os

# Configuration
INPUT_FILE = "train.csv"
OUTPUT_FILE = "train_fixed.csv"
BACKUP_FILE = "train_backup.csv"

def fix_csv_format(input_file, output_file):
    print("="*60)
    print("CSV FORMAT FIXER")
    print("="*60)
    
    if not os.path.exists(input_file):
        print(f"\nError: {input_file} not found!")
        return
    
    # backup
    print(f"\nCreating backup: {BACKUP_FILE}")
    import shutil
    shutil.copy2(input_file, BACKUP_FILE)
    print("Backup created")
    
    print(f"\nReading {input_file}...")
    fixed_rows = []
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            
            if not headers or len(headers) < 3:
                print(f"Error: Invalid CSV headers: {headers}")
                return
            
            print(f"Headers: {headers}")
            
            for i, row in enumerate(reader, start=2):  # Starting from 2 (line 1 is header)
                try:
                    # Extract fields
                    roman_urdu = row.get('roman_urdu', '').strip()
                    english = row.get('english', '').strip()
                    mbti = row.get('mbti', '').strip()
                    
                    # Replace any newlines/multilines with space
                    roman_urdu = ' '.join(roman_urdu.split())
                    english = ' '.join(english.split())
                    mbti = ' '.join(mbti.split())
                    
                    # Skip empty rows
                    if not roman_urdu or not english or not mbti:
                        error_count += 1
                        continue
                    
                    fixed_rows.append({
                        'roman_urdu': roman_urdu,
                        'english': english,
                        'mbti': mbti
                    })
                    
                except Exception as e:
                    print(f"  Error on line {i}: {e}")
                    error_count += 1
                    continue
        
        print(f" Read {len(fixed_rows)} valid rows")
        if error_count > 0:
            print(f" Skipped {error_count} invalid/empty rows")
        
    except Exception as e:
        print(f" Error reading file: {e}")
        return
    
    print(f"\nWriting fixed CSV to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['roman_urdu', 'english', 'mbti'], 
                                   quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(fixed_rows)
        
        print(f" Saved {len(fixed_rows)} rows to {output_file}")
        
    except Exception as e:
        print(f" Error writing file: {e}")
        return
    
    # Show sample
    print("\n" + "="*60)
    print("SAMPLE OUTPUT (first 5 rows)")
    print("="*60)
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 6:  # Header + 5 rows
                break
            print(line.rstrip())
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Backup file: {BACKUP_FILE}")
    print(f"Rows processed: {len(fixed_rows)}")
    print(f"Errors/skipped: {error_count}")
    print("\n Done!")


def replace_original():
    if os.path.exists(OUTPUT_FILE):
        print("\n" + "="*60)
        print("REPLACE ORIGINAL FILE")
        print("="*60)
        confirm = input(f"\nReplace {INPUT_FILE} with {OUTPUT_FILE}? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            import shutil
            shutil.move(OUTPUT_FILE, INPUT_FILE)
            print(f" Replaced {INPUT_FILE}")
            print(f"  Original backed up to {BACKUP_FILE}")
        else:
            print(f"Kept both files. Use {OUTPUT_FILE} for training.")


def main():
    print("\n" + "="*60)
    
    fix_csv_format(INPUT_FILE, OUTPUT_FILE)
    
    replace_original()


if __name__ == "__main__":
    main()
