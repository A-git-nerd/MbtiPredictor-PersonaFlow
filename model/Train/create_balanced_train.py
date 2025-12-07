import pandas as pd
import os

#for balance training set (classification)
# Configuration
INPUT_FILE = "sample.csv"
OUTPUT_FILE = "train.csv"
MAX_SAMPLES_PER_CLASS = 2000

def create_balanced_train(input_file, output_file, max_per_class):
    print("="*60)
    print("CREATING BALANCED TRAINING DATASET")
    print("="*60)
    
    if not os.path.exists(input_file):
        print(f"\n Error: {input_file} not found!")
        return
    
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} total samples")
    
    print("\n" + "="*60)
    print("ORIGINAL DISTRIBUTION")
    print("="*60)
    original_counts = df['mbti'].value_counts().sort_index()
    for mbti, count in original_counts.items():
        print(f"{mbti}: {count:5d} samples")
    
    print("\n" + "="*60)
    print(f"BALANCING (Max {max_per_class} samples per class)")
    print("="*60)
    
    balanced_dfs = []
    skipped_count = 0
    
    for mbti_type in sorted(df['mbti'].unique()):
        mbti_df = df[df['mbti'] == mbti_type]
        original_count = len(mbti_df)
        
        if original_count > max_per_class:
            # Sample randomly to reduce to max_per_class
            mbti_df_sampled = mbti_df.sample(n=max_per_class, random_state=42)
            skipped = original_count - max_per_class
            skipped_count += skipped
            print(f"{mbti_type}: {original_count:5d} → {max_per_class:5d} (skipped {skipped})")
            balanced_dfs.append(mbti_df_sampled)
        else:
            print(f"{mbti_type}: {original_count:5d} → {original_count:5d} (kept all)")
            balanced_dfs.append(mbti_df)
    
    # Combine all balanced data
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("FINAL DISTRIBUTION")
    print("="*60)
    final_counts = balanced_df['mbti'].value_counts().sort_index()
    for mbti, count in final_counts.items():
        print(f"{mbti}: {count:5d} samples")
    
    print(f"\nTotal samples in balanced dataset: {len(balanced_df)}")
    print(f"Total samples skipped: {skipped_count}")
    
    print(f"\nSaving balanced dataset to {output_file}...")
    balanced_df.to_csv(output_file, index=False)
    print(f"Saved successfully!")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input file:  {input_file} ({len(df)} samples)")
    print(f"Output file: {output_file} ({len(balanced_df)} samples)")
    print(f"Reduction:   {len(df) - len(balanced_df)} samples removed")
    print(f"Max per class: {max_per_class}")
    print("\nDone! You can now train with train.csv")


def main():
    print("\n" + "="*60)
    print("BALANCED TRAIN.CSV CREATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input:  {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Max samples per class: {MAX_SAMPLES_PER_CLASS}")
    
    if os.path.exists(OUTPUT_FILE):
        print(f"\nWarning: {OUTPUT_FILE} already exists and will be overwritten!")
        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            return
    
    create_balanced_train(INPUT_FILE, OUTPUT_FILE, MAX_SAMPLES_PER_CLASS)


if __name__ == "__main__":
    main()
