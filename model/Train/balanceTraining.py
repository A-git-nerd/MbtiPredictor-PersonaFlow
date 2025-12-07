import pandas as pd
import numpy as np
import os
import time
from collections import Counter
from google import genai
from dotenv import load_dotenv

load_dotenv()

#generating corpous to make train.csv balanced using gemini
# Configuration
API_KEY = os.getenv("GEMINI_API_KEY_1")
INPUT_FILE = "train.csv"
TARGET_SAMPLES_PER_CLASS = 2100  # Target samples per MBTI type
BATCH_SIZE = 50  
DELAY_SECONDS = 2 

# MBTI personality descriptions for better generation
MBTI_DESCRIPTIONS = {
    "ISTJ": "Practical, fact-minded, responsible, organized, logical. Values tradition and reliability. Messages are direct, informative, and focused on facts.",
    "ISFJ": "Warm, caring, detail-oriented, loyal, supportive. Focuses on helping others. Messages show concern for others and attention to details.",
    "INFJ": "Insightful, idealistic, deep, empathetic, visionary. Seeks meaning and connection. Messages are thoughtful, philosophical, and compassionate.",
    "INTJ": "Strategic, analytical, independent, innovative, perfectionist. Focuses on systems and ideas. Messages are logical, strategic, and sometimes critical.",
    "ISTP": "Practical, hands-on, logical, adaptable, observant. Enjoys problem-solving. Messages are brief, practical, and action-oriented.",
    "ISFP": "Artistic, sensitive, gentle, flexible, appreciative. Values personal expression. Messages are gentle, supportive, and emotionally expressive.",
    "INFP": "Idealistic, empathetic, creative, introspective, emotional. Seeks authenticity. Messages are emotional, idealistic, and deeply personal.",
    "INTP": "Analytical, curious, logical, theoretical, detached. Loves exploring ideas. Messages are analytical, questioning, and intellectually curious.",
    "ESTP": "Energetic, bold, direct, action-oriented, spontaneous. Lives in the moment. Messages are bold, casual, and action-focused.",
    "ESFP": "Outgoing, fun-loving, spontaneous, enthusiastic, friendly. Enjoys social interactions. Messages are upbeat, friendly, and often use emojis.",
    "ENFP": "Enthusiastic, creative, optimistic, warm, imaginative. Loves new possibilities. Messages are excited, imaginative, and emotionally expressive.",
    "ENTP": "Innovative, clever, argumentative, quick-witted, challenging. Debates ideas. Messages are witty, argumentative, and intellectually challenging.",
    "ESTJ": "Organized, practical, decisive, direct, responsible. Values structure and order. Messages are direct, commanding, and focused on efficiency.",
    "ESFJ": "Caring, social, cooperative, organized, helpful. Focuses on harmony. Messages are warm, inclusive, and socially oriented.",
    "ENFJ": "Charismatic, empathetic, inspirational, organized, persuasive. Natural leaders. Messages are inspiring, encouraging, and people-focused.",
    "ENTJ": "Commanding, strategic, assertive, decisive, efficient. Natural commanders. Messages are authoritative, strategic, and goal-oriented."
}

def initialize_gemini():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY or GEMINI_API_KEY_2 not found in environment variables!")
    
    try:
        client = genai.Client(api_key=API_KEY)
        print("Gemini API initialized successfully")
        return client
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        exit(1)


def analyze_class_distribution(csv_file):
    print(f"\n{'='*60}")
    print("ANALYZING CURRENT DATASET")
    print('='*60)
    
    df = pd.read_csv(csv_file)
    mbti_counts = df['mbti'].value_counts().sort_index()
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nCurrent distribution:")
    for mbti, count in mbti_counts.items():
        print(f"  {mbti}: {count:5d} samples")
    
    return df, mbti_counts


def calculate_needed_samples(mbti_counts, target):
    needed = {}
    
    print(f"\n{'='*60}")
    print(f"TARGET: {target} SAMPLES PER CLASS")
    print('='*60)
    
    for mbti in MBTI_DESCRIPTIONS.keys():
        current = mbti_counts.get(mbti, 0)
        if current < target:
            needed[mbti] = target - current
        else:
            needed[mbti] = 0
    
    total_needed = sum(needed.values())
    print(f"\nSamples needed:")
    for mbti, count in sorted(needed.items()):
        if count > 0:
            print(f"  {mbti}: {count:5d} samples needed")
    
    print(f"\n  Total new samples to generate: {total_needed}")
    
    return needed


def get_sample_messages(df, mbti_type, num_samples=5):
    mbti_df = df[df['mbti'] == mbti_type]
    if len(mbti_df) > 0:
        samples = mbti_df.sample(n=min(num_samples, len(mbti_df)))['roman_urdu'].tolist()
        return samples
    return []


def generate_messages_batch(client, mbti_type, num_samples, sample_messages):
    description = MBTI_DESCRIPTIONS[mbti_type]
    
    # Create examples context
    examples_context = ""
    if sample_messages:
        examples_context = "\n\nExample messages from existing data:\n"
        for i, msg in enumerate(sample_messages[:5], 1):
            examples_context += f"{i}. {msg}\n"
    
    prompt = f"""Generate {num_samples} realistic WhatsApp chat messages that reflect {mbti_type} personality traits.

MBTI Type: {mbti_type}
Personality: {description}
{examples_context}

REQUIREMENTS:
1. Generate messages in ROMAN URDU (Urdu written in English alphabet) - this is CRITICAL
2. Mix casual chat, questions, statements, reactions, and opinions
3. Reflect {mbti_type} personality traits authentically
4. Include variety: short messages (2-5 words) and longer ones (10-20 words)
5. Some with emojis 😊👍, some without
6. Keep it natural - like real WhatsApp conversations
7. Use common Urdu/English code-mixing (Hinglish style)

Examples of Roman Urdu style:
- "Yar kya scene hai?"
- "Bhai assignment complete ho gaya"
- "Mujhe lagta hai ye sahi nahi hai"
- "Chalo phir milte hain"

FORMAT YOUR RESPONSE AS:
1. [roman_urdu message]|||[English translation]
2. [roman_urdu message]|||[English translation]
...

Generate exactly {num_samples} messages in this format. Each line must have BOTH Roman Urdu and English separated by |||"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'temperature': 0.9,
                'top_p': 0.95,
                'max_output_tokens': 2048,
            }
        )
        
        return response.text
    except Exception as e:
        print(f"    API Error: {e}")
        return None


def parse_generated_messages(response_text, mbti_type):
    messages = []
    lines = response_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('**'):
            continue
        
        # Remove numbering (1. 2. etc.)
        line = line.lstrip('0123456789. ')
        
        # Split by |||
        if '|||' in line:
            parts = line.split('|||')
            if len(parts) >= 2:
                roman_urdu = parts[0].strip()
                english = parts[1].strip()
                
                # Clean up any markdown or quotes
                roman_urdu = roman_urdu.strip('"\'*`')
                english = english.strip('"\'*`')
                
                if roman_urdu and english and len(roman_urdu) > 2:
                    messages.append({
                        'roman_urdu': roman_urdu,
                        'english': english,
                        'mbti': mbti_type
                    })
    
    return messages


def append_to_csv(csv_file, new_messages):
    if not new_messages:
        return 0
    
    df_new = pd.DataFrame(new_messages)
    df_new.to_csv(csv_file, mode='a', header=False, index=False)
    return len(new_messages)


def balance_dataset(client, csv_file, target_per_class, batch_size):
    df, mbti_counts = analyze_class_distribution(csv_file)
    needed = calculate_needed_samples(mbti_counts, target_per_class)
    
    total_needed = sum(needed.values())
    if total_needed == 0:
        print("\nDataset is already balanced!")
        return
    
    print(f"\n{'='*60}")
    print("STARTING GENERATION PROCESS")
    print('='*60)
    
    total_generated = 0
    
    # Process each MBTI type that needs more samples
    for mbti_type, samples_needed in sorted(needed.items()):
        if samples_needed == 0:
            continue
        
        print(f"\n--- Generating for {mbti_type} ---")
        print(f"Need: {samples_needed} samples")
        
        # Get sample messages for context
        sample_messages = get_sample_messages(df, mbti_type, num_samples=5)
        
        # Generate in batches
        generated_for_type = 0
        batches = (samples_needed + batch_size - 1) // batch_size
        
        for batch_num in range(batches):
            remaining = samples_needed - generated_for_type
            current_batch_size = min(batch_size, remaining)
            
            if current_batch_size <= 0:
                break
            
            print(f"  Batch {batch_num + 1}/{batches}: Generating {current_batch_size} messages... ", end='', flush=True)
            
            # Generate messages
            response = generate_messages_batch(client, mbti_type, current_batch_size, sample_messages)
            
            if response:
                # Parse response
                new_messages = parse_generated_messages(response, mbti_type)
                
                # Append to CSV
                appended = append_to_csv(csv_file, new_messages)
                generated_for_type += appended
                total_generated += appended
                
                print(f"Added {appended} messages")
            else:
                print("Failed")
            
            if batch_num < batches - 1:
                time.sleep(DELAY_SECONDS)
        
        print(f"  Total for {mbti_type}: {generated_for_type} messages generated")
    
    print(f"\n{'='*60}")
    print("BALANCING COMPLETE")
    print('='*60)
    print(f"\nTotal new messages generated: {total_generated}")
    
    df_final, final_counts = analyze_class_distribution(csv_file)
    
    print(f"\n{'='*60}")
    print("FINAL DISTRIBUTION")
    print('='*60)
    for mbti in sorted(MBTI_DESCRIPTIONS.keys()):
        count = final_counts.get(mbti, 0)
        status = "Done " if count >= target_per_class * 0.95 else "Not Done "
        print(f"{status} {mbti}: {count:5d} / {target_per_class} ({count/target_per_class*100:.1f}%)")


def main():
    print("\n" + "="*60)
    print("MBTI DATASET BALANCER WITH GEMINI")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Target samples per class: {TARGET_SAMPLES_PER_CLASS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Delay between batches: {DELAY_SECONDS}s")
    
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: {INPUT_FILE} not found!")
        return
    
    client = initialize_gemini()
    
    print(f"\nThis will append new messages to {INPUT_FILE}")
    confirm = input("Continue? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("Cancelled.")
        return
    
    balance_dataset(client, INPUT_FILE, TARGET_SAMPLES_PER_CLASS, BATCH_SIZE)
    
    print(f"\nDone! Train your model with the balanced dataset.")


if __name__ == "__main__":
    main()
