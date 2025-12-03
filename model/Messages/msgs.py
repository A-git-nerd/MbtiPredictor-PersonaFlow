import re
import csv
import time
from google import genai
from google.genai.errors import APIError
import os
from dotenv import load_dotenv

load_dotenv()

# roman_urdu -> english by gemini
API_KEY = os.getenv("GEMINI_API_KEY_1")
# API_KEY = os.getenv("GEMINI_API_KEY_2")

INPUT_FILE = "../../data/chat.txt"
OUTPUT_FILE = "../../data/translated_chat.csv"
BATCH_SIZE = 50  # Processing 50 messages per API call 
DELAY_SECONDS = 5  # Delay between API calls to prevent rate limiting

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    print("Please ensure you have the 'google-genai' library installed and a valid API key.")
    exit()

def is_unwanted_line(text):
    """
    Checks if a message should be skipped.
    """
    # Regex to check for common non-message content (omitted media, status messages)
    if re.search(r"<(Media omitted|audio omitted|video omitted|Image omitted)>", text, re.IGNORECASE):
        return True
    
    # Check for simple punctuation-only messages
    if re.fullmatch(r"[\s._-]*", text):
        return True
    
    # Check for common WhatsApp metadata/system messages
    if re.search(r"(Messages and calls are end-to-end encrypted|created group)", text, re.IGNORECASE):
        return True
    
    return False

def extract_messages(file_path):
    whatsapp_line_regex = re.compile(
        r"^(?:\d{1,2}/\d{1,2}/\d{2,4}, \s*\d{1,2}:\d{2}(?:\s*[ap]m)?\s*-\s*(?:.*?:\s*))?(.+)$", 
        re.MULTILINE
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    entries = re.split(r"^\d{1,2}/\d{1,2}/\d{2,4}, \s*\d{1,2}:\d{2}(?:\s*[ap]m)?\s*-\s*", content, flags=re.MULTILINE)

    for entry in entries:
        if not entry.strip():
            continue

        # Spliting the entry to get the last part like 
        # 'Ibrahim SE: Da fuc @\u2068Abdul Rafay SE\u2069'
        parts = entry.split(':', 1)
        
        #  actual message is the last part
        if len(parts) > 1:
            message = parts[-1].strip()
        else:
            # Handle group-change messages or lines 
            # (like "Abdullah Latif SE added you")
            message = entry.strip()

        # Clean up msgs like remove phone numbers and @mentions
        message = re.sub(r'^\+\d{1,3} \d{10,12}: ', '', message).strip() 
        message = re.sub(r'@[^\s]+\s*', '', message).strip() 
        
        if message and not is_unwanted_line(message):
            yield message

def get_translation_prompt(messages):
    messages_str = "\n".join(f"[{i+1}]: {msg}" for i, msg in enumerate(messages))
    prompt = f"""
    You are an expert translator. Your task is to translate the following messages from Roman Urdu or any other language to English.
    
    **CRITICAL RULES:**
    1.  **Output Format:** You MUST return the result as a single, valid CSV string with two columns: **"roman_urdu", "english"**.
    2.  **Quotation Marks:** The values must be enclosed in double quotes (e.g., `"Original Message"`, `"English Translation"`).
    3.  **Input/Output Mapping:** The translated output MUST contain one line for every numbered input message.
    4.  **Special Filtering:** If a message contains code (like Python, C++, Java, etc.), or is pure numerical data, DO NOT translate it. Instead, for both the "roman_urdu" and "english" columns, use the word **"Study"** (e.g., `"Study"`, `"Study"`).
    5.  **Language Check:** If the original message is already in proper English, just keep the "roman_urdu" and "english" columns identical.
    
    **Input Messages:**
    {messages_str}
    
    **Example Output Format:**
    "Bro kidr ho?","Where are you bro?"
    "print('Hello')","Study"
    "My name is John.","My name is John."
    """
    return prompt

def process_and_translate_messages(input_file, output_file, batch_size):
    print(f"Starting message extraction and translation...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(["roman_urdu", "english"])

    all_messages = list(extract_messages(input_file))
    total_messages = len(all_messages)
    print(f"Successfully extracted {total_messages} clean messages.")

    for i in range(0, total_messages, batch_size):
        batch = all_messages[i:i + batch_size]
        print(f"\n--- Processing batch {i//batch_size + 1}/{total_messages//batch_size + 1} ({len(batch)} messages) ---")

        if not batch:
            continue

        prompt = get_translation_prompt(batch)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                csv_output = response.text.strip()

                # Appending the translated data to the output CSV file
                with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
                    outfile.write(csv_output + "\n")
                
                print(f"Batch {i//batch_size + 1} processed and saved successfully.")
                break

            except APIError as e:
                print(f"API Error on batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {DELAY_SECONDS * (attempt + 1)} seconds...")
                    time.sleep(DELAY_SECONDS * (attempt + 1)) 
                else:
                    print(f"Failed to process batch {i//batch_size + 1} after {max_retries} attempts. Skipping.")
            
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Skipping batch {i//batch_size + 1}.")
                break 

        if (i + batch_size) < total_messages:
            print(f"Waiting for {DELAY_SECONDS} seconds before next batch...")
            time.sleep(DELAY_SECONDS)

    print("\nAll batches processed. Translation complete.")
    print(f"Output saved to {output_file}.")


if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            pass 
    except FileNotFoundError:
        print(f"'{INPUT_FILE}' not found. Creating a dummy file for demonstration.")
        os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)

    process_and_translate_messages(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE)