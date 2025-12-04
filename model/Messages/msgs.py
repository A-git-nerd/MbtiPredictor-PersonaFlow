import re
import csv
import time
from google import genai
from google.genai.errors import APIError
import os
from dotenv import load_dotenv
from cleanMsgs import clean_chat

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY_1")

INPUT_FILE = "../../data/chatBoys.txt"
OUTPUT_FILE = "../../data/translated_chat.csv"
BATCH_SIZE = 50
DELAY_SECONDS = 5

try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit()


def is_unwanted_line(text):
    if re.search(r"<(Media omitted|audio omitted|video omitted|Image omitted)>", text, re.IGNORECASE):
        return True
    if re.fullmatch(r"[\s._-]*", text):
        return True
    if re.search(r"(Messages and calls are end-to-end encrypted|created group)", text, re.IGNORECASE):
        return True
    return False


def extract_messages(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    entries = re.split(
        r"^\d{1,2}/\d{1,2}/\d{2,4}, \s*\d{1,2}:\d{2}(?:\s*[ap]m)?\s*-\s*",
        content,
        flags=re.MULTILINE
    )

    for entry in entries:
        if not entry.strip():
            continue

        parts = entry.split(':', 1)
        if len(parts) > 1:
            message = parts[-1].strip()
        else:
            message = entry.strip()

        message = re.sub(r'^\+\d{1,3} \d{10,12}: ', '', message).strip()
        message = re.sub(r'@[^\s]+\s*', '', message).strip()

        if message and not is_unwanted_line(message):
            yield message


def get_translation_prompt(messages):
    messages_str = "\n".join(f"[{i+1}]: {msg}" for i, msg in enumerate(messages))

    prompt = f"""
You are an expert translator, emoji interpreter, and MBTI classifier.

Your task:
Translate Roman Urdu (or any language) to English AND assign an MBTI personality type.

IMPORTANT RULES:
1. Output MUST be a valid CSV string with THREE columns:
   "roman_urdu","english","mbti"

2. All values must be in double quotes.

3. There must be EXACTLY one CSV row per input message.

4. MBTI must be one of:
   ISTJ, ISFJ, INFJ, INTJ,
   ISTP, ISFP, INFP, INTP,
   ESTP, ESFP, ENFP, ENTP,
   ESTJ, ESFJ, ENFJ, ENTJ

5. If a message contains code or numeric-only content → output:
   "Study","Study","ISTJ"

6. If a message is already in English, keep English same.

7. EMOJI RULES:
   - If the message is ONLY emojis, translate emoji to a simple English description.
   - Assign MBTI based on the emoji emotion:
       ❤️😩😭🥺 = INFP or ISFP  
       😂🤣😆 = ESFP or ENFP  
       😡🤬 = ESTJ or ENTJ  
       🙂😊 = ISFJ or ESFJ  
       👍👌🙏 = ISTJ or ESTJ  
       😎🔥💪 = ESTP  
       😐😑 = ISTJ  
   - If emoji appears INSIDE text, translate normally and choose MBTI based on emotional tone.

8. MBTI general rule:
   - Logical, short, factual → ISTJ
   - Emotional, expressive → ESFP / ISFP / INFP
   - Argumentative / assertive → ENTJ
   - Problem-solving / technical → INTP

Input Messages:
{messages_str}

Example output:
"Bro kidr ho?","Bro, where are you?","ESTP"
"😂😂","Laughing","ESFP"
"print('hi')","Study","ISTJ"
"I am fine","I am fine","ISTJ"
"""
    return prompt


def process_and_translate_messages(input_file, output_file, batch_size):
    print("Starting message extraction and translation...")

    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(["roman_urdu", "english", "mbti"])

    all_messages = list(extract_messages(input_file))
    total_messages = len(all_messages)
    print(f"Successfully extracted {total_messages} clean messages.")

    for i in range(0, total_messages, batch_size):
        batch = all_messages[i:i + batch_size]
        print(f"\n--- Processing batch {i//batch_size + 1}/{(total_messages//batch_size) + 1} ({len(batch)} messages) ---")

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

                with open(output_file, 'a', newline='', encoding='utf-8') as outfile:
                    outfile.write(csv_output + "\n")

                print(f"Batch {i//batch_size + 1} processed and saved successfully.")
                break

            except APIError as e:
                print(f"API Error in batch {i//batch_size + 1}, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(DELAY_SECONDS * (attempt + 1))
                else:
                    print("Skipping batch due to repeated failure.")

            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        if (i + batch_size) < total_messages:
            time.sleep(DELAY_SECONDS)

    print("All batches processed. Translation complete.")
    print(f"Saved to {output_file}.")


if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8'):
            pass
    except FileNotFoundError:
        os.makedirs(os.path.dirname(INPUT_FILE), exist_ok=True)

    clean_chat(INPUT_FILE)
    process_and_translate_messages(INPUT_FILE, OUTPUT_FILE, BATCH_SIZE)
