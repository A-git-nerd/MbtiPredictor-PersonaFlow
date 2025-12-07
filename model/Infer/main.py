import os
import re
import csv
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_DIR = "../Train/mbti_model"
CHAT_FILE = "../../data/chatAbdur.txt"
OUTPUT_DIR = "../Train/output"

# --- Neutral Words Filter (Same as training) ---
NEUTRAL_WORDS = {
    "ok", "okay", "k", "kk", "g", "han", "haan", "hmm", "hmmm", "acha", "achaa",
    "thanks", "thx", "thank you", "shukriya", "sahi", "theek", "theek hai",
    "yes", "no", "yup", "nope", "na", "nhi", "nai", "nahi",
    "lol", "lmao", "haha", "hahaha", "xd",
    "??", "?", "...", "..",
    "deleted message", "you deleted this message", "this message was deleted",
    "missed voice call", "missed video call",
    "sticker", "image", "video", "gif", "audio",
    "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿",
    "done", "waiting", "coming", "omw"
}

def is_neutral(text):
    if not isinstance(text, str):
        return True
    text = text.strip().lower()
    text_clean = re.sub(r'[^\w\s]', '', text)
    if not text_clean: return True
    if text in NEUTRAL_WORDS: return True
    if text_clean in NEUTRAL_WORDS: return True
    if len(text_clean) < 2: return True
    return False

# --- Chat Parsing ---
def parse_chat(file_path):
    print(f"Parsing {file_path}...")
    line_regex = re.compile(r"(\d{2}/\d{2}/\d{4}),\s*(\d{1,2}:\d{2}\s*[apAP][mM])\s*-\s*(.*?):\s*(.*)")
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = line_regex.match(line)
            if match:
                date_str, time_str, sender, msg = match.groups()
                try:
                    dt = datetime.strptime(date_str, "%d/%m/%Y")
                    messages.append({"datetime": dt, "sender": sender.strip(), "message": msg.strip()})
                except ValueError:
                    continue
    return messages

def get_unique_senders(messages):
    senders = set()
    for m in messages:
        sender = m["sender"]
        if "removed" not in sender.lower() and "left" not in sender.lower() and "added" not in sender.lower():
            senders.add(sender)
    return sorted(list(senders))

# --- Grouping Logic ---
MONTHS_PER_GROUP = 2

def group_messages(messages, sender_name):
    sender_msgs = [m for m in messages if m["sender"] == sender_name]
    if not sender_msgs:
        return []

    start_date = min(m["datetime"] for m in sender_msgs)
    
    def get_group_id(msg_date):
        delta_months = (msg_date.year - start_date.year) * 12 + (msg_date.month - start_date.month)
        return delta_months // MONTHS_PER_GROUP

    grouped_data = defaultdict(list)
    
    for msg in sender_msgs:
        group_id = get_group_id(msg["datetime"])
        
        # Calculate group start/end
        group_start_month_idx = group_id * MONTHS_PER_GROUP
        
        # Start Date
        start_delta_years = (start_date.month + group_start_month_idx - 1) // 12
        start_month = ((start_date.month + group_start_month_idx - 1) % 12) + 1
        group_start_date = datetime(start_date.year + start_delta_years, start_month, 1)
        
        # End Date
        end_delta_months = MONTHS_PER_GROUP
        end_total_months = start_month + end_delta_months - 1
        end_delta_years = end_total_months // 12
        end_month = (end_total_months % 12) + 1
        
        # Get last day of end month
        if end_month == 12:
            next_month = datetime(group_start_date.year + end_delta_years + 1, 1, 1)
        else:
            next_month = datetime(group_start_date.year + end_delta_years, end_month + 1, 1)
            
        group_end_date = next_month - timedelta(days=1)
        
        # Cap end date to current date if it exceeds it
        if group_end_date > datetime.now():
            group_end_date = datetime.now()
        
        grouped_data[group_id].append({
            "group_id": group_id,
            "start_date": group_start_date.strftime("%d/%m/%Y"),
            "end_date": group_end_date.strftime("%d/%m/%Y"),
            "message": msg["message"]
        })
        
    return grouped_data

# --- Prediction ---
def predict_group_mbti(model, tokenizer, classes, messages, device):
    # Filter neutral messages
    valid_msgs = [m for m in messages if not is_neutral(m)]
    
    if not valid_msgs:
        return "Unknown"
        
    # Tokenize (process in batches to avoid OOM if many messages)
    batch_size = 16
    all_probs = []
    
    for i in range(0, len(valid_msgs), batch_size):
        batch_msgs = valid_msgs[i:i+batch_size]
        inputs = tokenizer(batch_msgs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probs)
    
    if not all_probs:
        return "Unknown"
        
    # Concatenate all probabilities: [total_msgs, num_classes]
    all_probs = torch.cat(all_probs, dim=0)
    
    # Soft Voting: Average probabilities across all messages
    avg_probs = torch.mean(all_probs, dim=0).cpu().numpy()
    
    # Get top 3 predictions
    top3_indices = avg_probs.argsort()[-3:][::-1]
    top3_labels = [(classes[i], avg_probs[i]) for i in top3_indices]
    
    print(f"    Top 3: {', '.join([f'{l}: {p:.2f}' for l, p in top3_labels])}")
    
    dominant_mbti = classes[top3_indices[0]]
    return dominant_mbti

def main():
    # 1. Load Model
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        classes = np.load(os.path.join(MODEL_DIR, "classes.npy"), allow_pickle=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run trainMBTI.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # 2. Parse Chat
    messages = parse_chat(CHAT_FILE)
    if not messages:
        print("No messages found.")
        return
        
    # 3. Select User
    senders = get_unique_senders(messages)
    print("\nAvailable Users:")
    for i, s in enumerate(senders):
        print(f"{i}: {s}")
        
    try:
        choice = int(input("\nSelect user (enter number): "))
        selected_user = senders[choice]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
        
    print(f"\nProcessing for user: {selected_user}")
    
    # 4. Group Messages
    grouped_data = group_messages(messages, selected_user)
    
    # 5. Predict & Generate Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # groups.csv
    groups_csv_path = os.path.join(OUTPUT_DIR, "groups.csv")
    timeline_data = []
    
    print("Generating groups.csv and predicting MBTI...")
    with open(groups_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "start_date", "end_date", "message"])
        writer.writeheader()
        
        for group_id in sorted(grouped_data.keys()):
            msgs = grouped_data[group_id]
            # Write to groups.csv
            for m in msgs:
                writer.writerow(m)
            
            # Predict
            msg_texts = [m["message"] for m in msgs]
            dominant_mbti = predict_group_mbti(model, tokenizer, classes, msg_texts, device)
            
            start_date = msgs[0]["start_date"]
            end_date = msgs[0]["end_date"]
            
            timeline_data.append({
                "group_id": group_id,
                "start_date": start_date,
                "end_date": end_date,
                "dominant_mbti": dominant_mbti
            })
            print(f"Group {group_id}: {dominant_mbti}")
            
    # timeline.csv
    timeline_csv_path = os.path.join(OUTPUT_DIR, "timeline.csv")
    with open(timeline_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_id", "start_date", "end_date", "dominant_mbti"])
        writer.writeheader()
        writer.writerows(timeline_data)
        
    print(f"\nDone! Files saved to {OUTPUT_DIR}/")
    print(f"- {groups_csv_path}")
    print(f"- {timeline_csv_path}")

if __name__ == "__main__":
    main()
