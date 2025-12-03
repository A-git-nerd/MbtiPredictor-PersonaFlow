import os
import re
import csv
import numpy as np
import torch
from datetime import datetime, timedelta
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../mbti_model" 
TIMELINE = "data/timeline.csv"
GROUPS = "data/groups.csv"
ALL_CHATS = "data/all_chats.csv"

# --- Neutral Words Filter ---
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


tokenizer = None
model = None
classes = None
device = None

def load_model():
    global tokenizer, model, classes, device
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        classes = np.load(os.path.join(MODEL_DIR, "classes.npy"), allow_pickle=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# --- Chat Parsing ---
def parse_chat_users(file_path):
    print(f"Parsing {file_path} for users...")
    # Updated regex to handle single digit days/months
    line_regex = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}\s*[apAP][mM])\s*-\s*(.*?):\s*(.*)")
    senders = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            match = line_regex.match(line)
            if match:
                _, _, sender, _ = match.groups()
                sender = sender.strip()
                if "removed" not in sender.lower() and "left" not in sender.lower() and "added" not in sender.lower():
                    senders.add(sender)
    
    sorted_senders = sorted(list(senders))
    print(f"Found users: {sorted_senders}")
    return sorted_senders

def parse_chat_messages(file_path):
    print(f"Parsing {file_path} for messages...")
    # Updated regex to handle single digit days/months
    line_regex = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}),\s*(\d{1,2}:\d{2}\s*[apAP][mM])\s*-\s*(.*?):\s*(.*)")
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
        
        # Calculating group start/end
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
       
        last_msg_date = max(m["datetime"] for m in grouped_data[group_id]) if group_id in grouped_data and grouped_data[group_id] else group_start_date
        grouped_data[group_id].append({
            "group_id": group_id,
            "datetime": msg["datetime"], 
            "message": msg["message"]
        })

    final_grouped_data = {}
    for gid, msgs in grouped_data.items():
        if not msgs: continue
        msgs.sort(key=lambda x: x["datetime"])
        start_date = msgs[0]["datetime"]
        end_date = msgs[-1]["datetime"]

        final_grouped_data[gid] = []
        for m in msgs:
            final_grouped_data[gid].append({
                "group_id": gid,
                "start_date": start_date.strftime("%d/%m/%Y"),
                "end_date": end_date.strftime("%d/%m/%Y"),
                "message": m["message"],
                "datetime": m["datetime"]
            })
            
    return final_grouped_data

# --- Prediction ---
def predict_batch_mbti(msg_texts):
    global model, tokenizer, classes, device
    
    results = []
    valid_indices = []
    valid_texts = []
    
    for i, text in enumerate(msg_texts):
        if is_neutral(text):
            results.append("Neutral")
        else:
            results.append(None)
            valid_indices.append(i)
            valid_texts.append(text)
            
    if not valid_texts:
        return results
        
    batch_size = 16
    
    for i in range(0, len(valid_texts), batch_size):
        batch_texts = valid_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_indices = probs.argmax(dim=-1).cpu().numpy()
            
            for j, idx in enumerate(top_indices):
                original_index = valid_indices[i + j]
                results[original_index] = classes[idx]
                
    return results

def predict_group_mbti(msg_texts):
    predictions = predict_batch_mbti(msg_texts)
    valid_preds = [p for p in predictions if p != "Neutral"]
    
    if not valid_preds:
        return "Unknown"
    from collections import Counter
    counts = Counter(valid_preds)
    return counts.most_common(1)[0][0]


def generate_timeline(file_path, selected_user):
    messages = parse_chat_messages(file_path)
    grouped_data = group_messages(messages, selected_user)
    
    timeline = []
    all_chats = []
    all_groups = []
    
    for group_id in sorted(grouped_data.keys()):
        msgs = grouped_data[group_id]
        msg_texts = [m["message"] for m in msgs]
        
        # Predict for each message
        predictions = predict_batch_mbti(msg_texts)
        
        # Calculate dominant for group
        valid_preds = [p for p in predictions if p != "Neutral"]
        if valid_preds:
            from collections import Counter
            dominant_mbti = Counter(valid_preds).most_common(1)[0][0]
        else:
            dominant_mbti = "Unknown"
            
        start_date = msgs[0]["start_date"]
        end_date = msgs[0]["end_date"]
        
        # Filter messages matching dominant MBTI
        matching_msgs = []
        for m, pred in zip(msgs, predictions):
            if pred == dominant_mbti:
                matching_msgs.append(m["message"])
                
        # Get top 2 samples
        sample_messages = matching_msgs[:2]
        
        timeline.append({
            "group_id": group_id,
            "start_date": start_date,
            "end_date": end_date,
            "dominant_mbti": dominant_mbti,
            "sample_messages": sample_messages,
            "all_messages": matching_msgs
        })
        
        # Collect data for CSVs
        all_groups.append([group_id, start_date, end_date, len(msgs)])
        
        for m, pred in zip(msgs, predictions):
            all_chats.append([m["datetime"].strftime("%d/%m/%Y"), selected_user, m["message"], pred])

    # Save CSVs
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(base_dir, GROUPS), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Group_ID", "Start_Date", "End_Date", "Message_Count"])
            writer.writerows(all_groups)
            
        with open(os.path.join(base_dir, TIMELINE), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Group_ID", "Start_Date", "End_Date", "Dominant_MBTI"])
            for t in timeline:
                writer.writerow([t["group_id"], t["start_date"], t["end_date"], t["dominant_mbti"]])
                
        with open(os.path.join(base_dir, ALL_CHATS), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Sender", "Message", "Predicted_MBTI"])
            writer.writerows(all_chats)
            
        print("CSVs generated successfully.")
    except Exception as e:
        print(f"Error generating CSVs: {e}")

    return timeline
