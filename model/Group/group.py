import os
import re
import csv
from datetime import datetime, timedelta
from collections import defaultdict

# for grouping messages based on 2 months interval
CHAT_FILE = "../data/chatAbdur.txt"
OUTPUT_DIR = "model/Group/chat_by_sender"

# Making output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex to parse WhatsApp line
line_regex = re.compile(r"(\d{2}/\d{2}/\d{4}),\s*(\d{1,2}:\d{2}\s*[apAP][mM])\s*-\s*(.*?):\s*(.*)")

with open(CHAT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

messages = []
for line in lines:
    line = line.strip()
    match = line_regex.match(line)
    if match:
        date_str, time_str, sender, msg = match.groups()
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        messages.append({"datetime": dt, "sender": sender.strip(), "message": msg.strip()})

# Find the earliest date to start grouping
if not messages:
    raise ValueError("No valid messages found in chat.txt")

start_date = min(msg["datetime"] for msg in messages)
# Group messages into 2-month intervals
def get_group_id(msg_date):
    delta_months = (msg_date.year - start_date.year) * 12 + (msg_date.month - start_date.month)
    return delta_months // 2

# Collect messages per sender
sender_messages = defaultdict(list)

for msg in messages:
    group_id = get_group_id(msg["datetime"])
    # Calculate group start and end dates
    group_start_month = start_date.month + (group_id * 2)
    group_start_year = start_date.year + (group_start_month - 1) // 12
    group_start_month = ((group_start_month - 1) % 12) + 1
    group_start_date = datetime(group_start_year, group_start_month, 1)
    # End date = last day of second month
    end_month = group_start_month + 1
    end_year = group_start_year + (end_month - 1) // 12
    end_month = ((end_month - 1) % 12) + 1
    # last day of month
    group_end_date = datetime(end_year, end_month, 28) + timedelta(days=4)  
    group_end_date = group_end_date - timedelta(days=group_end_date.day)    

    sender_messages[msg["sender"]].append({
        "group_id": group_id,
        "start_date": group_start_date.strftime("%d/%m/%Y"),
        "end_date": group_end_date.strftime("%d/%m/%Y"),
        "message": msg["message"]
    })


for sender, msgs in sender_messages.items():
    safe_name = re.sub(r"[^\w\-]", "-", sender)
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["group_id", "start_date", "end_date", "message"])
        writer.writeheader()
        for m in msgs:
            writer.writerow(m)

print(f"Done! CSV files are in the folder '{OUTPUT_DIR}'")
