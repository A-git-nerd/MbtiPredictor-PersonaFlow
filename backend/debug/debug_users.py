import mbti_service
import os

file_path = os.path.join("uploads", "chatBoys.txt")
users = mbti_service.parse_chat_users(file_path)
print("Users found:", users)

target_name = "Saim SE" 
for u in users:
    if "shobaan" in u.lower() or "shoban" in u.lower():
        print(f"Found similar user: {u}")
        target_name = u

messages = mbti_service.parse_chat_messages(file_path)
shobaan_msgs = [m for m in messages if m["sender"] == target_name]

if shobaan_msgs:
    print(f"Messages for {target_name}: {len(shobaan_msgs)}")
    print(f"First message: {shobaan_msgs[0]['datetime']}")
    print(f"Last message: {shobaan_msgs[-1]['datetime']}")
else:
    print(f"No messages found for {target_name}")
