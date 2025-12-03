import pandas as pd
import re

# cleaning train.csv
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

df = pd.read_csv("train.csv")
df.dropna(subset=['roman_urdu', 'mbti'], inplace=True)
df['is_neutral'] = df['roman_urdu'].apply(is_neutral)
df_clean = df[~df['is_neutral']]

print("Original counts:")
print(df['mbti'].value_counts())
print("\nCleaned counts:")
print(df_clean['mbti'].value_counts())
