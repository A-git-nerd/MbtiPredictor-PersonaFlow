import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
import os
import re

# --- Config ---
MODEL_NAME = "bert-base-multilingual-cased"
DATA_FILE = "train.csv"
OUTPUT_DIR = "mbti_model"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# --- Neutral Words Filter ---
# These are words/phrases that are too generic to indicate personality
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
    # Remove punctuation
    text_clean = re.sub(r'[^\w\s]', '', text)
    
    if not text_clean:
        return True
        
    if text in NEUTRAL_WORDS:
        return True
        
    if text_clean in NEUTRAL_WORDS:
        return True
        
    # Filter very short messages (less than 3 chars) 
    if len(text_clean) < 2: 
        return True
        
    return False

def load_and_preprocess_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    df.dropna(subset=['english', 'mbti'], inplace=True)
    
    text_col = 'roman_urdu'
    initial_count = len(df)
    print(f"Initial records: {initial_count}")
    
    # Filter neutral messages
    df['is_neutral'] = df[text_col].apply(is_neutral)
    df = df[~df['is_neutral']]
    
    # Filter valid MBTI labels
    valid_mbtis = {'ISTJ', 'ISFJ', 'INFJ', 'INTJ', 
                   'ISTP', 'ISFP', 'INFP', 'INTP', 
                   'ESTP', 'ESFP', 'ENFP', 'ENTP', 
                   'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ'}
    df = df[df['mbti'].isin(valid_mbtis)]
    
    print(f"Records after filtering neutral/short messages and invalid labels: {len(df)} (Removed {initial_count - len(df)})")
    
    return df, text_col

class MBTIDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    df, text_col = load_and_preprocess_data(DATA_FILE)
    
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['mbti'])
    
    np.save(os.path.join(OUTPUT_DIR, 'classes.npy'), label_encoder.classes_)
    print(f"Classes: {label_encoder.classes_}")
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[text_col].tolist(), 
        df['label'].tolist(), 
        test_size=0.1, 
        random_state=42,
        stratify=df['label'] 
    )
    
    class_counts = np.bincount(df['label'])
    total_samples = len(df)
    num_classes = len(label_encoder.classes_)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"Class Weights: {class_weights}")
    
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=MAX_LEN)
    
    train_dataset = MBTIDataset(train_encodings, train_labels)
    val_dataset = MBTIDataset(val_encodings, val_labels)
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(label_encoder.classes_)
    )
    
    # Custom Trainer for Class Weights 
    # CUZ there are many INTJ msgs
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    #Training
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=0
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
