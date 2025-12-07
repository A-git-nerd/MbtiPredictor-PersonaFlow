import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def log(x):
    print(f"[LOG] {x}", flush=True)

def load_model():
    log("Loading model...")
    tok = AutoTokenizer.from_pretrained("../Train/mbti_model")
    mdl = AutoModelForSequenceClassification.from_pretrained("../Train/mbti_model")
    mdl.eval()
    
    # Load classes from classes.npy instead of labels.csv
    classes = np.load("../Train/mbti_model/classes.npy", allow_pickle=True)
    m = {i: label for i, label in enumerate(classes)}
    log(f"Loaded model with classes: {classes}")
    return tok, mdl, m

def predict_batch(model, tokenizer, texts, batch_size=32):
    all_preds = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        ids = []
        mask = []
        for t in batch_texts:
            k = tokenizer(
                t,
                padding="max_length",
                truncation=True,
                max_length=96,
                return_tensors="pt"
            )
            ids.append(k["input_ids"])
            mask.append(k["attention_mask"])
        ids = torch.cat(ids, dim=0)
        mask = torch.cat(mask, dim=0)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        p = torch.argmax(out.logits, dim=1)
        all_preds.extend(p.tolist())
    
    return all_preds

def process_group(df):
    g = df.groupby("group_id")
    total_groups = len(g)
    log(f"Processing {total_groups} groups...")
    out = []
    processed = 0
    for gid, d in g:
        processed += 1
        num_messages = len(d)
        log(f"Processing group {processed}/{total_groups} (Group ID: {gid}, Messages: {num_messages})")
        
        msgs = []
        for i in range(len(d)):
            ru = str(d.iloc[i]["message"])
            en = ru
            c = ru + " [SEP] " + en
            msgs.append(c)
        
        if not msgs:
            out.append([gid, d.iloc[0]["start_date"], d.iloc[0]["end_date"], "UNKNOWN"])
            continue
        
        log(f"  Predicting MBTI for {len(msgs)} messages...")
        preds = predict_batch(model, tokenizer, msgs)
        log(f"  Predictions complete, calculating dominant type...")
        
        freq = {}
        for p in preds:
            if p not in freq:
                freq[p] = 0
            freq[p] += 1
        dom = max(freq, key=freq.get)
        dominant_type = label_map[dom]
        log(f"  Group {gid} dominant type: {dominant_type}")
        out.append([gid, d.iloc[0]["start_date"], d.iloc[0]["end_date"], dominant_type])
    
    log(f"Completed processing all {total_groups} groups")
    return out

tokenizer, model, label_map = load_model()

def infer(in_csv, out_csv):
    log(f"Reading input CSV: {in_csv}")
    df = pd.read_csv(in_csv)
    log(f"Loaded {len(df)} rows from CSV")
    log(f"CSV columns: {df.columns.tolist()}")
    
    r = process_group(df)
    
    log(f"Saving results to {out_csv}")
    pd.DataFrame(r, columns=["group_id","start_date","end_date","dominant_mbti"]).to_csv(out_csv, index=False)
    log(f"DONE! Results saved to {out_csv}")

if __name__ == "__main__":
    infer("../Group/groups.csv", "../Infer/results.csv")
