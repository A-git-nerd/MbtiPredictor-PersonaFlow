import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def log(x):
    print(f"[LOG] {x}", flush=True)

def load_model():
    tok = XLMRobertaTokenizer.from_pretrained("./mbti_model")
    mdl = XLMRobertaForSequenceClassification.from_pretrained("./mbti_model")
    mdl.eval()
    lbl = pd.read_csv("./mbti_model/labels.csv")
    m = {row["id"]: row["label"] for _, row in lbl.iterrows()}
    return tok, mdl, m

def predict_batch(model, tokenizer, texts):
    ids = []
    mask = []
    for t in texts:
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
    return p.tolist()

def process_group(df):
    g = df.groupby("group_id")
    out = []
    for gid, d in g:
        msgs = []
        for i in range(len(d)):
            ru = str(d.iloc[i]["message"])
            en = ru
            c = ru + " [SEP] " + en
            msgs.append(c)
        if not msgs:
            out.append([gid, d.iloc[0]["start_date"], d.iloc[0]["end_date"], "UNKNOWN"])
            continue
        preds = predict_batch(model, tokenizer, msgs)
        freq = {}
        for p in preds:
            if p not in freq:
                freq[p] = 0
            freq[p] += 1
        dom = max(freq, key=freq.get)
        out.append([gid, d.iloc[0]["start_date"], d.iloc[0]["end_date"], label_map[dom]])
    return out

tokenizer, model, label_map = load_model()

def infer(in_csv, out_csv):
    df = pd.read_csv(in_csv)
    r = process_group(df)
    pd.DataFrame(r, columns=["group_id","start_date","end_date","dominant_mbti"]).to_csv(out_csv, index=False)
    log("DONE")

if __name__ == "__main__":
    infer("../Group/groups.csv", "../Infer/results.csv")
