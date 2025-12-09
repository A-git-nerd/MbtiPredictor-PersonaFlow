import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Config ---
MODEL_DIR = "../Train/mbti_model"
MAX_LEN = 128

def predict_mbti(text):
    print(f"Input text: '{text}'\n")
    
    # Load model and tokenizer
    print("Loading model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # Load class labels
    classes = np.load(os.path.join(MODEL_DIR, "classes.npy"), allow_pickle=True)
    print(f"Model loaded (Device: {device})\n")
    
    # Tokenize input
    print("Tokenizing input...")
    encoding = tokenizer(
        text, 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN, 
        return_tensors='pt'
    )
    print(f"Token IDs: {encoding['input_ids'].tolist()[0][:20]}... (showing first 20)\n")
    
    # model prediction
    print("Running model inference...")
    with torch.no_grad():
        # Move input to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        pred_confidence = probs[pred_idx]
    
    print(f"Model inference complete\n")
    
    print("-" * 60)
    print("PREDICTION RESULTS")
    print("-" * 60)
    print(f"\nPredicted MBTI Type: {pred_label}")
    print(f"Confidence: {pred_confidence:.2%}\n")
    
    # top 5 predictions
    print("Top 5 Predictions:")
    print("-" * 40)
    top5_indices = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(top5_indices, 1):
        print(f"{rank}. {classes[idx]:5s} - {probs[idx]:.2%}")
        
    # Return results
    return {
        'predicted_type': pred_label,
        'confidence': float(pred_confidence),
        'all_probabilities': {classes[i]: float(probs[i]) for i in range(len(classes))},
        'top_5': [(classes[i], float(probs[i])) for i in top5_indices]
    }


if __name__ == "__main__":
    result1 = predict_mbti("bro kidhr ho?")