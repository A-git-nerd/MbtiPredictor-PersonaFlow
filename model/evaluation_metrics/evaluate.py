import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json

# --- Config ---
MODEL_DIR = "../../mbti_model"  
VALID_DATA = "../Train/valid.csv"
OUTPUT_DIR = "."
MAX_LEN = 128
BATCH_SIZE = 16

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

def evaluate_model():
    print("=" * 60)
    print("MBTI Model Evaluation")
    print("=" * 60)
    
    # Load validation data
    print(f"\n1. Loading validation data from {VALID_DATA}...")
    df = pd.read_csv(VALID_DATA)
    df.dropna(subset=['roman_urdu', 'mbti'], inplace=True)
    
    print(f"   Total validation samples: {len(df)}")
    
    # Load model and tokenizer
    print(f"\n2. Loading model from {MODEL_DIR}...")
    
    # Force CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"   Using device: cuda:0 (GPU)")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"   Using device: cpu (WARNING: This will be slow!)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # Load classes
    classes = np.load(os.path.join(MODEL_DIR, "classes.npy"), allow_pickle=True)
    print(f"   Classes: {classes}")
    
    # Create label mapping and filter out invalid labels
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    df['label'] = df['mbti'].map(label_to_idx)
    
    # Remove rows with NaN labels (MBTI types not in training set)
    initial_count = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"   Removed {removed_count} samples with invalid MBTI types")
    
    # Tokenize
    print("\n3. Tokenizing validation data...")
    texts = df['roman_urdu'].tolist()
    labels = df['label'].tolist()
    
    print(f"   Tokenizing {len(texts)} samples...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors='pt')
    print(f"   ✓ Tokenization complete")
    
    # Predict
    print("\n4. Running predictions...")
    print(f"   Processing {len(texts)} samples in batches of {BATCH_SIZE}")
    print(f"   Total batches: {(len(texts) + BATCH_SIZE - 1) // BATCH_SIZE}")
    
    all_preds = []
    all_labels = []
    
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE)):
            # Progress indicator
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f"   Batch {batch_idx + 1}/{num_batches} ({((batch_idx + 1) / num_batches * 100):.1f}%)")
            
            batch_encodings = {k: v[i:i+BATCH_SIZE].to(device) for k, v in encodings.items()}
            batch_labels = labels[i:i+BATCH_SIZE]
            
            outputs = model(**batch_encodings)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_labels)
    
    print(f"   ✓ Predictions complete!")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    
    print("\n5. Calculating metrics...")
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    for i, cls in enumerate(classes):
        print(f"{cls:5s} - Precision: {precision_per_class[i]:.4f}, Recall: {recall_per_class[i]:.4f}, F1: {f1_per_class[i]:.4f}, Support: {support_per_class[i]}")
    
    # Saving metrics to JSON
    metrics_data = {
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "per_class": {
            classes[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i]),
                "support": int(support_per_class[i])
            } for i in range(len(classes))
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n✓ Metrics saved to {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    
    #  visualizations
    print("\n6. Generating visualizations...")
    
    # Overall Metrics Bar Chart
    plt.figure(figsize=(10, 6))
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = [accuracy, precision, recall, f1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
    plt.ylim(0, 1.0)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('MBTI Model - Overall Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adding value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "overall_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: overall_metrics.png")
    
    # Per-Class F1 Scores
    plt.figure(figsize=(12, 6))
    sorted_indices = np.argsort(f1_per_class)[::-1]
    sorted_classes = classes[sorted_indices]
    sorted_f1 = f1_per_class[sorted_indices]
    
    colors_gradient = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_classes)))
    bars = plt.barh(sorted_classes, sorted_f1, color=colors_gradient, edgecolor='black')
    plt.xlabel('F1 Score', fontsize=12, fontweight='bold')
    plt.ylabel('MBTI Type', fontsize=12, fontweight='bold')
    plt.title('F1 Score by MBTI Type', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Adding value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_f1)):
        plt.text(val + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}',
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "f1_per_class.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: f1_per_class.png")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - MBTI Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: confusion_matrix.png")
    
    # Precision, Recall, F1 Comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(classes))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision_per_class, width, label='Precision', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, recall_per_class, width, label='Recall', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, f1_per_class, width, label='F1 Score', color='#f39c12', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('MBTI Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, and F1 Score by MBTI Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: metrics_comparison.png")
    
    # Classification Report (txt file)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write("MBTI Model Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"   ✓ Saved: classification_report.txt")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files in '{OUTPUT_DIR}':")
    print("  - metrics.json")
    print("  - overall_metrics.png")
    print("  - f1_per_class.png")
    print("  - confusion_matrix.png")
    print("  - metrics_comparison.png")
    print("  - classification_report.txt")
    print("\n")

if __name__ == "__main__":
    evaluate_model()
