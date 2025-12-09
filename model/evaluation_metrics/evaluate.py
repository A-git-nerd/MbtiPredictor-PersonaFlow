import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import json
from itertools import cycle
from sklearn.preprocessing import label_binarize

# --- Config ---
MODEL_DIR = "../Train/mbti_model"  
VALID_DATA = "../Train/valid.csv"
OUTPUT_DIR = "."
MAX_LEN = 128
BATCH_SIZE = 50

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
    
    # Force CUDA:0 if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)  # Explicitly set to GPU 0
        print(f"   Using device: cuda:0 (GPU)")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"   Using device: cpu (WARNING: This will be slow!)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # Verify model is on correct device
    print(f"   Model device: {next(model.parameters()).device}")
    
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
    print(f"   Tokenization complete")
    
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
    
    print(f"   Predictions complete!")
    
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
    print(f"\n  Metrics saved to {os.path.join(OUTPUT_DIR, 'metrics.json')}")
    
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
    print(f"   Saved: overall_metrics.png")
    
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
    print(f"   Saved: f1_per_class.png")
    
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
    print(f"    Saved: confusion_matrix.png")
    
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
    print(f"    Saved: metrics_comparison.png")
    
    # Classification Report (txt file)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write("MBTI Model Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"   Saved: classification_report.txt")
    
    # Load training history from checkpoint
    print("\n7. Loading training history...")
    training_history = load_training_history()
    
    if training_history:
        generate_training_visualizations(training_history, OUTPUT_DIR)
    else:
        print("   Training history not found. Skipping training visualizations.")
    
    # Generate additional evaluation visualizations
    print("\n8. Generating advanced evaluation visualizations...")
    
    # Class distribution comparison
    generate_class_distribution(all_labels, all_preds, classes, OUTPUT_DIR)
    
    # Prediction confidence analysis
    generate_confidence_analysis(model, encodings, device, classes, OUTPUT_DIR)
    
    # Error analysis
    generate_error_analysis(all_labels, all_preds, df['roman_urdu'].tolist(), classes, OUTPUT_DIR)
    
    # ROC curves (one-vs-rest)
    generate_roc_curves(model, encodings, labels, device, classes, OUTPUT_DIR)
    
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
    if training_history:
        print("  - training_loss.png")
        print("  - validation_loss.png")
        print("  - learning_rate_schedule.png")
        print("  - training_metrics.png")
    print("  - class_distribution.png")
    print("  - prediction_confidence.png")
    print("  - error_analysis.png")
    print("  - roc_curves.png")
    print("\n")

def load_training_history():
    """Load training history from trainer_state.json"""
    checkpoint_dirs = ["../Train/results/checkpoint-10608", 
                      "../Train/results/checkpoint-7072",
                      "../Train/results/checkpoint-3536"]
    
    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            print(f"   Found training history at: {trainer_state_path}")
            with open(trainer_state_path, 'r') as f:
                return json.load(f)
    return None

def generate_training_visualizations(training_history, output_dir):
    """Generate training loss, validation loss, and learning rate visualizations"""
    log_history = training_history.get('log_history', [])
    
    if not log_history:
        print("   No log history found")
        return
    
    # Extract training and validation data
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    learning_rates = []
    lr_steps = []
    
    train_metrics = {'accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []}
    val_metrics = {'accuracy': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []}
    metrics_steps = []
    
    for entry in log_history:
        step = entry.get('step', entry.get('epoch', 0))
        
        # Training loss
        if 'loss' in entry:
            train_steps.append(step)
            train_losses.append(entry['loss'])
        
        # Validation loss
        if 'eval_loss' in entry:
            val_steps.append(step)
            val_losses.append(entry['eval_loss'])
            
            # Collect validation metrics
            if 'eval_accuracy' in entry:
                if step not in metrics_steps:
                    metrics_steps.append(step)
                    for metric in val_metrics.keys():
                        val_metrics[metric].append(entry.get(f'eval_{metric}', 0))
        
        # Learning rate
        if 'learning_rate' in entry:
            lr_steps.append(step)
            learning_rates.append(entry['learning_rate'])
    
    # 1. Training Loss Graph
    if train_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, train_losses, linewidth=2, color='#e74c3c', label='Training Loss')
        plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: training_loss.png")
    
    # 2. Validation Loss Graph
    if val_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(val_steps, val_losses, linewidth=2, color='#3498db', marker='o', 
                markersize=8, label='Validation Loss')
        plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Validation Loss Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "validation_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: validation_loss.png")
    
    # 3. Combined Train/Val Loss
    if train_losses and val_losses:
        plt.figure(figsize=(12, 6))
        plt.plot(train_steps, train_losses, linewidth=2, color='#e74c3c', 
                label='Training Loss', alpha=0.8)
        plt.plot(val_steps, val_losses, linewidth=2, color='#3498db', marker='o',
                markersize=6, label='Validation Loss')
        plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "train_val_loss.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: train_val_loss.png")
    
    # 4. Learning Rate Schedule
    if learning_rates:
        plt.figure(figsize=(12, 6))
        plt.plot(lr_steps, learning_rates, linewidth=2, color='#2ecc71')
        plt.xlabel('Training Steps', fontsize=12, fontweight='bold')
        plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learning_rate_schedule.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: learning_rate_schedule.png")
    
    # 5. Training Metrics Over Time
    if metrics_steps and any(val_metrics.values()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics_info = [
            ('accuracy', 'Accuracy', '#3498db'),
            ('f1_macro', 'F1 Score (Macro)', '#2ecc71'),
            ('precision_macro', 'Precision (Macro)', '#f39c12'),
            ('recall_macro', 'Recall (Macro)', '#e74c3c')
        ]
        
        for idx, (metric_key, metric_name, color) in enumerate(metrics_info):
            ax = axes[idx // 2, idx % 2]
            if val_metrics[metric_key]:
                ax.plot(metrics_steps, val_metrics[metric_key], linewidth=2, 
                       color=color, marker='o', markersize=6)
                ax.set_xlabel('Training Steps', fontsize=10, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
                ax.set_title(f'{metric_name} Over Time', fontsize=11, fontweight='bold')
                ax.grid(alpha=0.3, linestyle='--')
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   Saved: training_metrics.png")

def generate_class_distribution(true_labels, pred_labels, classes, output_dir):
    """Generate class distribution comparison"""
    true_counts = np.bincount(true_labels, minlength=len(classes))
    pred_counts = np.bincount(pred_labels, minlength=len(classes))
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', 
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Labels',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('MBTI Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('True vs Predicted Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: class_distribution.png")

def generate_confidence_analysis(model, encodings, device, classes, output_dir):
    """Generate prediction confidence analysis"""
    print("   Analyzing prediction confidence...")
    
    all_confidences = []
    with torch.no_grad():
        for i in range(0, len(encodings['input_ids']), BATCH_SIZE):
            batch_encodings = {k: v[i:i+BATCH_SIZE].to(device) for k, v in encodings.items()}
            outputs = model(**batch_encodings)
            probs = torch.softmax(outputs.logits, dim=-1)
            max_probs = probs.max(dim=-1).values.cpu().numpy()
            all_confidences.extend(max_probs)
    
    # Confidence distribution
    plt.figure(figsize=(12, 6))
    plt.hist(all_confidences, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(all_confidences), color='#e74c3c', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(all_confidences):.3f}')
    plt.axvline(np.median(all_confidences), color='#2ecc71', linestyle='--',
               linewidth=2, label=f'Median: {np.median(all_confidences):.3f}')
    plt.xlabel('Prediction Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_confidence.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: prediction_confidence.png")

def generate_error_analysis(true_labels, pred_labels, texts, classes, output_dir):
    """Generate error analysis visualization"""
    errors = true_labels != pred_labels
    error_rate = errors.sum() / len(errors)
    
    # Error rate by class
    error_rates_by_class = []
    for i, cls in enumerate(classes):
        mask = true_labels == i
        if mask.sum() > 0:
            class_error_rate = (errors & mask).sum() / mask.sum()
            error_rates_by_class.append(class_error_rate)
        else:
            error_rates_by_class.append(0)
    
    plt.figure(figsize=(12, 6))
    colors = ['#e74c3c' if rate > error_rate else '#2ecc71' for rate in error_rates_by_class]
    bars = plt.bar(classes, error_rates_by_class, color=colors, alpha=0.8, edgecolor='black')
    plt.axhline(error_rate, color='#3498db', linestyle='--', linewidth=2, 
               label=f'Overall Error Rate: {error_rate:.3f}')
    plt.xlabel('MBTI Type', fontsize=12, fontweight='bold')
    plt.ylabel('Error Rate', fontsize=12, fontweight='bold')
    plt.title('Error Rate by MBTI Type', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: error_analysis.png")

def generate_roc_curves(model, encodings, labels, device, classes, output_dir):
    """Generate ROC curves for multi-class classification (One-vs-Rest)"""
    print("   Generating ROC curves...")
    
    # Get all predictions and probabilities
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(encodings['input_ids']), BATCH_SIZE):
            batch_encodings = {k: v[i:i+BATCH_SIZE].to(device) for k, v in encodings.items()}
            outputs = model(**batch_encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
    
    all_probs = np.array(all_probs)
    labels_array = np.array(labels)
    
    # Binarize labels for ROC
    labels_bin = label_binarize(labels_array, classes=range(len(classes)))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(14, 10))
    colors = cycle(['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', 
                    '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#d35400',
                    '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#16a085', '#f39c12'])
    
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('ROC Curves - One-vs-Rest (MBTI Classification)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: roc_curves.png")

if __name__ == "__main__":
    evaluate_model()