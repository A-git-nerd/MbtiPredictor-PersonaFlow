# MBTI Model Evaluation

This folder contains scripts and outputs for evaluating the MBTI classification model.

## Files

- `evaluate.py` - Main evaluation script
- `metrics.json` - Detailed metrics in JSON format
- `overall_metrics.png` - Bar chart of overall performance
- `f1_per_class.png` - F1 scores for each MBTI type
- `confusion_matrix.png` - Confusion matrix heatmap
- `metrics_comparison.png` - Precision, Recall, F1 comparison by class
- `classification_report.txt` - Detailed classification report

## How to Run Evaluation

### Prerequisites
Make sure you have the required packages:
```bash
pip install pandas torch transformers scikit-learn matplotlib seaborn numpy
```

### Running the Evaluation

1. Navigate to this directory:
```bash
cd model/evaluation_metrics
```

2. Run the evaluation script:
```bash
python evaluate.py
```

The script will:
- Load the trained model from `../mbti_model`
- Evaluate on the validation set from `../Train/valid.csv`
- Generate metrics and visualizations
- Save all outputs to this folder

## Metrics Explained

### Overall Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: Of all predicted positives, how many were actually positive
- **Recall**: Of all actual positives, how many were correctly identified
- **F1 Score**: Harmonic mean of precision and recall (balanced metric)

### Per-Class Metrics
Individual precision, recall, and F1 scores for each of the 16 MBTI types.

### Confusion Matrix
Shows where the model makes mistakes - which MBTI types get confused with each other.

## Understanding the Results

### Accuracy
**What it is:** The percentage of all predictions that were correct.

**Formula:** `(Correct Predictions) / (Total Predictions)`

**Example:** If the model made 100 predictions and 85 were correct, accuracy = 85%

**When to use:** Good for balanced datasets where all classes have similar numbers of samples.

**Limitation:** Can be misleading with imbalanced data. If 95% of your data is INTJ, a model that always predicts INTJ would have 95% accuracy but be useless!

---

### Precision
**What it is:** Of all the times the model predicted a specific MBTI type, how many were actually that type?

**Formula:** `True Positives / (True Positives + False Positives)`

**Example:** Model predicted "INTJ" 100 times. Of those, 80 were actually INTJ → Precision = 80%

**Real-world meaning:** "When the model says someone is INTJ, how confident can I be that they really are INTJ?"

**High precision = Few false alarms**

---

### Recall (Sensitivity)
**What it is:** Of all the actual instances of a specific MBTI type, how many did the model correctly identify?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Example:** There were 100 actual INTJs in the data. Model correctly identified 75 of them → Recall = 75%

**Real-world meaning:** "Of all the real INTJs, how many did the model catch?"

**High recall = Few missed cases**

---

### F1 Score
**What it is:** The harmonic mean of precision and recall. Balances both metrics.

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**Why use it:** 
- Better than accuracy for imbalanced datasets
- Balances precision and recall (you need both to be good)
- A model with high precision but low recall (or vice versa) will have a low F1 score

**Interpretation:**
- **F1 > 0.8**: Excellent performance
- **F1 = 0.6-0.8**: Good performance
- **F1 < 0.6**: Needs improvement

---

### Confusion Matrix
**What it is:** A table showing where your model makes mistakes.

**How to read it:**
```
                Predicted
              INTJ  INFJ  ENTJ  ...
         INTJ  [85]   10     5   ...  ← 85 correctly predicted as INTJ
Actual   INFJ   12  [78]    8   ...  ← 78 correctly predicted as INFJ
         ENTJ    7    5   [88]  ...  ← 88 correctly predicted as ENTJ
         ...
```

**Diagonal (bold numbers):** Correct predictions
**Off-diagonal numbers:** Mistakes

**What to look for:**
1. **Dark diagonal:** Good! Most predictions are correct
2. **Bright off-diagonal cells:** Problem areas - these types are being confused
3. **Patterns:** 
   - If INTJ row has high values in INTP column → Model confuses INTJ with INTP
   - If ESFP column has low values everywhere → Model rarely predicts ESFP

**Example insights:**
- "The model confuses INTJ with INTP (both introverted thinkers)"
- "ESFP has low recall - model misses many ESFPs"
- "ENTJ has high precision - when model says ENTJ, it's usually right"

---

### Practical Example

Imagine evaluating INTJ predictions:
- **100 actual INTJs** in test data
- **Model predicted INTJ 120 times**
- **Of those 120 predictions, 80 were correct**

Results:
- **Precision = 80/120 = 66.7%** (when model says INTJ, it's right 67% of the time)
- **Recall = 80/100 = 80%** (model found 80% of all real INTJs)
- **F1 Score = 2×(0.667×0.8)/(0.667+0.8) = 0.727** (balanced score)

**What this means:**
- Model is good at finding INTJs (80% recall)
- But has some false alarms (67% precision)
- Check confusion matrix to see what it's confusing INTJ with

---

## Understanding the Results

- **F1 Score > 0.8**: Excellent performance
- **F1 Score 0.6-0.8**: Good performance
- **F1 Score < 0.6**: Needs improvement

Check the confusion matrix to see which personality types are being confused.
