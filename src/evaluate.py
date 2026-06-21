import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def assign_card_risk_tier(prob):
    if prob < 0.25: return "Safe"
    elif prob < 0.70: return "Suspicious"
    else: return "Critical Risk"

def generate_report_plots(y_test, lstm_preds, lstm_probs, rf_preds, rf_probs, train_losses, val_losses, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Class Distribution Plot (Using direct target label data instead of dataframe)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_test, hue=y_test, palette=['#0A192F', '#FF007F'], legend=False)
    plt.title('Class Distribution Graph (Highly Imbalanced Test Array State)')
    plt.xlabel('Class (0 = Legitimate, 1 = Fraud)')
    plt.ylabel('Total Transactions Count')
    plt.savefig(f"{output_dir}/class_distribution_graph.png")
    plt.close()
    
    # 2. Training vs Validation Loss Curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Training Loss', color='#FF007F', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='#0A192F', marker='x')
    plt.title('LSTM Model: Training vs Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross Entropy Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/training_vs_validation_loss.png")
    plt.close()
    
    # 3. Confusion Matrix Layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm_lstm = confusion_matrix(y_test, lstm_preds)
    sns.heatmap(cm_lstm, annot=True, fmt='d', ax=axes[0], cmap='Reds')
    axes[0].set_title('LSTM Confusion Matrix')
    axes[0].set_xlabel('Predicted Classes')
    axes[0].set_ylabel('Actual Ground Truth Labels')
    
    cm_rf = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm_rf, annot=True, fmt='d', ax=axes[1], cmap='Blues')
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted Classes')
    axes[1].set_ylabel('Actual Ground Truth Labels')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png")
    plt.close()
    
    # 4. ROC-AUC Curve Performance Evaluation
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_probs)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM Model (AUC = {roc_auc_score(y_test, lstm_probs):.4f})', color='magenta')
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest Model (AUC = {roc_auc_score(y_test, rf_probs):.4f})', color='cyan', linestyle='--')
    plt.plot([0, 1], [0, 1], linestyle=':', color='gray')
    plt.title('ROC-AUC Curve Performance Evaluation Profile')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f"{output_dir}/roc_auc_curve.png")
    plt.close()
    
    # 5. Live Simulated Predictions Panel
    dashboard = pd.DataFrame({
        'Index': np.arange(len(lstm_probs[:15])),
        'Raw_Fraud_Probability': lstm_probs[:15],
        'Assigned_Risk_Tier': [assign_card_risk_tier(p) for p in lstm_probs[:15]],
        'Actual_Ground_Truth': y_test[:15]
    })
    print("\n[SAMPLE FRAUD PREDICTIONS DASHBOARD]")
    print(dashboard.to_string(index=False))