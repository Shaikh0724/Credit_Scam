import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd

from src.preprocessing import load_and_preprocess_card_data
from src.models import CardShieldLSTM, get_random_forest_model
from src.evaluate import generate_report_plots

def main():
    data_path = "data/creditcard.csv"
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Preprocessing now returns scalers for persistence
    X_train_seq, X_test_seq, X_train_res, X_test, y_train_res, y_test, amount_scaler, time_scaler = load_and_preprocess_card_data(data_path)
    
    # Save scalers immediately so inference uses the exact same fitted state
    joblib.dump(amount_scaler, "saved_models/amount_scaler.pkl")
    joblib.dump(time_scaler, "saved_models/time_scaler.pkl")
    print("[SCALERS] Amount and Time scalers saved to saved_models/\n")
    
    val_X_seq, val_y = X_test_seq[:5000], y_test[:5000]
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_res)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_X_seq), torch.FloatTensor(val_y)), batch_size=1024, shuffle=False)
    
    # ── LSTM Training ──
    print("[LSTM ARCHITECTURE SPECIFICATIONS]")
    print("Sequential multi-feature tracking input space topology mapping.")
    print("Recurrent network architecture initialized with 32 units storage space gates.")
    print("Sigmoid normalization mapping layer applied to final state sequence vectors.\n")
    
    model_lstm = CardShieldLSTM(input_dim=X_train_seq.shape[2], hidden_dim=32)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_lstm.parameters(), lr=0.005)
    
    # Extended training with early stopping
    EPOCHS = 15
    PATIENCE = 3
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("[TRAINING ITERATION PROGRESS LOGS]")
    train_losses, val_losses = [], []
    for epoch in range(EPOCHS):
        model_lstm.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model_lstm(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        model_lstm.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_batch_X, val_batch_y in val_loader:
                val_preds = model_lstm(val_batch_X)
                v_loss = criterion(val_preds, val_batch_y)
                epoch_val_loss += v_loss.item()
                
        avg_train = epoch_train_loss / len(train_loader)
        avg_val = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch+1}/{EPOCHS} | Training Loss: {avg_train:.4f} | Validation Loss: {avg_val:.4f}")
        
        # Early stopping: save best model, stop if no improvement
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model_lstm.state_dict(), "saved_models/card_lstm_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
                break
    
    # Load best checkpoint for evaluation
    model_lstm.load_state_dict(torch.load("saved_models/card_lstm_model.pth", map_location="cpu"))
    model_lstm.eval()
    with torch.no_grad():
        lstm_probs = model_lstm(torch.FloatTensor(X_test_seq)).numpy()
    lstm_preds = (lstm_probs >= 0.5).astype(int)
    
    # ── Random Forest ──
    print("\n[MACHINE LEARNING SELECTION METRICS]")
    print("Baseline Selection: Random Forest Classifier")
    print("Ensemble parameters resolve non-linear cross-feature boundaries without outlier distortion.")
    print("Bootstrap aggregation trees naturally mitigate variance under severe minority anomalies.\n")
    
    rf_model = get_random_forest_model()
    rf_model.fit(X_train_res, y_train_res)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # ── XGBoost ──
    print("[XGBOOST GRADIENT BOOSTED ENSEMBLE]")
    print("XGBoost classifier with scale_pos_weight for class imbalance handling.")
    print("Gradient boosting with regularization for robust fraud boundary detection.\n")
    
    neg_count = int(np.sum(y_train_res == 0))
    pos_count = int(np.sum(y_train_res == 1))
    scale_ratio = neg_count / pos_count if pos_count > 0 else 1.0
    
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        scale_pos_weight=scale_ratio,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
    )
    xgb_model.fit(X_train_res, y_train_res)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # ── Performance Report ──
    print("[PERFORMANCE EVALUATION METRICS REPORT]")
    lstm_rep = classification_report(y_test, lstm_preds, output_dict=True)
    rf_rep = classification_report(y_test, rf_preds, output_dict=True)
    xgb_rep = classification_report(y_test, xgb_preds, output_dict=True)
    
    print(f"LSTM    Accuracy: {lstm_rep['accuracy']:.4f} | Precision: {lstm_rep['1']['precision']:.4f} | Recall: {lstm_rep['1']['recall']:.4f} | F1: {lstm_rep['1']['f1-score']:.4f}")
    print(f"RF      Accuracy: {rf_rep['accuracy']:.4f} | Precision: {rf_rep['1']['precision']:.4f} | Recall: {rf_rep['1']['recall']:.4f} | F1: {rf_rep['1']['f1-score']:.4f}")
    print(f"XGBoost Accuracy: {xgb_rep['accuracy']:.4f} | Precision: {xgb_rep['1']['precision']:.4f} | Recall: {xgb_rep['1']['recall']:.4f} | F1: {xgb_rep['1']['f1-score']:.4f}")
    
    generate_report_plots(y_test, lstm_preds, lstm_probs, rf_preds, rf_probs, train_losses, val_losses)
    
    # Save all models
    joblib.dump(rf_model, "saved_models/card_rf.pkl")
    joblib.dump(xgb_model, "saved_models/card_xgboost.pkl")
    print("\n[MODELS SAVED] LSTM, Random Forest, and XGBoost saved to saved_models/")

if __name__ == "__main__":
    main()