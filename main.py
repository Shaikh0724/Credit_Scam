import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd

from src.preprocessing import load_and_preprocess_card_data
from src.models import CardShieldLSTM, get_random_forest_model
from src.evaluate import generate_report_plots

def main():
    data_path = "data/creditcard.csv"
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    
    # Preprocessing ke matching variables load ho rahe hain
    X_train_seq, X_test_seq, X_train_res, X_test, y_train_res, y_test = load_and_preprocess_card_data(data_path)
    val_X_seq, val_y = X_test_seq[:5000], y_test[:5000]
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_res)), batch_size=1024, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(val_X_seq), torch.FloatTensor(val_y)), batch_size=1024, shuffle=False)
    
    print("[LSTM ARCHITECTURE SPECIFICATIONS]")
    print("Sequential multi-feature tracking input space topology mapping.")
    print("Recurrent network architecture initialized with 32 units storage space gates.")
    print("Sigmoid normalization mapping layer applied to final state sequence vectors.\n")
    
    model_lstm = CardShieldLSTM(input_dim=X_train_seq.shape[2], hidden_dim=32)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_lstm.parameters(), lr=0.005)
    
    print("[TRAINING ITERATION PROGRESS LOGS]")
    train_losses, val_losses = [], []
    for epoch in range(5):
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
                
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))
        print(f"Epoch {epoch+1}/5 | Training Loss: {train_losses[-1]:.4f} | Validation Loss: {val_losses[-1]:.4f}")
        
    model_lstm.eval()
    with torch.no_grad():
        lstm_probs = model_lstm(torch.FloatTensor(X_test_seq)).numpy()
    lstm_preds = (lstm_probs >= 0.5).astype(int)
    
    print("\n[MACHINE LEARNING SELECTION METRICS]")
    print("Baseline Selection: Random Forest Classifier")
    print("Ensemble parameters resolve non-linear cross-feature boundaries without outlier distortion.")
    print("Bootstrap aggregation trees naturally mitigate variance under severe minority anomalies.\n")
    
    rf_model = get_random_forest_model()
    rf_model.fit(X_train_res, y_train_res)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    
    print("[PERFORMANCE EVALUATION METRICS REPORT]")
    lstm_rep = classification_report(y_test, lstm_preds, output_dict=True)
    rf_rep = classification_report(y_test, rf_preds, output_dict=True)
    
    print(f"LSTM Accuracy: {lstm_rep['accuracy']:.4f} | Precision: {lstm_rep['1']['precision']:.4f} | Recall: {lstm_rep['1']['recall']:.4f} | F1: {lstm_rep['1']['f1-score']:.4f}")
    print(f"RF Accuracy:   {rf_rep['accuracy']:.4f} | Precision: {rf_rep['1']['precision']:.4f} | Recall: {rf_rep['1']['recall']:.4f} | F1: {rf_rep['1']['f1-score']:.4f}")
    
    generate_report_plots(y_test, lstm_preds, lstm_probs, rf_preds, rf_probs, train_losses, val_losses)    
    torch.save(model_lstm.state_dict(), "saved_models/card_lstm_model.pth")
    import joblib
    joblib.dump(rf_model, "saved_models/card_rf.pkl")

if __name__ == "__main__":
    main()