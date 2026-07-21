import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

class CardShieldLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(CardShieldLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze(-1)

def get_random_forest_model():
    return RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)