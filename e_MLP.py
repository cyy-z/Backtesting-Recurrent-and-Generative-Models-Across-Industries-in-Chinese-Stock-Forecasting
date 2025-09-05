import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

class MLPExpert(nn.Module):
    def __init__(self, input_dim):
        super(MLPExpert, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)

def train_mlp(train_preds, train_actuals, test_preds, runs=10,EPOCHS=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds = []

    for _ in range(runs):
        model = MLPExpert(input_dim=train_preds.shape[1]).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        X_train = torch.tensor(train_preds, dtype=torch.float32).to(device)
        y_train = torch.tensor(np.array(train_actuals).reshape(-1, 1), dtype=torch.float32).to(device)

        model.train()
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # 用训练好的模型预测测试集
        model.eval()
        X_test = torch.tensor(test_preds, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(X_test).cpu().numpy().flatten()
            all_preds.append(pred)

        # 🧹 清理内存（每次循环结束后）
        del model
        torch.cuda.empty_cache()
        gc.collect()

        del optimizer, X_train, y_train, X_test, output, loss, pred
        gc.collect()

    return all_preds
