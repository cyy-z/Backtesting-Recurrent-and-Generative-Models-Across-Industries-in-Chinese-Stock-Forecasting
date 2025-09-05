import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from tools import cal,load

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])  # shape: (batch_size, 1)

def train_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')


def evaluate_model(model, test_loader, device):
    model.eval()
    actuals, predictions, dates = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_dates in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).cpu().numpy()

            predictions.extend(outputs.flatten())
            actuals.extend(batch_y.numpy().flatten())
            dates.extend(batch_dates)

    return dates, actuals, predictions

def lstm(runs, train_loader,test_loader, stock_name='sh000300', industry_name='default',EPOCHS = 300):
    HIDDEN_SIZE = 64
    long_threshold = 0.001
    short_threshold = -0.001
    result_dir = f'./result/lstm/{industry_name}/{stock_name}'
    os.makedirs(result_dir, exist_ok=True)


    sample_x, _, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_train_preds = []
    all_train_actuals = []
    all_test_preds = []

    test_dates, test_actuals = load.extract_ground_truth_from_dataset(test_loader.dataset)
    train_dates, train_actuals = load.extract_ground_truth_from_dataset(train_loader.dataset)

    actual_returns = np.array(test_actuals)
    cum_benchmark = np.cumprod(1 + actual_returns)

    plt.figure(figsize=(12, 6))

    for run in range(runs):
        print(f"\n===== Run {run + 1}/{runs} =====")
        model = LSTMModel(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            dropout=0.2
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        print("\n开始训练...")
        train_model(model, train_loader, criterion, optimizer, device, epochs=EPOCHS)

        print("\n开始评估...")
        _, _, test_preds = evaluate_model(model, test_loader, device)
        all_test_preds.append(test_preds)

        # 训练集预测
        model.eval()
        train_preds = []
        with torch.no_grad():
            for batch_x, _, _ in train_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x).cpu().numpy()
                train_preds.extend(outputs.flatten())
        all_train_preds.append(train_preds)
        all_train_actuals.append(train_actuals)

        # 测试集策略回测绘制

        predictions = np.array(test_preds)
        positions = np.zeros_like(predictions)
        positions[predictions > long_threshold] = 1
        positions[predictions < short_threshold] = -1

        strategy_returns = positions * actual_returns
        cum_strategy = np.cumprod(1 + strategy_returns)


        plt.plot(test_dates, cum_strategy, alpha=0.5)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        del outputs, train_preds, test_preds, predictions, positions, strategy_returns, cum_strategy
        gc.collect()

    # 基准线加粗画一次
    plt.plot(test_dates, cum_benchmark, label='Benchmark', linewidth=3, color='black')

    all_test_preds_array = np.array(all_test_preds)  # shape: [10, T]

    # 分位数计算
    q40 = np.percentile(all_test_preds_array, 40, axis=0)
    q60 = np.percentile(all_test_preds_array, 60, axis=0)

    # 信号生成
    quantile_positions = np.zeros_like(q40)
    quantile_positions[q40 > long_threshold] = 1
    quantile_positions[q60 < short_threshold] = -1

    # 策略收益与累计收益
    quantile_strategy_returns = quantile_positions * np.array(test_actuals)
    quantile_cum_strategy = np.cumprod(1 + quantile_strategy_returns)

    # 画图
    plt.plot(test_dates, quantile_cum_strategy, label='Strategy', linewidth=3, color='red')

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = f"{result_dir}/{stock_name}_multi_run_backtest.png"
    plt.savefig(save_path)
    plt.close()

    mean_train_actuals = train_actuals  # 训练集实际值都一样，取任意一次即可

    cal.save_predictions_and_metrics(
        test_dates,
        actual_returns,
        np.mean(all_test_preds_array, axis=0),
        quantile_strategy_returns,
        stock_name,
        result_dir,
        model_name="LSTM"
    )

    return (mean_train_actuals,
        np.mean(np.array(all_train_preds), axis=0), np.mean(all_test_preds_array, axis=0),
       test_dates,
       test_actuals)

def run_lstm_on_all_stocks():
    root_dir = './assets/data_own_f'
    result_dir = './result/lstm'
    os.makedirs(result_dir, exist_ok=True)

    for industry in os.listdir(root_dir):
        industry_path = os.path.join(root_dir, industry)
        if not os.path.isdir(industry_path):
            continue

        print(f"\n🏭 正在处理行业：{industry}")

        for stock_file in os.listdir(industry_path):
            if not stock_file.endswith(".csv"):
                continue

            stock_name = stock_file.replace(".csv", "")
            file_path = os.path.join(industry_path, stock_file)

            print(f"\n📈 正在训练股票：{stock_name}")

            try:
                lstm(
                    runs=10,
                    data_path=file_path,
                    stock_name=stock_name,
                    industry_name=industry
                )
            except Exception as e:
                print(f"❌ 股票 {stock_name} 处理失败：{e}")


if __name__ == "__main__":
    run_lstm_on_all_stocks()