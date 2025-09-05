import torch
import gc
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tools import cal,load

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size + 1, 1)

    def forward(self, x, y):
        _, (hn, _) = self.lstm(x)
        combined = torch.cat([hn[-1], y], dim=1)
        return self.fc(combined)  # No sigmoid in WGAN


# WGAN training with weight clipping
def train_wgan(generator, discriminator, train_loader, device, epochs=100, clip_value=0.01, n_critic=5):
    g_optimizer = torch.optim.RMSprop(generator.parameters(), lr=0.00005)
    d_optimizer = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        d_steps = 0
        g_loss_value = 0
        for i, (batch_x, batch_y, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # ========== Train Discriminator ==========
            for _ in range(n_critic):
                fake_y = generator(batch_x).detach()
                d_real = discriminator(batch_x, batch_y)
                d_fake = discriminator(batch_x, fake_y)
                d_loss = -(torch.mean(d_real) - torch.mean(d_fake))

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                for p in discriminator.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # ========== Train Generator ==========
            fake_y = generator(batch_x)
            g_loss = -torch.mean(discriminator(batch_x, fake_y))

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_steps += 1
            g_loss_value = g_loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f} | G_loss: {g_loss_value:.4f}")


def evaluate_model(generator, test_loader, device):
    generator.eval()
    actuals, predictions, dates = [], [], []

    with torch.no_grad():
        for batch_x, batch_y, batch_dates in test_loader:
            batch_x = batch_x.to(device)
            preds = generator(batch_x).cpu().numpy()

            predictions.extend(preds.flatten())
            actuals.extend(batch_y.numpy().flatten())
            dates.extend(batch_dates)

    dates = pd.to_datetime(dates)

    return dates, actuals, predictions

def wgan(runs,train_loader,test_loader,stock_name='sh000300', industry_name='default',EPOCHS = 300):
    HIDDEN_SIZE = 64
    long_threshold = 0.001
    short_threshold = -0.001
    # 创建行业目录
    result_dir = f'./result/wgan/{industry_name}/{stock_name}'
    os.makedirs(result_dir, exist_ok=True)

    sample_x, _, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_train_preds = []
    all_test_preds = []

    # 获取测试集真实值和日期（只取一次）
    test_dates, test_actuals = load.extract_ground_truth_from_dataset(test_loader.dataset)

    actual_returns = np.array(test_actuals)
    cum_benchmark = np.cumprod(1 + actual_returns)

    plt.figure(figsize=(12, 6))

    for run in range(runs):
        print(f"\n===== Run {run + 1}/{runs} =====")
        generator = Generator(input_size, hidden_size=HIDDEN_SIZE).to(device)
        discriminator = Discriminator(input_size, hidden_size=HIDDEN_SIZE).to(device)

        print("\n开始训练 WGAN...")
        train_wgan(generator, discriminator, train_loader, device, epochs=EPOCHS)

        print("\n开始评估...")
        _, _, test_preds = evaluate_model(generator, test_loader, device)
        all_test_preds.append(test_preds)

        # 训练集预测
        generator.eval()
        train_preds = []
        with torch.no_grad():
            for batch_x, _, _ in train_loader:
                batch_x = batch_x.to(device)
                outputs = generator(batch_x).cpu().numpy()
                train_preds.extend(outputs.flatten())
        all_train_preds.append(train_preds)

        # 回测测试集预测
        predictions = np.array(test_preds)
        positions = np.zeros_like(predictions)
        positions[predictions > long_threshold] = 1
        positions[predictions < short_threshold] = -1

        strategy_returns = positions * actual_returns
        cum_strategy = np.cumprod(1 + strategy_returns)

        plt.plot(test_dates, cum_strategy, alpha=0.5)

        del generator, discriminator
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
    plt.plot(test_dates, quantile_cum_strategy, label='Strategy',linewidth=3, color='red')

    plt.xlabel("Date")
    plt.ylabel("Cumulative Ret")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = f"{result_dir}/{stock_name}_multi_run_backtest.png"
    plt.savefig(save_path)
    plt.close()

    cal.save_predictions_and_metrics(
        test_dates,
        actual_returns,
        np.mean(all_test_preds_array, axis=0),
        quantile_strategy_returns,
        stock_name,
        result_dir,
        model_name="WGAN"
    )

    return np.mean(np.array(all_train_preds), axis=0), np.mean(all_test_preds_array, axis=0)


def run_wgan_on_all_stocks():
    root_dir = './assets/data_own_f'
    result_dir = './result/wgan'
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
                wgan(
                    runs=10,
                    data_path=file_path,
                    stock_name=stock_name,
                    industry_name=industry
                )
            except Exception as e:
                print(f"❌ 股票 {stock_name} 处理失败：{e}")


if __name__ == "__main__":
    run_wgan_on_all_stocks()
