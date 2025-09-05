from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    df = df.sort_index()
    return df

class FinancialDataset(Dataset):
    def __init__(self, data, lookback, target_col='next_week_return', scaler=None, selected_features=None):
        self.data = data.dropna()
        self.target_col = target_col
        self.lookback = lookback

        if selected_features:
            self.features = [col for col in selected_features if col in self.data.columns]
        else:
            self.features = self.data.columns.drop(target_col)

        self.x_raw = []
        self.y_raw = []
        self.dates = self.data.index[self.lookback:]

        for i in range(len(self.data) - self.lookback):
            self.x_raw.append(self.data[self.features].iloc[i:i + self.lookback].values)
            self.y_raw.append(self.data[self.target_col].iloc[i + self.lookback])

        self.scaler = scaler if scaler else StandardScaler()
        if scaler is None:
            X_train = self.data[self.features].iloc[:-(self.lookback)].values
            stds = X_train.std(axis=0)
            valid_columns = stds != 0
            self.features = [f for f, valid in zip(self.features, valid_columns) if valid]
            X_train = X_train[:, valid_columns]
            self.scaler.fit(X_train)
            # 重新构建 raw
            self.x_raw = [x[:, valid_columns] for x in self.x_raw]

        self.x_scaled = np.array([self.scaler.transform(x) for x in self.x_raw])

    def __len__(self):
        return len(self.x_scaled)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.x_scaled[idx]),
            torch.FloatTensor([self.y_raw[idx]]),
            str(self.dates[idx])
        )

def create_dataloaders(
    data,
    lookback,
    batch_size=32,
    train_start_date=None,
    train_end_date=None,
    test_start_date=None,
    test_end_date=None,
    selected_features=None
):
    # 🧪 校验日期参数
    if not all([train_start_date, train_end_date, test_start_date, test_end_date]):
        raise ValueError("必须指定 train_start_date, train_end_date, test_start_date, test_end_date")

    # 🧠 日期格式标准化
    train_start_date = pd.to_datetime(train_start_date)
    train_end_date = pd.to_datetime(train_end_date)
    test_start_date = pd.to_datetime(test_start_date)
    test_end_date = pd.to_datetime(test_end_date)

    # ✂️ 训练数据
    train_data = data.loc[train_start_date:train_end_date]
    if len(train_data) < lookback:
        raise ValueError("训练数据长度不足以构建窗口，请调整训练区间或lookback长度")

    # ✂️ 测试数据（提前lookback窗口）
    test_start_adjusted = test_start_date - pd.Timedelta(days=lookback)
    test_data = data.loc[test_start_adjusted:test_end_date]
    if len(test_data) < lookback:
        raise ValueError("测试数据长度不足以构建窗口，请调整测试区间或lookback长度")

    # 🧹 构建数据集
    train_dataset = FinancialDataset(train_data, lookback, selected_features=selected_features)
    test_dataset = FinancialDataset(test_data, lookback, scaler=train_dataset.scaler, selected_features=selected_features)

    # 📦 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def extract_ground_truth_from_dataset(dataset):

    actuals = np.array(dataset.y_raw)
    dates = pd.to_datetime(dataset.dates)
    return dates, actuals
