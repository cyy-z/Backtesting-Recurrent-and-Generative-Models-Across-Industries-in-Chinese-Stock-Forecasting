import numpy as np
import pandas as pd
import os

def save_predictions_and_metrics(
    dates,
    actual_returns,
    predicted_returns,
    strategy_returns,
    stock_name,
    result_dir,
    model_name="model"
):
    """
    Save predicted returns to Excel and backtest metrics to a TXT file.
    """
    os.makedirs(result_dir, exist_ok=True)

    # Save predictions to Excel
    df_pred = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'actual_return': actual_returns,
        'predicted_return': predicted_returns
    })
    excel_path = os.path.join(result_dir, f"{stock_name}_{model_name}_prediction.xlsx")
    df_pred.to_excel(excel_path, index=False)

    # Compute backtest metrics
    strategy_returns = np.array(strategy_returns)
    benchmark_returns = np.array(actual_returns)

    cumulative = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = 1 - cumulative / running_max
    max_drawdown = np.max(drawdown)

    win_rate = np.mean(strategy_returns > 0)
    annualized_return = cumulative[-1] ** (52 / len(strategy_returns)) - 1
    annualized_volatility = np.std(strategy_returns) * np.sqrt(52)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else 0

    # ➕ Alpha: 年化超额收益 = 年化策略收益 - 年化基准收益
    benchmark_cum = np.cumprod(1 + benchmark_returns)
    annualized_benchmark_return = benchmark_cum[-1] ** (52 / len(benchmark_returns)) - 1
    alpha = annualized_return - annualized_benchmark_return

    # Save metrics to TXT file
    txt_path = os.path.join(result_dir, f"{stock_name}_{model_name}_metrics.txt")
    with open(txt_path, 'w', encoding="utf-8") as f:
        f.write(f"Backtest Metrics for {stock_name} using {model_name}\n")
        f.write(f"Win Rate: {win_rate:.2%}\n")
        f.write(f"Max Drawdown: {max_drawdown:.2%}\n")
        f.write(f"Annualized Return: {annualized_return:.2%}\n")
        f.write(f"Annualized Benchmark Return: {annualized_benchmark_return:.2%}\n")
        f.write(f"Alpha (Annualized Excess Return): {alpha:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.4f}\n")
        f.write(f"Calmar Ratio: {calmar_ratio:.4f}\n")

    print(f"✅ Metrics saved to: {txt_path}")
    print(f"✅ Predictions saved to: {excel_path}")
