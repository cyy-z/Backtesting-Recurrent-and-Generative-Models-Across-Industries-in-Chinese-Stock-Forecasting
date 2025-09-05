import WGAN,LSTM_price_only,GRU,WGAN_GRU,e_MLP,getdata,preposess,summary
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tools import cal,load
import gc

long_threshold = 0.001
short_threshold = -0.001
root_dir = './assets/data_own_f'

target_industries =['300']
#target_industries=['小金属']
#target1=['000938_紫光股份']
LOOKBACK = 4
BATCH_SIZE = 32
epoch=200
for industry_get in target_industries:
    print(industry_get)
    #getdata.get_data([industry_get])
    #preposess.main()
    for industry in os.listdir(root_dir):
        if industry not in industry_get:
            continue

        industry_path = os.path.join(root_dir, industry)

        if not os.path.isdir(industry_path):
            continue

        print(f"\n🏭 正在处理行业：{industry}")

        for stock_file in os.listdir(industry_path):
            if not stock_file.endswith(".csv"):
                continue

            stock_name = stock_file.replace(".csv", "")

            file_path = os.path.join(industry_path, stock_file)
            result_dir = f'./result/mlp/{industry}/{stock_name}'
            os.makedirs(result_dir, exist_ok=True)

            print(f"\n📈 正在训练股票：{stock_name}")

            try:

                data = load.load_data(file_path)
                selected_features = ['close', 'open', '5_day_ma', '2_week_mom', '4_week_mom', 'true_3day_mom']

                # 构造训练与测试数据
                train_loader, test_loader = load.create_dataloaders(
                    data,
                    lookback=LOOKBACK,
                    batch_size=BATCH_SIZE,
                    train_start_date='2018-01-01',
                    train_end_date='2022-06-30',
                    test_start_date='2022-07-01',
                    test_end_date='2024-07-01',
                    selected_features=selected_features
                )

                train_actuals, lstm_train_pre, lstm_pre, dates, actuals = LSTM_price_only.lstm(runs=10,
                                                                                               train_loader=train_loader,
                                                                                               test_loader=test_loader,
                                                                                               stock_name=stock_name,
                                                                                               industry_name=industry,
                                                                                               EPOCHS=epoch)
                gru_train_pre, gru_pre = GRU.gru(runs=10, train_loader=train_loader,
                                                 test_loader=test_loader,
                                                 stock_name=stock_name,
                                                 industry_name=industry, EPOCHS=epoch)
                wgan_train_pre, wgan_pre = WGAN.wgan(runs=10, train_loader=train_loader,
                                                     test_loader=test_loader,
                                                     stock_name=stock_name,
                                                     industry_name=industry, EPOCHS=epoch)
                wgan_gru_train_pre, wgan_gru_pre = WGAN_GRU.wgan_gru(runs=10, train_loader=train_loader,
                                                                     test_loader=test_loader,
                                                                     stock_name=stock_name,
                                                                     industry_name=industry, EPOCHS=epoch)

                train_preds = np.stack([
                    lstm_train_pre,
                    gru_train_pre,
                    wgan_train_pre,
                    wgan_gru_train_pre
                ], axis=1)

                test_preds = np.stack([
                    lstm_pre,
                    gru_pre,
                    wgan_pre,
                    wgan_gru_pre
                ], axis=1)

                train_actuals = np.array(train_actuals)

                # 使用专家 MLP 模型训练并预测
                all_preds = e_MLP.train_mlp(train_preds, train_actuals, test_preds, EPOCHS=epoch)

                plt.figure(figsize=(12, 6))
                save_path = f"{result_dir}/{stock_name}_multi_run_backtest.png"

                actual_returns = np.array(actuals)
                cum_benchmark = np.cumprod(1 + actual_returns)
                plt.plot(dates, cum_benchmark, label='Benchmark', linewidth=3, color='black')

                for i, pred in enumerate(all_preds):
                    positions = np.zeros_like(pred)
                    positions[pred > long_threshold] = 1
                    positions[pred < short_threshold] = -1
                    strategy_returns = positions * np.array(actuals)
                    cum_strategy = np.cumprod(1 + strategy_returns)
                    plt.plot(dates, cum_strategy, alpha=0.5)

                all_test_preds_array = np.array(all_preds)  # shape: [10, T]

                # 分位数计算
                q40 = np.percentile(all_test_preds_array, 40, axis=0)
                q60 = np.percentile(all_test_preds_array, 60, axis=0)

                # 信号生成
                quantile_positions = np.zeros_like(q40)
                quantile_positions[q40 > long_threshold] = 1
                quantile_positions[q60 < short_threshold] = -1

                # 策略收益与累计收益
                quantile_strategy_returns = quantile_positions * np.array(actuals)
                quantile_cum_strategy = np.cumprod(1 + quantile_strategy_returns)

                # 画图
                plt.plot(dates, quantile_cum_strategy, label='Strategy', linewidth=3, color='red')

                plt.xlabel("Date")
                plt.ylabel("Cumulative Return")
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()

                cal.save_predictions_and_metrics(
                    dates,
                    actual_returns,
                    np.mean(all_test_preds_array, axis=0),
                    quantile_strategy_returns,
                    stock_name,
                    result_dir,
                    model_name="EXPERTISE"
                )

            except Exception as e:
                print(f"❌ 股票 {stock_name} 处理失败：{e}")

            finally:
                # ✅ 无论是否报错，都进行内存清理
                for var in [
                    'lstm_train_pre', 'lstm_pre',
                    'gru_train_pre', 'gru_pre',
                    'wgan_train_pre', 'wgan_pre',
                    'wgan_gru_train_pre', 'wgan_gru_pre',
                    'train_preds', 'test_preds',
                    'train_actuals', 'all_preds', 'actuals',
                    'avg_pred', 'mean_strategy_returns', 'mean_cum_strategy'
                ]:
                    if var in locals():
                        del locals()[var]

                plt.clf()
                plt.cla()
                plt.close('all')

                gc.collect()
                torch.cuda.empty_cache()


summary.main()