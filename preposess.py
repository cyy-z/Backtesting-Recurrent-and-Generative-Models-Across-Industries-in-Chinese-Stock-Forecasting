import pandas as pd
import numpy as np
import os

def generate_weekly_factors(daily_df):
    """从日频数据生成周度因子，遇除0抛异常跳过"""
    weekly = pd.DataFrame()

    # 基础价格数据
    weekly['close'] = daily_df['close'].resample('W-FRI').last()
    weekly['open'] = daily_df['open'].resample('W-FRI').first()

    # 日频移动平均线
    daily_df['5_day_ma'] = daily_df['close'].rolling(5, min_periods=3).mean().round(4)
    daily_df['20_day_ma'] = daily_df['close'].rolling(20, min_periods=10).mean().round(4)

    # 周频MA
    weekly['5_day_ma'] = daily_df['5_day_ma'].resample('W-FRI').last()
    # 动量因子
    weekly['2_week_mom'] = weekly['close'].pct_change(2, fill_method=None)
    weekly['4_week_mom'] = weekly['close'].pct_change(4, fill_method=None)

    # 精确3日动量，除0抛异常
    def true_3day_mom(g):
        if len(g) < 3:
            return np.nan
        if g.iloc[-3] == 0:
            raise ValueError("除数为0，数据异常，跳过该股票")
        return g.iloc[-1] / g.iloc[-3] - 1

    weekly['true_3day_mom'] = daily_df['close'].groupby(pd.Grouper(freq='W-FRI')).apply(true_3day_mom)

    # 下周收益率（目标）
    weekly['next_week_return'] = weekly['close'].pct_change(fill_method=None).shift(-1)

    return weekly.dropna()


def main():
    input_base = "./assets/original_data"
    output_base = "./assets/data_own_f"
    os.makedirs(output_base, exist_ok=True)

    for industry_name in os.listdir(input_base):
        industry_path = os.path.join(input_base, industry_name)
        if not os.path.isdir(industry_path):
            continue

        output_dir = os.path.join(output_base, industry_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n📂 正在处理行业：{industry_name}")

        for file in os.listdir(industry_path):
            if not file.endswith(".csv"):
                continue

            input_file = os.path.join(industry_path, file)
            output_file = os.path.join(output_dir, file)

            try:
                df = pd.read_csv(input_file, encoding='utf-8')
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').set_index('date')

                df.rename(columns=lambda x: x.strip(), inplace=True)
                if '收盘' in df.columns:
                    df.rename(columns={
                        '收盘': 'close',
                        '开盘': 'open',
                    }, inplace=True)

                # 生成周度因子，遇除零异常跳过该股票
                weekly = generate_weekly_factors(df)

                weekly.to_csv(output_file)
                print(f"✅ 已保存：{output_file}")



            except Exception as e:
                print(f"❌ 处理失败：{file}，错误：{e}")

if __name__ == "__main__":
    main()

