import os
import re
import pandas as pd

def analyze_alpha(df):
    # 确保列名统一
    df.columns = [col.strip().lower() for col in df.columns]

    # 按 model 和 industry 分组
    grouped = df.groupby(['model', 'industry'])

    result = []

    for (model, industry), group in grouped:
        total = len(group)
        positive_alpha = group[group['alpha_annualized_excess_return'] > 0].shape[0]
        ratio = round(positive_alpha / total, 4) if total > 0 else 0.0
        result.append({
            'model': model,
            'industry': industry,
            'total': total,
            'alpha>0_count': positive_alpha,
            'alpha>0_ratio': ratio
        })

    result_df = pd.DataFrame(result)
    result_df.to_excel('./count.xlsx', index=False)


def extract_metrics_from_txt(txt_path):
    """
    从txt文件中提取所有指标，返回一个字典。
    """
    metrics = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # 统一提取格式："Win Rate: 58.33%" => ("win_rate", 0.5833)
        match = re.match(r"([\w\s\(\)-]+):\s*([0-9\.\-]+)%?", line)
        if match:
            name_raw, value = match.groups()
            name = name_raw.lower().strip().replace(' ', '_').replace('(', '').replace(')', '')
            try:
                metrics[name] = float(value.strip('%')) / 100 if '%' in line else float(value)
            except:
                metrics[name] = None
    return metrics

def collect_all_metrics(result_root='./result', save_path='all_model_metrics.xlsx'):
    """
    遍历所有模型的回测结果目录，收集各模型的回测指标并保存到Excel。
    """
    all_data = []

    for model in os.listdir(result_root):
        model_path = os.path.join(result_root, model)
        if not os.path.isdir(model_path):
            continue

        for industry in os.listdir(model_path):
            industry_path = os.path.join(model_path, industry)
            if not os.path.isdir(industry_path):
                continue

            for stock in os.listdir(industry_path):
                stock_path = os.path.join(industry_path, stock)
                if not os.path.isdir(stock_path):
                    continue

                for file in os.listdir(stock_path):
                    if file.endswith("_metrics.txt"):
                        txt_path = os.path.join(stock_path, file)
                        metrics = extract_metrics_from_txt(txt_path)
                        metrics.update({
                            "model": model,
                            "industry": industry,
                            "stock": stock
                        })
                        all_data.append(metrics)

    df = pd.DataFrame(all_data)
    df = df[['model', 'industry', 'stock'] + [col for col in df.columns if col not in ['model', 'industry', 'stock']]]
    df.to_excel(save_path, index=False)
    print(f"✅ 所有指标已保存至: {save_path}")
    return df



def main():
    df = collect_all_metrics(result_root='./result', save_path='./all_model_metrics.xlsx')
    analyze_alpha(df)

if __name__ == "__main__":
    main()

