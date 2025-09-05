import akshare as ak
import pandas as pd
import os
import time
import random
def get_data(target_industries):
    # 设置全局时间参数
    start_date = "20170701"
    end_date = "20250701"
    cutoff_date = pd.to_datetime("2017-07-15")

    # 设置要爬取的行业列表

    # 获取所有行业及对应代码
    industry_info_df = ak.stock_board_industry_name_em()

    # 根目录
    base_output_dir = "./assets/original_data"
    os.makedirs(base_output_dir, exist_ok=True)

    # 遍历行业
    for industry_name in target_industries:
        try:
            # 获取行业代码
            row = industry_info_df[industry_info_df["板块名称"] == industry_name]
            if row.empty:
                print(f"❌ 未找到行业: {industry_name}")
                continue
            industry_code = row["板块代码"].values[0]
            print(f"\n🔍 正在处理行业: {industry_name} ({industry_code})")

            # 获取行业成分股
            stock_list = ak.stock_board_industry_cons_em(symbol=industry_code)
            output_dir = os.path.join(base_output_dir, industry_name)
            os.makedirs(output_dir, exist_ok=True)

            filtered_count = 0  # 用于统计筛掉的股票数量

            # 遍历每只股票
            for code, name in zip(stock_list["代码"], stock_list["名称"]):
                # 🧹 筛掉 ST 和 *ST 股票
                if "ST" in name.upper():
                    print(f"  🚫 跳过 ST 股票: {code} {name}")
                    continue

                print(f"  📈 尝试下载: {code} {name}")
                try:
                    stock_data = ak.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                    if stock_data.empty or "日期" not in stock_data.columns:
                        print(f"    ⚠️ 无数据，跳过 {code}")
                        continue

                    # 转换时间格式
                    stock_data['date'] = pd.to_datetime(stock_data['日期'])
                    first_date = stock_data['date'].min()

                    if first_date > cutoff_date:
                        print(f"    ⏩ 起始时间 {first_date.date()} 晚于2017，跳过 {code}")
                        continue

                    # 🚨 新增：筛选
                    low_volume_days1 = (stock_data['成交量'] < 25000).sum()
                    low_volume_days2 = (stock_data['成交额'] < 50000000).sum()
                    total_days = len(stock_data)
                    if total_days > 0 and (low_volume_days1 / total_days) > 0.05:
                        print(f"    ❌ 成交量过低：{low_volume_days1}/{total_days}，跳过 {code}")
                        filtered_count += 1
                        continue

                    if total_days > 0 and (low_volume_days2 / total_days) > 0.05:
                        print(f"    ❌ 成交额过低：{low_volume_days2}/{total_days}，跳过 {code}")
                        filtered_count += 1
                        continue

                    # 设置索引并保存
                    stock_data.set_index('date', inplace=True)
                    file_path = os.path.join(output_dir, f"{code}_{name}.csv")
                    stock_data.to_csv(file_path)
                    print(f"    ✅ 已保存到: {file_path}")

                except Exception as e:
                    print(f"    ❌ 下载失败: {code} {name}，原因：{e}")

                # 防止限流：随机延迟
                time.sleep(random.uniform(6, 5))

            # 打印该行业被筛掉的个数
            print(f"📉 行业【{industry_name}】筛掉成交量低的股票数：{filtered_count}")

        except Exception as e:
            print(f"❌ 行业处理失败: {industry_name}，原因: {e}")

if __name__ == "__main__":
    target_industries = ['医疗服务', '互联网服务', '工程机械', '半导体','非金属材料', '光伏设备', '电源设备', '能源金属', '钢铁行业','生物制品', '医疗服务', '互联网服务', '汽车整车', '化学制药', '工程咨询服务', '电子化学品', '工程机械', '航天航空',
    '多元金融', '电子元件', '半导体', '消费电子', '医疗器械', '房地产服务', '汽车零部件', '通信设备', '教育', '仪器仪表', '文化传媒',
    '专业服务', '物流行业', '汽车服务','光学光电子', '橡胶制品', '装修装饰', '玻璃玻纤', '水泥建材', '通信服务', '专用设备', '交运设备',
    '综合行业', '铁路公路', '游戏', '电网设备', '塑料制品', '环保行业', '非金属材料', '小金属', '通用设备', '光伏设备', '贸易行业',
    '电源设备', '房地产开发', '风电设备', '中药', '电力行业', '航运港口', '化学制品', '电池', '工程建设', '包装材料', '电机', '船舶制造',
    '造纸印刷', '旅游酒店', '家电行业', '医药商业', '能源金属', '装修建材', '有色金属', '纺织服装', '商业百货', '公用事业', '煤炭行业','农牧饲渔', '化学原料', '采掘行业', '钢铁行业', '燃气', '保险', '化肥行业', '农药兽药', '石油行业', '家用轻工', '酿酒行业', '航空机场',
    '化纤行业', '食品饮料', '美容护理', '贵金属', '珠宝首饰']
    get_data(target_industries)