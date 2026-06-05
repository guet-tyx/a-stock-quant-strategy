import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("模拟盘建仓建议")
print("日期: 2026年6月5日 (周四)")
print("=" * 70)

# 加载数据
data = pd.read_csv('data/latest_data.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

print(f"\n数据概况:")
print(f"  数据范围: {data['trade_date'].min().strftime('%Y-%m-%d')} ~ {data['trade_date'].max().strftime('%Y-%m-%d')}")
print(f"  股票数量: {data['ts_code'].nunique()}")

# 计算技术因子
data['ret_1d'] = data.groupby('ts_code')['close'].pct_change(1)
data['ret_5d'] = data.groupby('ts_code')['close'].pct_change(5)
data['ret_20d'] = data.groupby('ts_code')['close'].pct_change(20)
data['vol_20d'] = data.groupby('ts_code')['ret_1d'].transform(lambda x: x.rolling(20).std())
data['ma5'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(5).mean())
data['ma20'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(20).mean())
data['ma60'] = data.groupby('ts_code')['close'].transform(lambda x: x.rolling(60).mean())
data['vol_ma20'] = data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(20).mean())
data['vol_ma60'] = data.groupby('ts_code')['vol'].transform(lambda x: x.rolling(60).mean())
data['amihud'] = data['ret_1d'].abs() / data['amount'].replace(0, np.nan)
data['amihud_ma20'] = data.groupby('ts_code')['amihud'].transform(lambda x: x.rolling(20).mean())

# 获取最新一天的数据
latest_date = data['trade_date'].max()
latest = data[data['trade_date'] == latest_date].copy()

print(f"  最新交易日: {latest_date.strftime('%Y-%m-%d')}")

# 计算综合得分
latest['momentum'] = (
    0.4 * latest['ret_5d'].rank(pct=True) +
    0.3 * latest['ret_20d'].rank(pct=True) +
    0.3 * (1 - latest['vol_20d'].rank(pct=True))
)
latest['liquidity'] = 1 - latest['amihud_ma20'].rank(pct=True)
latest['trend'] = (
    (latest['close'] > latest['ma20']).astype(float) * 0.4 +
    (latest['close'] > latest['ma60']).astype(float) * 0.3 +
    (latest['ma20'] > latest['ma60']).astype(float) * 0.3
)
latest['combined'] = (
    0.4 * latest['momentum'].fillna(0.5) + 
    0.3 * latest['liquidity'].fillna(0.5) + 
    0.3 * latest['trend'].fillna(0.5)
)

# 排序
latest['rank'] = latest['combined'].rank(ascending=False)
top50 = latest[latest['rank'] <= 50].sort_values('rank')

# 获取股票名称
import tushare as ts
token = "78f5571ac75c61721d05f2ffad3836724590553a613753db16f933fd"
ts.set_token(token)
pro = ts.pro_api()
stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stock_dict = dict(zip(stock_info['ts_code'], stock_info['name']))

print("\n" + "=" * 70)
print("今日建仓建议 - Top 50 股票")
print("=" * 70)

print(f"\n{'排名':<4} {'代码':<12} {'名称':<8} {'收盘价':>8} {'涨跌幅':>7} {'得分':>6}")
print("-" * 55)

for idx, row in top50.iterrows():
    code = row['ts_code']
    name = stock_dict.get(code, 'N/A')
    if len(name) > 4:
        name = name[:4]
    close = row['close']
    pct = row.get('pct_chg', 0)
    if pd.isna(pct):
        pct = 0
    score = row['combined']
    rank = int(row['rank'])
    
    print(f"{rank:<4} {code:<12} {name:<8} {close:>8.2f} {pct:>6.2f}% {score:>6.3f}")

print("\n" + "=" * 70)
print("建仓方案 (以100万资金为例)")
print("=" * 70)

print("""
持仓配置: 50只股票，每只2%权重
总仓位: 80-100%
预留现金: 0-20万

建仓节奏建议:

  方案A: 一次性建仓 (适合50万以下)
    今天全部买入50只股票，每只2万元

  方案B: 分批建仓 (适合50万以上)
    第1天(今天): 买入前25只，投入50万
    第2天(明天): 买入第26-40只，投入30万
    第3天(后天): 买入第41-50只，投入20万

风险控制:
  个股止损: -10% 触发卖出
  组合止损: -15% 全部清仓
  调仓频率: 每月月末调仓一次
  最大持仓: 单只股票不超过5%
""")

print("=" * 70)
print("今日操作清单 - 前20只")
print("=" * 70)

print("\n买入清单 (按优先级排序):")
print("-" * 60)

for idx, row in top50.head(20).iterrows():
    code = row['ts_code']
    name = stock_dict.get(code, 'N/A')
    if len(name) > 6:
        name = name[:6]
    close = row['close']
    rank = int(row['rank'])
    
    # 计算买入股数 (每只2万元)
    buy_amount = 20000
    shares = int(buy_amount / close / 100) * 100  # 向下取整到100股
    actual_amount = shares * close
    
    print(f"{rank:>2}. {code} {name:<8} {close:>8.2f}元  x {shares:>5}股 = {actual_amount:>10,.0f}元")

print("""
注意事项:
  1. 股价超过100元的股票，减少买入数量
  2. 优先买入流动性好的大盘股
  3. 避免在开盘集合竞价时买入，等开盘后观察
  4. 每只股票投入约2万元，总投入约100万元
""")

# 保存建议
top50[['ts_code', 'rank', 'combined', 'close', 'pct_chg']].to_csv('data/today_recommendation.csv', index=False)
print("建议已保存到 data/today_recommendation.csv")
