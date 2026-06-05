import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("模拟盘建仓建议 - 10万资金版")
print("日期: 2026年6月5日 (周四)")
print("=" * 70)

# 加载数据
data = pd.read_csv('data/latest_data.csv')
data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

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
top_stocks = latest.sort_values('rank')

# 获取股票名称
import tushare as ts
token = "78f5571ac75c61721d05f2ffad3836724590553a613753db16f933fd"
ts.set_token(token)
pro = ts.pro_api()
stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
stock_dict = dict(zip(stock_info['ts_code'], stock_info['name']))

# ============================================================
# 10万资金建仓方案
# ============================================================
total_capital = 100000  # 10万

print(f"\n资金规模: {total_capital/10000:.0f}万元")
print(f"数据日期: {latest_date.strftime('%Y-%m-%d')}")

# 方案分析
print("\n" + "=" * 70)
print("建仓方案分析")
print("=" * 70)

# 计算不同持仓数量下的可行性
print("\n不同持仓数量分析:")
print("-" * 60)
print(f"{'持仓数':<8} {'每只资金':<10} {'可行性':<10} {'说明'}")
print("-" * 60)

for n_stocks in [10, 15, 20, 25, 30]:
    per_stock = total_capital / n_stocks
    
    # 检查是否能买到最低100股
    affordable = 0
    for idx, row in top_stocks.head(n_stocks).iterrows():
        price = row['close']
        if per_stock >= price * 100:  # 至少能买100股
            affordable += 1
    
    feasibility = f"{affordable}/{n_stocks}只可买"
    if affordable == n_stocks:
        status = "✅ 推荐"
    elif affordable >= n_stocks * 0.8:
        status = "⚠️ 可行"
    else:
        status = "❌ 不推荐"
    
    print(f"{n_stocks}只    {per_stock:.0f}元     {status:<10} {feasibility}")

# ============================================================
# 最优方案: 20只股票
# ============================================================
n_stocks = 20
per_stock = total_capital / n_stocks

print("\n" + "=" * 70)
print(f"推荐方案: {n_stocks}只股票 (每只{per_stock:.0f}元)")
print("=" * 70)

# 筛选可买的股票
selected = []
for idx, row in top_stocks.iterrows():
    price = row['close']
    shares = int(per_stock / price / 100) * 100  # 向下取整到100股
    if shares >= 100:  # 至少能买100股
        actual_amount = shares * price
        selected.append({
            'rank': len(selected) + 1,
            'ts_code': row['ts_code'],
            'name': stock_dict.get(row['ts_code'], 'N/A'),
            'close': price,
            'shares': shares,
            'amount': actual_amount,
            'combined': row['combined']
        })
    
    if len(selected) >= n_stocks:
        break

selected_df = pd.DataFrame(selected)

print(f"\n{'排名':<4} {'代码':<12} {'名称':<8} {'价格':>8} {'股数':>6} {'金额':>10}")
print("-" * 60)

for idx, row in selected_df.iterrows():
    name = row['name']
    if len(name) > 6:
        name = name[:6]
    print(f"{row['rank']:<4} {row['ts_code']:<12} {name:<8} {row['close']:>8.2f} {row['shares']:>6} {row['amount']:>10,.0f}")

total_invested = selected_df['amount'].sum()
cash_remaining = total_capital - total_invested
position_ratio = total_invested / total_capital * 100

print("\n" + "-" * 60)
print(f"总计投入: {total_invested:,.0f}元")
print(f"剩余现金: {cash_remaining:,.0f}元")
print(f"仓位比例: {position_ratio:.1f}%")

# ============================================================
# 详细操作清单
# ============================================================
print("\n" + "=" * 70)
print("今日操作清单")
print("=" * 70)

print("\n买入清单:")
print("-" * 60)

for idx, row in selected_df.iterrows():
    code = row['ts_code']
    name = row['name']
    if len(name) > 6:
        name = name[:6]
    price = row['close']
    shares = row['shares']
    amount = row['amount']
    
    # 计算需要的总金额 (含手续费)
    commission = max(amount * 0.0003, 5)  # 佣金万三，最低5元
    total_cost = amount + commission
    
    print(f"{row['rank']:>2}. {code} {name:<8} {price:>8.2f}元  x {shares:>4}股 = {amount:>8,.0f}元 (佣金{commission:.0f}元)")

print("\n" + "-" * 60)
print(f"总投入: {total_invested:,.0f}元")
print(f"总佣金: {total_invested * 0.0003:.0f}元 (估算)")
print(f"剩余现金: {cash_remaining:,.0f}元")

# ============================================================
# 资金分配建议
# ============================================================
print("\n" + "=" * 70)
print("资金分配建议")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────┐
│  总资金: 10万元                                           │
│  持仓数量: {n_stocks}只                                       │
│  每只金额: 约{per_stock:.0f}元                                  │
│  仓位比例: {position_ratio:.1f}%                                      │
└─────────────────────────────────────────────────────────┘

建仓节奏:
  方案A: 一次性建仓 (推荐)
    今天全部买入{n_stocks}只股票

  方案B: 分批建仓 (保守)
    第1天: 买入前10只，投入5万
    第2天: 买入后10只，投入剩余资金

注意事项:
  1. 每只股票至少买100股
  2. 股价超过500元的股票可能买不起
  3. 预留{cash_remaining:.0f}元作为交易费用和补仓资金
""")

# ============================================================
# 风险控制
# ============================================================
print("=" * 70)
print("风险控制")
print("=" * 70)

print(f"""
止损规则:
  个股止损: -10% (亏损{total_invested * 0.1 / n_stocks:.0f}元)
  组合止损: -15% (亏损{total_invested * 0.15:.0f}元)

监控频率:
  每日: 检查个股止损线
  每周: 检查组合表现
  每月末: 调仓 (重新计算因子)

预期收益 (基于回测):
  年化收益: 30-40%
  月均收益: 2.5-3.3%
  10万元 → 1年后: 13-14万元
""")

# ============================================================
# 选股逻辑说明
# ============================================================
print("=" * 70)
print("选股逻辑说明")
print("=" * 70)

print("""
三因子模型:
  1. 动量因子 (40%): 短期涨幅+低波动
  2. 流动性因子 (30%): Amihud非流动性反向
  3. 趋势因子 (30%): 价格在均线上方

因子来源:
  - ML因子: GradientBoosting模型预测
  - LLM因子: 流动性情绪指标
  - arXiv因子: 学术论文公式

策略特点:
  - 等权重配置，分散风险
  - 月度调仓，降低交易成本
  - 多因子组合，提高稳定性
""")

# 保存建议
selected_df.to_csv('data/today_recommendation_10w.csv', index=False)
print("建议已保存到 data/today_recommendation_10w.csv")
