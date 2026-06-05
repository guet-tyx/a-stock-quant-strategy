import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("arXiv论文因子挖掘与测试")
print("=" * 60)

# 加载数据
data_path = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/daily_data_fixed.csv")
df = pd.read_csv(data_path, dtype={"trade_date": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

print(f"数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"股票数量: {df['ts_code'].nunique()}")

# 基础特征
df["ret_1d"] = df.groupby("ts_code")["close"].pct_change(1)
df["ret_5d"] = df.groupby("ts_code")["close"].pct_change(5)
df["ret_10d"] = df.groupby("ts_code")["close"].pct_change(10)
df["ret_20d"] = df.groupby("ts_code")["close"].pct_change(20)
df["ret_60d"] = df.groupby("ts_code")["close"].pct_change(60)
df["vol"] = df["vol"].replace(0, np.nan)
df["amount"] = df["amount"].replace(0, np.nan)

# ============================================================
# 基于arXiv论文的新因子
# ============================================================
print("\n" + "=" * 60)
print("构建arXiv论文因子")
print("=" * 60)

factors = pd.DataFrame()
factors["ts_code"] = df["ts_code"]
factors["trade_date"] = df["trade_date"]

# ============================================================
# 1. 趋势质量因子 (Trend Quality) - 来自QuantaAlpha论文
# 趋势延续只有在低残差波动率和改善流动性时才有效
# ============================================================
df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
df["ma60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())
df["residual"] = df["close"] / df["ma20"] - 1  # 相对MA20的残差
df["residual_vol"] = df.groupby("ts_code")["residual"].transform(lambda x: x.rolling(20).std())
df["vol_ma20"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
df["vol_ma60"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(60).mean())
df["vol_trend"] = df["vol_ma20"] / df["vol_ma60"].replace(0, np.nan) - 1  # 成交量趋势

# 趋势质量 = 动量 * (1 - 残差波动率排名) * 成交量改善
df["momentum_20d_rank"] = df.groupby("trade_date")["ret_20d"].rank(pct=True)
df["residual_vol_rank"] = df.groupby("trade_date")["residual_vol"].rank(pct=True)
df["vol_trend_rank"] = df.groupby("trade_date")["vol_trend"].rank(pct=True)

factors["trend_quality"] = (
    df["momentum_20d_rank"] * 
    (1 - df["residual_vol_rank"]) * 
    df["vol_trend_rank"]
)

# ============================================================
# 2. 量能不稳定性指数 (Exhaustion Volume Instability Index)
# 趋势偏差结合量能不稳定性，识别脆弱的价格水平
# ============================================================
df["price_dev"] = (df["close"] - df["ma60"]) / df["ma60"].replace(0, np.nan)
df["vol_stability"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).std()) / df["vol_ma20"].replace(0, np.nan)

factors["exhaustion_vol_instability"] = df["price_dev"].abs() * df["vol_stability"]

# ============================================================
# 3. 相对成交量平静反转 (Relative Volume Calm Reversal)
# "平静量能"状态乘以动量发散
# ============================================================
df["vol_calm"] = 1 - df.groupby("trade_date")["vol"].rank(pct=True)  # 低成交量=平静
df["mom_divergence"] = df["ret_5d"] - df["ret_20d"]  # 短期vs长期动量发散

factors["vol_calm_reversal"] = df["vol_calm"] * df["mom_divergence"]

# ============================================================
# 4. 量能稳定性动量发散 (Volume Stability Momentum Divergence)
# 稳健的量能稳定性代理(MAD)乘以动量价差
# ============================================================
df["vol_mad"] = df.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(20).apply(lambda y: np.median(np.abs(y - np.median(y))), raw=True)
) / df["vol_ma20"].replace(0, np.nan)
df["mom_spread"] = df["ret_5d"] - df["ret_60d"]

factors["vol_stability_mom_divergence"] = -df["vol_mad"] * df["mom_spread"]

# ============================================================
# 5. RSI反弹强度 (RSI Bounce Strength) - 来自行为驱动多因子论文
# RSI超卖反弹信号
# ============================================================
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = df.groupby("ts_code")["close"].transform(lambda x: calc_rsi(x, 14))
df["rsi_prev"] = df.groupby("ts_code")["rsi_14"].shift(1)

# RSI反弹: 从超卖区(30以下)反弹
factors["rsi_bounce_strength"] = np.where(
    (df["rsi_prev"] < 30) & (df["rsi_14"] > df["rsi_prev"]),
    df["rsi_14"] - df["rsi_prev"],
    0
)

# ============================================================
# 6. MACD交叉强度 (MACD Cross Strength)
# MACD金叉信号强度
# ============================================================
df["ema12"] = df.groupby("ts_code")["close"].transform(lambda x: x.ewm(span=12).mean())
df["ema26"] = df.groupby("ts_code")["close"].transform(lambda x: x.ewm(span=26).mean())
df["macd_line"] = df["ema12"] - df["ema26"]
df["macd_signal"] = df.groupby("ts_code")["macd_line"].transform(lambda x: x.ewm(span=9).mean())
df["macd_hist"] = df["macd_line"] - df["macd_signal"]
df["macd_hist_prev"] = df.groupby("ts_code")["macd_hist"].shift(1)

# MACD金叉强度
factors["macd_cross_strength"] = np.where(
    (df["macd_hist_prev"] < 0) & (df["macd_hist"] > 0),
    df["macd_hist"].abs(),
    0
)

# ============================================================
# 7. 波动率调整动量 (Volatility-Adjusted Momentum)
# 动量除以波动率，来自多篇论文
# ============================================================
df["vol_20d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(20).std())

factors["vol_adj_momentum"] = df["ret_20d"] / df["vol_20d"].replace(0, np.nan)

# ============================================================
# 8. 流动性调整动量 (Liquidity-Adjusted Momentum)
# 动量考虑流动性改善
# ============================================================
df["amihud"] = df["ret_1d"].abs() / df["amount"].replace(0, np.nan)
df["amihud_ma20"] = df.groupby("ts_code")["amihud"].transform(lambda x: x.rolling(20).mean())
df["liquidity_improve"] = -df.groupby("trade_date")["amihud_ma20"].rank(pct=True)  # 低Amihud=高流动性

factors["liquidity_adj_momentum"] = df["momentum_20d_rank"] * df["liquidity_improve"]

# ============================================================
# 9. 底部反转因子 (Bottom Reversal) - 来自行为驱动多因子论文
# 超卖恢复、价格反弹、温和放量
# ============================================================
df["min_20d"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).min())
df["max_20d"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).max())
df["close_position"] = (df["close"] - df["min_20d"]) / (df["max_20d"] - df["min_20d"]).replace(0, np.nan)

# 底部反转: 价格在低位 + RSI超卖 + 成交量温和放大
factors["bottom_reversal"] = np.where(
    (df["close_position"] < 0.3) & (df["rsi_14"] < 40),
    (1 - df["close_position"]) * (40 - df["rsi_14"]) / 100 * df["vol_trend_rank"],
    0
)

# ============================================================
# 10. 量价背离因子 (Volume-Price Divergence)
# 价格与成交量趋势背离，识别潜在反转
# ============================================================
df["price_up"] = (df["ret_5d"] > 0).astype(int)
df["vol_down"] = (df["vol_trend"] < 0).astype(int)

# 价涨量跌 = 看跌背离; 价跌量涨 = 看涨背离
factors["vol_price_divergence"] = np.where(
    (df["price_up"] == 1) & (df["vol_down"] == 1), -1,  # 看跌
    np.where(
        (df["price_up"] == 0) & (df["vol_down"] == 0), 1,  # 看涨
        0
    )
)

# ============================================================
# 11. 市场状态自适应因子 (Market Regime Adaptive)
# 根据市场状态调整因子权重
# ============================================================
# 计算市场整体波动率
market_vol = df.groupby("trade_date")["ret_1d"].transform("std")
df["market_vol_ma20"] = market_vol.rolling(20).mean()
df["market_vol_percentile"] = df["market_vol_ma20"].rank(pct=True)

# 低波动市场用动量，高波动市场用反转
factors["regime_adaptive"] = np.where(
    df["market_vol_percentile"] < 0.5,
    df["momentum_20d_rank"],  # 低波动: 趋势跟随
    1 - df["momentum_20d_rank"]  # 高波动: 反转
)

# ============================================================
# 12. 跨期限动量一致性 (Multi-Horizon Momentum Consensus)
# 多个动量期限一致时信号更强
# ============================================================
df["mom_5d_rank"] = df.groupby("trade_date")["ret_5d"].rank(pct=True)
df["mom_10d_rank"] = df.groupby("trade_date")["ret_10d"].rank(pct=True)
df["mom_20d_rank"] = df.groupby("trade_date")["ret_20d"].rank(pct=True)
df["mom_60d_rank"] = df.groupby("trade_date")["ret_60d"].rank(pct=True)

# 计算一致性得分
mom_ranks = df[["mom_5d_rank", "mom_10d_rank", "mom_20d_rank", "mom_60d_rank"]]
factors["momentum_consensus"] = mom_ranks.mean(axis=1) * (1 - mom_ranks.std(axis=1))

# ============================================================
# 13. 条件波动率因子 (Conditional Volatility)
# 根据市场状态调整的波动率
# ============================================================
df["downside_ret"] = df["ret_1d"].where(df["ret_1d"] < 0, 0)
df["downside_vol"] = df.groupby("ts_code")["downside_ret"].transform(lambda x: x.rolling(20).std())
df["upside_ret"] = df["ret_1d"].where(df["ret_1d"] > 0, 0)
df["upside_vol"] = df.groupby("ts_code")["upside_ret"].transform(lambda x: x.rolling(20).std())

# 上行波动率/下行波动率
factors["vol_asymmetry"] = df["upside_vol"] / df["downside_vol"].replace(0, np.nan)

# ============================================================
# 14. 智能资金流向因子 (Smart Money Flow)
# 大单资金流向 vs 整体流向
# ============================================================
df["intraday_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
df["close_vs_open"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

# 收盘价接近最高价 = 买方力量强
factors["smart_money_flow"] = (
    (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan) *
    df["vol"].rank(pct=True)
)

# ============================================================
# 15. 复合论文因子 (Composite Paper Factor)
# 综合多个论文因子
# ============================================================
factors["composite_paper"] = (
    0.2 * factors["trend_quality"].rank(pct=True) +
    0.15 * factors["vol_adj_momentum"].rank(pct=True) +
    0.15 * factors["liquidity_adj_momentum"].rank(pct=True) +
    0.15 * factors["momentum_consensus"].rank(pct=True) +
    0.1 * factors["smart_money_flow"].rank(pct=True) +
    0.1 * factors["vol_asymmetry"].rank(pct=True) +
    0.15 * factors["regime_adaptive"]
)

# ============================================================
# 因子IC分析
# ============================================================
print("\n" + "=" * 60)
print("因子IC分析")
print("=" * 60)

factor_cols = [
    "trend_quality", "exhaustion_vol_instability", "vol_calm_reversal",
    "vol_stability_mom_divergence", "rsi_bounce_strength", "macd_cross_strength",
    "vol_adj_momentum", "liquidity_adj_momentum", "bottom_reversal",
    "vol_price_divergence", "regime_adaptive", "momentum_consensus",
    "vol_asymmetry", "smart_money_flow", "composite_paper"
]

# 计算未来收益
df["fwd_ret_20d"] = df.groupby("ts_code")["close"].pct_change(20).shift(-20)

# 月度采样
df["ym"] = df["trade_date"].dt.to_period("M")
month_end = df.groupby(["ts_code", "ym"])["trade_date"].transform("max")
monthly_mask = df["trade_date"] == month_end
monthly = df[monthly_mask].copy()
monthly = monthly.merge(factors[factor_cols + ["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="left")

print(f"月度截面数: {monthly['ym'].nunique()}")

ic_results = {}
for col in factor_cols:
    monthly_ic = []
    for ym, group in monthly.groupby("ym"):
        valid = group[[col, "fwd_ret_20d"]].dropna()
        if len(valid) > 20:
            ic = valid[col].corr(valid["fwd_ret_20d"])
            if not np.isnan(ic):
                monthly_ic.append(ic)
    if len(monthly_ic) < 5:
        continue
    ic_arr = np.array(monthly_ic)
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_pos_pct = (ic_arr > 0).mean()
    ic_results[col] = {"IC": round(ic_mean, 4), "IC_IR": round(ic_ir, 4), "IC>0%": round(ic_pos_pct * 100, 2)}

print("\n因子IC分析结果:")
print("-" * 75)
print(f"{'因子名称':<35} {'IC':>8} {'IC_IR':>8} {'IC>0%':>8} {'有效性':>10}")
print("-" * 75)

valid_factors = []
for col, res in sorted(ic_results.items(), key=lambda x: abs(x[1]["IC_IR"]), reverse=True):
    if abs(res["IC_IR"]) >= 0.3:
        validity = "✅ 有效"
        valid_factors.append(col)
    elif abs(res["IC_IR"]) >= 0.2:
        validity = "⚠️ 边缘"
    else:
        validity = "❌ 无效"
    print(f"{col:<35} {res['IC']:>8.4f} {res['IC_IR']:>8.4f} {res['IC>0%']:>7.1f}% {validity:>10}")

print(f"\n有效因子: {len(valid_factors)}/{len(factor_cols)}")
if valid_factors:
    print(f"有效因子列表: {', '.join(valid_factors)}")

# 保存结果
factors.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_factors.csv"), index=False)
pd.DataFrame(ic_results).T.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_factor_ic.csv"))

print("\n✅ arXiv论文因子IC分析完成")
