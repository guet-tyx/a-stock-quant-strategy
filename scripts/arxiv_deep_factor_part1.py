import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("深度arXiv论文因子挖掘 - 第二轮")
print("=" * 60)

# 加载数据
data_path = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/daily_data_fixed.csv")
df = pd.read_csv(data_path, dtype={"trade_date": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

print(f"数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"股票数量: {df['ts_code'].nunique()}")

# ============================================================
# 来自论文的具体因子公式
# ============================================================
print("\n" + "=" * 60)
print("构建论文公式因子")
print("=" * 60)

factors = pd.DataFrame()
factors["ts_code"] = df["ts_code"]
factors["trade_date"] = df["trade_date"]

# ============================================================
# 101 Formulaic Alphas 经典因子
# ============================================================

# Alpha#6: -1 * Corr(open, volume, 10)
factors["alpha_006"] = -df.groupby("ts_code").apply(
    lambda x: x["open"].rolling(10).corr(x["vol"])
).reset_index(level=0, drop=True)

# Alpha#14: -1 * Rank(Delta(close/close_prev - 1, 3)) * Corr(open, volume, 10)
df["close_ret"] = df.groupby("ts_code")["close"].pct_change(1)
df["delta_ret_3"] = df.groupby("ts_code")["close_ret"].diff(3)
df["corr_open_vol_10"] = df.groupby("ts_code").apply(
    lambda x: x["open"].rolling(10).corr(x["vol"])
).reset_index(level=0, drop=True)
factors["alpha_014"] = -df.groupby("trade_date")["delta_ret_3"].rank(pct=True) * df["corr_open_vol_10"]

# Alpha#35: Rank(volume) * (1 - Rank(close+high-low)) * (1 - Rank(close_ret))
df["hl_range"] = df["close"] + df["high"] - df["low"]
factors["alpha_035"] = (
    df.groupby("trade_date")["vol"].rank(pct=True) *
    (1 - df.groupby("trade_date")["hl_range"].rank(pct=True)) *
    (1 - df.groupby("trade_date")["close_ret"].rank(pct=True))
)

# ============================================================
# 行为驱动因子 (来自行为驱动多因子论文)
# ============================================================

# K线实体强度
df["kline_body"] = (df["close"] - df["open"]).abs()
df["kline_range"] = df["high"] - df["low"] + 0.001
factors["kline_body_strength"] = (df["close"] - df["open"]) / df["kline_range"]

# 收盘价接近最低价 (卖方力量)
factors["close_near_low"] = (df["high"] - df["close"]) / df["kline_range"]

# 收盘价接近最高价 (买方力量)
factors["close_near_high"] = (df["close"] - df["low"]) / df["kline_range"]

# 收盘价相对中位数偏离
factors["close_median_dev"] = (df["close"] - (df["high"] + df["low"]) / 2) / df["kline_range"]

# 短期均线差异排名乘积
df["ma5"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(5).mean())
df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
df["ma60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())
df["ma120"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(120).mean())
df["ma200"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(200).mean())

factors["short_ma_diff_rank"] = (
    df.groupby("trade_date")["ma5"].apply(lambda x: (x - x.mean()) / x.std()).reset_index(level=0, drop=True) *
    df.groupby("trade_date")["ma20"].apply(lambda x: (x - x.mean()) / x.std()).reset_index(level=0, drop=True)
)

# 收盘价相对VWAP偏离 (假设VWAP ≈ (H+L+C)/3)
df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
factors["close_vwap_diff"] = df.groupby("trade_date").apply(
    lambda x: (x["close"] - x["vwap"]).rank(pct=True)
).reset_index(level=0, drop=True)

# 5日动量反转
factors["mom_5d_reversal"] = -df.groupby("trade_date").apply(
    lambda x: x.groupby("ts_code")["close"].pct_change(5).rank(pct=True)
).reset_index(level=0, drop=True)

# 收盘价相对MA200偏离
factors["close_ma200_diff"] = df.groupby("trade_date").apply(
    lambda x: (x["ma200"] - x["close"]).rank(pct=True)
).reset_index(level=0, drop=True)

# ============================================================
# AlphaForge 发现的高IC因子
# ============================================================

# ts_cov(high, volume, 20) 的对数变换
df["cov_high_vol_20"] = df.groupby("ts_code").apply(
    lambda x: x["high"].rolling(20).cov(x["vol"])
).reset_index(level=0, drop=True)
factors["alphaforge_cov"] = np.sign(df["cov_high_vol_20"]) * np.log1p(df["cov_high_vol_20"].abs())

# ts_corr(close, volume, 10) 的各种变换
df["corr_close_vol_10"] = df.groupby("ts_code").apply(
    lambda x: x["close"].rolling(10).corr(x["vol"])
).reset_index(level=0, drop=True)
factors["alphaforge_corr_cv"] = df["corr_close_vol_10"]

# ts_corr(high, volume, 5) 的最小值
df["corr_high_vol_5"] = df.groupby("ts_code").apply(
    lambda x: x["high"].rolling(5).corr(x["vol"])
).reset_index(level=0, drop=True)
factors["alphaforge_corr_hv"] = df.groupby("ts_code")["corr_high_vol_5"].transform(
    lambda x: x.rolling(10).min()
)

# ts_mad(volume, 50) 的变换
df["vol_mad_50"] = df.groupby("ts_code")["vol"].transform(
    lambda x: x.rolling(50).apply(lambda y: np.median(np.abs(y - np.median(y))), raw=True)
)
factors["alphaforge_vol_mad"] = np.sign(df["vol_mad_50"]) * np.log1p(df["vol_mad_50"].abs())

# ============================================================
# 复合技术因子
# ============================================================

# 波动率调整的动量
df["vol_20d"] = df.groupby("ts_code")["close_ret"].transform(lambda x: x.rolling(20).std())
df["ret_20d"] = df.groupby("ts_code")["close"].pct_change(20)
factors["vol_adj_mom_20"] = df["ret_20d"] / df["vol_20d"].replace(0, np.nan)

# 量价趋势强度
df["vol_ma20"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
df["vol_trend"] = df["vol"] / df["vol_ma20"].replace(0, np.nan) - 1
df["price_trend"] = df["close"] / df["ma20"] - 1
factors["vol_price_trend"] = df["vol_trend"].rank(pct=True) * df["price_trend"].rank(pct=True)

# 高低价位置波动率
df["high_20d"] = df.groupby("ts_code")["high"].transform(lambda x: x.rolling(20).max())
df["low_20d"] = df.groupby("ts_code")["low"].transform(lambda x: x.rolling(20).min())
df["close_pos"] = (df["close"] - df["low_20d"]) / (df["high_20d"] - df["low_20d"]).replace(0, np.nan)
factors["position_vol"] = df["close_pos"].rank(pct=True) * (1 - df["vol_20d"].rank(pct=True))

# 跳空缺口因子
df["gap"] = df["open"] / df.groupby("ts_code")["close"].shift(1) - 1
df["gap_ma5"] = df.groupby("ts_code")["gap"].transform(lambda x: x.rolling(5).mean())
factors["gap_factor"] = -df.groupby("trade_date")["gap_ma5"].rank(pct=True)

# 日内波动率因子
df["intraday_vol"] = (df["high"] - df["low"]) / df["open"]
df["intraday_vol_ma20"] = df.groupby("ts_code")["intraday_vol"].transform(lambda x: x.rolling(20).mean())
factors["intraday_vol_factor"] = -df.groupby("trade_date")["intraday_vol_ma20"].rank(pct=True)

# 成交量加权收益
df["vol_weighted_ret"] = df["close_ret"] * df["vol"].rank(pct=True)
factors["vol_weighted_mom"] = df.groupby("ts_code")["vol_weighted_ret"].transform(lambda x: x.rolling(20).mean())

# 价格效率因子 (趋势的持续性)
df["ret_5d"] = df.groupby("ts_code")["close"].pct_change(5)
df["ret_10d"] = df.groupby("ts_code")["close"].pct_change(10)
df["efficiency"] = df["ret_5d"].abs() / df["ret_10d"].abs().replace(0, np.nan)
factors["price_efficiency"] = df["efficiency"].rank(pct=True)

# 均线支撑强度
factors["ma_support"] = np.where(
    df["close"] > df["ma60"],
    (df["close"] - df["ma60"]) / df["close"],  # 在均线上方=正
    -(df["ma60"] - df["close"]) / df["close"]  # 在均线下方=负
)

# RSI因子
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

df["rsi_14"] = df.groupby("ts_code")["close"].transform(lambda x: calc_rsi(x, 14))
factors["rsi_factor"] = -df.groupby("trade_date")["rsi_14"].rank(pct=True)  # 低RSI=超卖

# MACD因子
df["ema12"] = df.groupby("ts_code")["close"].transform(lambda x: x.ewm(span=12).mean())
df["ema26"] = df.groupby("ts_code")["close"].transform(lambda x: x.ewm(span=26).mean())
df["macd_line"] = df["ema12"] - df["ema26"]
df["macd_signal"] = df.groupby("ts_code")["macd_line"].transform(lambda x: x.ewm(span=9).mean())
df["macd_hist"] = df["macd_line"] - df["macd_signal"]
factors["macd_factor"] = df.groupby("trade_date")["macd_hist"].rank(pct=True)

# 布林带位置
df["bb_mid"] = df["ma20"]
df["bb_std"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).std())
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
factors["bb_position_factor"] = -df.groupby("trade_date")["bb_position"].rank(pct=True)  # 低位置=超卖

# ============================================================
# 因子IC分析
# ============================================================
print("\n" + "=" * 60)
print("因子IC分析")
print("=" * 60)

factor_cols = [col for col in factors.columns if col not in ["ts_code", "trade_date"]]

# 计算未来收益
df["fwd_ret_20d"] = df.groupby("ts_code")["close"].pct_change(20).shift(-20)

# 月度采样
df["ym"] = df["trade_date"].dt.to_period("M")
month_end = df.groupby(["ts_code", "ym"])["trade_date"].transform("max")
monthly_mask = df["trade_date"] == month_end
monthly = df[monthly_mask].copy()
monthly = monthly.merge(factors[factor_cols + ["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="left")

print(f"月度截面数: {monthly['ym'].nunique()}")
print(f"测试因子数: {len(factor_cols)}")

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

print("\n因子IC分析结果 (按IC_IR排序):")
print("-" * 80)
print(f"{'因子名称':<30} {'IC':>8} {'IC_IR':>8} {'IC>0%':>8} {'有效性':>10}")
print("-" * 80)

valid_factors = []
for col, res in sorted(ic_results.items(), key=lambda x: abs(x[1]["IC_IR"]), reverse=True):
    if abs(res["IC_IR"]) >= 0.3:
        validity = "✅ 有效"
        valid_factors.append(col)
    elif abs(res["IC_IR"]) >= 0.2:
        validity = "⚠️ 边缘"
    else:
        validity = "❌ 无效"
    print(f"{col:<30} {res['IC']:>8.4f} {res['IC_IR']:>8.4f} {res['IC>0%']:>7.1f}% {validity:>10}")

print(f"\n有效因子: {len(valid_factors)}/{len(factor_cols)}")
if valid_factors:
    print(f"有效因子列表: {', '.join(valid_factors)}")

# 保存结果
factors.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_deep_factors.csv"), index=False)
pd.DataFrame(ic_results).T.to_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_deep_factor_ic.csv"))

print("\n✅ 深度arXiv论文因子IC分析完成")
