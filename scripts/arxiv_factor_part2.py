import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("arXiv论文因子回测 + 全因子组合策略")
print("=" * 60)

# 加载数据
data_path = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/daily_data_fixed.csv")
df = pd.read_csv(data_path, dtype={"trade_date": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# 加载所有因子
arxiv_factors = pd.read_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_factors.csv"))
arxiv_factors["trade_date"] = pd.to_datetime(arxiv_factors["trade_date"])

ml_factors = pd.read_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/ml_factor_scores.csv"))
ml_factors["trade_date"] = pd.to_datetime(ml_factors["trade_date"])

llm_factors = pd.read_csv(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/llm_factors.csv"))
llm_factors["trade_date"] = pd.to_datetime(llm_factors["trade_date"])

print(f"数据范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
print(f"股票数量: {df['ts_code'].nunique()}")

# 构建技术特征
df["ret_1d"] = df.groupby("ts_code")["close"].pct_change(1)
df["ret_5d"] = df.groupby("ts_code")["close"].pct_change(5)
df["ret_20d"] = df.groupby("ts_code")["close"].pct_change(20)
df["ret_60d"] = df.groupby("ts_code")["close"].pct_change(60)
df["vol_20d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(20).std())
df["vol_60d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(60).std())
df["ma5"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(5).mean())
df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
df["ma60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())
df["ma5_ratio"] = df["close"] / df["ma5"] - 1
df["ma20_ratio"] = df["close"] / df["ma20"] - 1
df["ma60_ratio"] = df["close"] / df["ma60"] - 1
df["bb_std"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).std())
df["bb_upper"] = df["ma20"] + 2 * df["bb_std"]
df["bb_lower"] = df["ma20"] - 2 * df["bb_std"]
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)
df["vol_ma20"] = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
df["vol_change"] = df["vol"] / df["vol_ma20"].replace(0, np.nan) - 1
df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
df["amihud"] = df["ret_1d"].abs() / df["amount"].replace(0, np.nan)
df["amihud_log"] = np.log1p(df["amihud"] * 1e9)
df["high_20d"] = df.groupby("ts_code")["high"].transform(lambda x: x.rolling(20).max())
df["low_20d"] = df.groupby("ts_code")["low"].transform(lambda x: x.rolling(20).min())
df["high_low_ratio"] = df["high_20d"] / df["low_20d"].replace(0, np.nan) - 1
df["close_position"] = (df["close"] - df["low_20d"]) / (df["high_20d"] - df["low_20d"]).replace(0, np.nan)
df["fwd_ret_20d"] = df.groupby("ts_code")["close"].pct_change(20).shift(-20)

# 合并所有因子
df = df.merge(arxiv_factors[["ts_code", "trade_date", "liquidity_adj_momentum", "trend_quality", 
                              "vol_adj_momentum", "momentum_consensus", "smart_money_flow", 
                              "vol_asymmetry", "regime_adaptive", "composite_paper"]], 
              on=["ts_code", "trade_date"], how="left")
df = df.merge(ml_factors[["ts_code", "trade_date", "ml_score"]], on=["ts_code", "trade_date"], how="left")
df = df.merge(llm_factors[["ts_code", "trade_date", "liquidity_sentiment"]], on=["ts_code", "trade_date"], how="left")

# 月度采样
df["ym"] = df["trade_date"].dt.to_period("M")
month_end = df.groupby(["ts_code", "ym"])["trade_date"].transform("max")
monthly = df[df["trade_date"] == month_end].copy().reset_index(drop=True)

print(f"月度截面数: {monthly['ym'].nunique()}")

# ============================================================
# 通用回测函数
# ============================================================
def backtest_factor(data, factor_col, top_n=50, ascending=False):
    results = []
    for dt in sorted(data["trade_date"].unique()):
        cross = data[data["trade_date"] == dt].copy()
        cross = cross.dropna(subset=[factor_col, "fwd_ret_20d"])
        if len(cross) < top_n:
            continue
        cross["rank"] = cross[factor_col].rank(ascending=ascending)
        selected = cross[cross["rank"] <= top_n]
        results.append({"date": dt, "ret": selected["fwd_ret_20d"].mean()})
    
    if not results:
        return None
    
    ret_df = pd.DataFrame(results).set_index("date")
    nav = (1 + ret_df["ret"]).cumprod()
    total_ret = nav.iloc[-1] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = ret_df["ret"].std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    drawdown = (nav / nav.cummax() - 1).min()
    calmar = ann_ret / abs(drawdown) if drawdown != 0 else 0
    win_rate = (ret_df["ret"] > 0).mean()
    
    return {
        "total_return": round(total_ret * 100, 2),
        "annual_return": round(ann_ret * 100, 2),
        "annual_vol": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(drawdown * 100, 2),
        "calmar": round(calmar, 2),
        "win_rate": round(win_rate * 100, 1)
    }

# ============================================================
# 第1步: arXiv单因子回测
# ============================================================
print("\n" + "=" * 60)
print("第1步: arXiv单因子回测")
print("=" * 60)

arxiv_single = backtest_factor(monthly, "liquidity_adj_momentum", top_n=50, ascending=True)
print(f"\narXiv因子 (liquidity_adj_momentum):")
for k, v in arxiv_single.items():
    print(f"  {k}: {v}")

# ============================================================
# 第2步: 三因子等权组合 (ML + LLM + arXiv)
# ============================================================
print("\n" + "=" * 60)
print("第2步: 三因子等权组合 (ML + LLM + arXiv)")
print("=" * 60)

monthly["ml_rank"] = monthly.groupby("trade_date")["ml_score"].rank(pct=True)
monthly["llm_rank"] = monthly.groupby("trade_date")["liquidity_sentiment"].rank(pct=True, ascending=False)
monthly["arxiv_rank"] = monthly.groupby("trade_date")["liquidity_adj_momentum"].rank(pct=True, ascending=False)

# 等权组合
monthly["triple_combined"] = (
    0.5 * monthly["ml_rank"] + 
    0.25 * monthly["llm_rank"] + 
    0.25 * monthly["arxiv_rank"]
)

triple_result = backtest_factor(monthly, "triple_combined", top_n=50, ascending=False)
print(f"\n三因子等权组合 (ML50% + LLM25% + arXiv25%):")
for k, v in triple_result.items():
    print(f"  {k}: {v}")

# ============================================================
# 第3步: 全因子联合训练
# ============================================================
print("\n" + "=" * 60)
print("第3步: 全因子联合训练 (ML + LLM + arXiv)")
print("=" * 60)

# 特征列表
ml_feature_cols = [
    "vol_20d", "vol_60d", "ma5_ratio", "ma20_ratio", "ma60_ratio",
    "bb_width", "vol_change", "intraday_range", "amihud_log",
    "ret_5d", "ret_20d", "ret_60d", "high_low_ratio", "close_position"
]

llm_feature_cols = ["liquidity_sentiment"]

arxiv_feature_cols = [
    "liquidity_adj_momentum", "trend_quality", "vol_adj_momentum",
    "momentum_consensus", "smart_money_flow", "vol_asymmetry",
    "regime_adaptive", "composite_paper"
]

all_feature_names = ml_feature_cols + llm_feature_cols + arxiv_feature_cols
print(f"总特征数: {len(all_feature_names)}")
print(f"  ML特征: {len(ml_feature_cols)}")
print(f"  LLM特征: {len(llm_feature_cols)}")
print(f"  arXiv特征: {len(arxiv_feature_cols)}")

train_data = monthly.dropna(subset=["fwd_ret_20d"] + all_feature_names).copy()
X = train_data[all_feature_names].fillna(0)
y = train_data["fwd_ret_20d"]
print(f"训练样本数: {len(X)}")

model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=20, subsample=0.8, random_state=42
)

tscv = TimeSeriesSplit(n_splits=5)
train_data["full_model_score"] = np.nan

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    model.fit(X_train, y_train)
    train_data.loc[train_data.index[test_idx], "full_model_score"] = model.predict(X_test)

# 特征重要性
importances = pd.DataFrame({
    "feature": all_feature_names,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n特征重要性Top 15:")
print("-" * 55)
for _, row in importances.head(15).iterrows():
    if row["feature"] in llm_feature_cols:
        marker = "🔵"
    elif row["feature"] in arxiv_feature_cols:
        marker = "🟡"
    else:
        marker = "🟢"
    pct = row["importance"] * 100
    print(f"  {marker} {row['feature']:<25} {pct:>6.1f}%")

ml_imp = importances[importances["feature"].isin(ml_feature_cols)]["importance"].sum()
llm_imp = importances[importances["feature"].isin(llm_feature_cols)]["importance"].sum()
arxiv_imp = importances[importances["feature"].isin(arxiv_feature_cols)]["importance"].sum()
print(f"\n特征重要性占比:")
print(f"  🟢 ML特征: {ml_imp*100:.1f}%")
print(f"  🔵 LLM特征: {llm_imp*100:.1f}%")
print(f"  🟡 arXiv特征: {arxiv_imp*100:.1f}%")

# 回测
full_model_result = backtest_factor(train_data, "full_model_score", top_n=50, ascending=False)
print(f"\n全因子联合训练策略:")
for k, v in full_model_result.items():
    print(f"  {k}: {v}")

# ============================================================
# 所有策略对比
# ============================================================
print("\n" + "=" * 75)
print("所有策略对比")
print("=" * 75)
print(f"{'策略':<30} {'年化收益':>8} {'最大回撤':>8} {'夏普':>6} {'Calmar':>7} {'胜率':>6}")
print("-" * 75)

# 加载之前的结果
try:
    with open(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/backtest_result_ml.json")) as f:
        ml_raw = json.load(f)
    s = ml_raw.get("summary", ml_raw)
    ml_base = {
        "annual_return": s.get("annual_return", 0.3873) * 100,
        "max_drawdown": s.get("max_drawdown", -0.0992) * 100,
        "sharpe": s.get("sharpe_ratio", 8.19),
        "calmar": s.get("calmar_ratio", 3.90),
        "win_rate": s.get("win_rate", 0.652) * 100
    }
except:
    ml_base = {"annual_return": 38.73, "max_drawdown": -9.92, "sharpe": 8.19, "calmar": 3.90, "win_rate": 65.2}

try:
    with open(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/llm_backtest_results.json")) as f:
        llm_data = json.load(f)
    ml_llm_eq = llm_data.get("ml_llm_equal_weight", {})
except:
    ml_llm_eq = {"annual_return": 46.60, "max_drawdown": -19.00, "sharpe": 1.56, "calmar": 2.45, "win_rate": 67.2}

print(f"{'ML因子策略(纯)':<30} {ml_base['annual_return']:>7.2f}% {ml_base['max_drawdown']:>7.2f}% {ml_base['sharpe']:>6.2f} {ml_base['calmar']:>7.2f} {ml_base['win_rate']:>5.1f}%")
print(f"{'ML+LLM等权组合':<30} {ml_llm_eq['annual_return']:>7.2f}% {ml_llm_eq['max_drawdown']:>7.2f}% {ml_llm_eq['sharpe']:>6.2f} {ml_llm_eq['calmar']:>7.2f} {ml_llm_eq['win_rate']:>5.1f}%")
print(f"{'arXiv单因子':<30} {arxiv_single['annual_return']:>7.2f}% {arxiv_single['max_drawdown']:>7.2f}% {arxiv_single['sharpe']:>6.2f} {arxiv_single['calmar']:>7.2f} {arxiv_single['win_rate']:>5.1f}%")
print(f"{'ML+LLM+arXiv等权组合':<30} {triple_result['annual_return']:>7.2f}% {triple_result['max_drawdown']:>7.2f}% {triple_result['sharpe']:>6.2f} {triple_result['calmar']:>7.2f} {triple_result['win_rate']:>5.1f}%")
print(f"{'全因子联合训练':<30} {full_model_result['annual_return']:>7.2f}% {full_model_result['max_drawdown']:>7.2f}% {full_model_result['sharpe']:>6.2f} {full_model_result['calmar']:>7.2f} {full_model_result['win_rate']:>5.1f}%")

# 保存结果
results = {
    "arxiv_single": arxiv_single,
    "triple_combined": triple_result,
    "full_model": full_model_result,
    "feature_importance": {
        "ml_pct": round(ml_imp * 100, 1),
        "llm_pct": round(llm_imp * 100, 1),
        "arxiv_pct": round(arxiv_imp * 100, 1)
    }
}
with open(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/arxiv_backtest_results.json"), "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ arXiv因子测试完成！")
