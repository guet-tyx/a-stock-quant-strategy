import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("策略全面评估系统")
print("=" * 70)

# ============================================================
# 加载数据和因子
# ============================================================
data_path = os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/daily_data_fixed.csv")
df = pd.read_csv(data_path, dtype={"trade_date": str})
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

# 加载因子
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
df["ret_20d"] = df.groupby("ts_code")["close"].pct_change(20)
df["vol_20d"] = df.groupby("ts_code")["ret_1d"].transform(lambda x: x.rolling(20).std())
df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
df["ma60"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(60).mean())

# 合并因子
df = df.merge(arxiv_factors[["ts_code", "trade_date", "liquidity_adj_momentum"]], on=["ts_code", "trade_date"], how="left")
df = df.merge(ml_factors[["ts_code", "trade_date", "ml_score"]], on=["ts_code", "trade_date"], how="left")
df = df.merge(llm_factors[["ts_code", "trade_date", "liquidity_sentiment"]], on=["ts_code", "trade_date"], how="left")

# 月度采样
df["ym"] = df["trade_date"].dt.to_period("M")
month_end = df.groupby(["ts_code", "ym"])["trade_date"].transform("max")
monthly = df[df["trade_date"] == month_end].copy().reset_index(drop=True)

# 计算未来收益
monthly["fwd_ret"] = monthly.groupby("ts_code")["close"].pct_change(1).shift(-1)

# 构建组合因子
monthly["ml_rank"] = monthly.groupby("trade_date")["ml_score"].rank(pct=True)
monthly["llm_rank"] = monthly.groupby("trade_date")["liquidity_sentiment"].rank(pct=True, ascending=False)
monthly["arxiv_rank"] = monthly.groupby("trade_date")["liquidity_adj_momentum"].rank(pct=True, ascending=False)
monthly["combined"] = (
    0.4 * monthly["ml_rank"] + 
    0.3 * monthly["llm_rank"] + 
    0.3 * monthly["arxiv_rank"]
)

# ============================================================
# 第1部分: 市场环境分析
# ============================================================
print("\n" + "=" * 70)
print("第1部分: 市场环境分析")
print("=" * 70)

# 计算市场收益
market_ret = monthly.groupby("trade_date")["fwd_ret"].mean().reset_index()
market_ret.columns = ["date", "market_ret"]
market_ret = market_ret.sort_values("date").reset_index(drop=True)

# 计算市场状态
market_ret["cum_market"] = (1 + market_ret["market_ret"]).cumprod()
market_ret["drawdown"] = market_ret["cum_market"] / market_ret["cum_market"].cummax() - 1
market_ret["vol_20d"] = market_ret["market_ret"].rolling(20).std()

# 定义市场环境
def classify_market_env(row):
    if row["market_ret"] > 0.05:  # 月涨>5%
        return "牛市"
    elif row["market_ret"] < -0.05:  # 月跌>5%
        return "熊市"
    elif abs(row["market_ret"]) < 0.02:  # 月波动<2%
        return "震荡市"
    else:
        return "温和市"

market_ret["market_env"] = market_ret.apply(classify_market_env, axis=1)

print("\n市场环境分布:")
print("-" * 50)
env_counts = market_ret["market_env"].value_counts()
for env, count in env_counts.items():
    pct = count / len(market_ret) * 100
    print(f"{env}: {count} 个月 ({pct:.1f}%)")

# ============================================================
# 第2部分: 策略在不同市场环境下的表现
# ============================================================
print("\n" + "=" * 70)
print("第2部分: 策略在不同市场环境下的表现")
print("=" * 70)

# 回测函数
def backtest_strategy(data, factor_col, top_n=50, ascending=False):
    results = []
    for dt in sorted(data["trade_date"].unique()):
        cross = data[data["trade_date"] == dt].copy()
        cross = cross.dropna(subset=[factor_col, "fwd_ret"])
        if len(cross) < top_n:
            continue
        cross["rank"] = cross[factor_col].rank(ascending=ascending)
        selected = cross[cross["rank"] <= top_n]
        results.append({"date": dt, "ret": selected["fwd_ret"].mean()})
    
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
        "nav": nav,
        "ret_df": ret_df,
        "total_return": total_ret,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "calmar": calmar,
        "win_rate": win_rate
    }

# 运行策略
strategy_result = backtest_strategy(monthly, "combined", top_n=50, ascending=False)

if strategy_result:
    # 合并市场环境
    strategy_ret = strategy_result["ret_df"].copy()
    strategy_ret = strategy_ret.merge(market_ret[["date", "market_env", "market_ret"]], left_index=True, right_on="date", how="left")
    strategy_ret = strategy_ret.set_index("date")
    
    print("\n策略在不同市场环境下的表现:")
    print("-" * 70)
    print(f"{'市场环境':<10} {'月数':>6} {'策略收益':>10} {'市场收益':>10} {'超额收益':>10} {'胜率':>8}")
    print("-" * 70)
    
    for env in ["牛市", "温和市", "震荡市", "熊市"]:
        env_data = strategy_ret[strategy_ret["market_env"] == env]
        if len(env_data) > 0:
            strategy_mean = env_data["ret"].mean() * 100
            market_mean = env_data["market_ret"].mean() * 100
            excess = strategy_mean - market_mean
            win_rate = (env_data["ret"] > 0).mean() * 100
            print(f"{env:<10} {len(env_data):>6} {strategy_mean:>9.2f}% {market_mean:>9.2f}% {excess:>9.2f}% {win_rate:>7.1f}%")

# ============================================================
# 第3部分: 风险指标分析
# ============================================================
print("\n" + "=" * 70)
print("第3部分: 风险指标分析")
print("=" * 70)

if strategy_result:
    ret_series = strategy_result["ret_df"]["ret"]
    
    print("\n风险指标:")
    print("-" * 50)
    
    # 基础风险指标
    print(f"年化波动率: {strategy_result['annual_vol']*100:.2f}%")
    print(f"最大回撤: {strategy_result['max_drawdown']*100:.2f}%")
    print(f"夏普比率: {strategy_result['sharpe']:.2f}")
    print(f"Calmar比率: {strategy_result['calmar']:.2f}")
    
    # VaR和CVaR
    var_95 = np.percentile(ret_series, 5)
    var_99 = np.percentile(ret_series, 1)
    cvar_95 = ret_series[ret_series <= var_95].mean()
    cvar_99 = ret_series[ret_series <= var_99].mean()
    
    print(f"\nVaR (95%): {var_95*100:.2f}%")
    print(f"VaR (99%): {var_99*100:.2f}%")
    print(f"CVaR (95%): {cvar_95*100:.2f}%")
    print(f"CVaR (99%): {cvar_99*100:.2f}%")
    
    # 偏度和峰度
    skewness = ret_series.skew()
    kurtosis = ret_series.kurtosis()
    
    print(f"\n偏度: {skewness:.2f}")
    print(f"峰度: {kurtosis:.2f}")
    
    # 收益分布
    print(f"\n收益分布:")
    print(f"  正收益月数: {(ret_series > 0).sum()} ({(ret_series > 0).mean()*100:.1f}%)")
    print(f"  负收益月数: {(ret_series < 0).sum()} ({(ret_series < 0).mean()*100:.1f}%)")
    print(f"  最大单月收益: {ret_series.max()*100:.2f}%")
    print(f"  最大单月亏损: {ret_series.min()*100:.2f}%")
    print(f"  平均收益: {ret_series.mean()*100:.2f}%")
    print(f"  中位数收益: {ret_series.median()*100:.2f}%")

# ============================================================
# 第4部分: 收益归因分析
# ============================================================
print("\n" + "=" * 70)
print("第4部分: 收益归因分析")
print("=" * 70)

if strategy_result:
    # 计算各因子的贡献
    print("\n因子收益贡献:")
    print("-" * 50)
    
    # 计算因子IC
    factor_cols = ["ml_score", "liquidity_sentiment", "liquidity_adj_momentum"]
    factor_names = ["ML因子", "LLM因子", "arXiv因子"]
    
    for factor, name in zip(factor_cols, factor_names):
        ic_series = []
        for dt in sorted(monthly["trade_date"].unique()):
            cross = monthly[monthly["trade_date"] == dt].copy()
            cross = cross.dropna(subset=[factor, "fwd_ret"])
            if len(cross) > 20:
                ic = cross[factor].corr(cross["fwd_ret"])
                if not np.isnan(ic):
                    ic_series.append(ic)
        
        if ic_series:
            ic_arr = np.array(ic_series)
            print(f"{name}:")
            print(f"  平均IC: {ic_arr.mean():.4f}")
            print(f"  IC标准差: {ic_arr.std():.4f}")
            print(f"  IC_IR: {ic_arr.mean()/ic_arr.std():.4f}")
            print(f"  IC>0占比: {(ic_arr > 0).mean()*100:.1f}%")

# ============================================================
# 第5部分: 稳定性分析
# ============================================================
print("\n" + "=" * 70)
print("第5部分: 稳定性分析")
print("=" * 70)

if strategy_result:
    nav = strategy_result["nav"]
    ret_series = strategy_result["ret_df"]["ret"]
    
    print("\n稳定性指标:")
    print("-" * 50)
    
    # 滚动夏普比率
    rolling_sharpe = ret_series.rolling(12).mean() / ret_series.rolling(12).std() * np.sqrt(12)
    print(f"滚动夏普比率:")
    print(f"  均值: {rolling_sharpe.mean():.2f}")
    print(f"  标准差: {rolling_sharpe.std():.2f}")
    print(f"  最小值: {rolling_sharpe.min():.2f}")
    print(f"  最大值: {rolling_sharpe.max():.2f}")
    
    # 滚动胜率
    rolling_win_rate = ret_series.rolling(12).apply(lambda x: (x > 0).mean())
    print(f"\n滚动胜率:")
    print(f"  均值: {rolling_win_rate.mean()*100:.1f}%")
    print(f"  标准差: {rolling_win_rate.std()*100:.1f}%")
    print(f"  最小值: {rolling_win_rate.min()*100:.1f}%")
    print(f"  最大值: {rolling_win_rate.max()*100:.1f}%")
    
    # 最大回撤分析
    drawdown_series = nav / nav.cummax() - 1
    max_dd = drawdown_series.min()
    max_dd_end = drawdown_series.idxmin()
    
    # 找到回撤开始点
    max_dd_start = nav[:max_dd_end].idxmax()
    
    print(f"\n最大回撤分析:")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    print(f"  回撤开始: {max_dd_start}")
    print(f"  回撤结束: {max_dd_end}")
    print(f"  持续时间: {(pd.Timestamp(max_dd_end) - pd.Timestamp(max_dd_start)).days} 天")
    
    # 恢复时间
    recovery_mask = nav[max_dd_end:] >= nav[max_dd_start]
    if recovery_mask.any():
        recovery_date = recovery_mask.idxmax()
        recovery_days = (pd.Timestamp(recovery_date) - pd.Timestamp(max_dd_end)).days
        print(f"  恢复时间: {recovery_days} 天")
    else:
        print(f"  恢复时间: 未恢复")

# ============================================================
# 第6部分: 交易成本分析
# ============================================================
print("\n" + "=" * 70)
print("第6部分: 交易成本分析")
print("=" * 70)

if strategy_result:
    print("\n交易成本假设:")
    print("-" * 50)
    
    # 假设参数
    commission_rate = 0.0003  # 佣金率
    slippage_rate = 0.001     # 滑点率
    stamp_tax_rate = 0.001    # 印花税（卖出）
    
    print(f"佣金率: {commission_rate*100:.2f}%")
    print(f"滑点率: {slippage_rate*100:.2f}%")
    print(f"印花税: {stamp_tax_rate*100:.2f}%")
    
    # 计算换手率
    # 假设每月换仓50%的持仓
    turnover_rate = 0.5
    
    # 计算交易成本
    monthly_cost = turnover_rate * (commission_rate * 2 + slippage_rate * 2 + stamp_tax_rate)
    annual_cost = monthly_cost * 12
    
    print(f"\n交易成本:")
    print(f"  假设月换手率: {turnover_rate*100:.0f}%")
    print(f"  月度交易成本: {monthly_cost*100:.4f}%")
    print(f"  年度交易成本: {annual_cost*100:.2f}%")
    
    # 扣除成本后的收益
    net_annual_return = strategy_result["annual_return"] - annual_cost
    net_sharpe = net_annual_return / strategy_result["annual_vol"]
    
    print(f"\n扣除成本后:")
    print(f"  年化收益: {net_annual_return*100:.2f}% (原: {strategy_result['annual_return']*100:.2f}%)")
    print(f"  夏普比率: {net_sharpe:.2f} (原: {strategy_result['sharpe']:.2f})")

# ============================================================
# 第7部分: 压力测试
# ============================================================
print("\n" + "=" * 70)
print("第7部分: 压力测试")
print("=" * 70)

if strategy_result:
    print("\n压力测试场景:")
    print("-" * 50)
    
    # 定义压力场景
    stress_scenarios = {
        "2022年4月（上海封城）": {"start": "2022-04", "end": "2022-04"},
        "2022年10月（市场恐慌）": {"start": "2022-10", "end": "2022-10"},
        "2023年8月（活跃资本市场）": {"start": "2023-08", "end": "2023-08"},
        "2024年2月（量化危机）": {"start": "2024-02", "end": "2024-02"},
        "2024年9月（政策转向）": {"start": "2024-09", "end": "2024-09"},
    }
    
    strategy_ret = strategy_result["ret_df"].copy()
    strategy_ret["ym"] = strategy_ret.index.to_period("M")
    
    print(f"{'压力场景':<25} {'市场收益':>10} {'策略收益':>10} {'超额收益':>10}")
    print("-" * 60)
    
    for scenario_name, period in stress_scenarios.items():
        # 该月市场收益
        market_month = market_ret[
            (market_ret["date"].dt.to_period("M") >= period["start"]) &
            (market_ret["date"].dt.to_period("M") <= period["end"])
        ]
        
        # 该月策略收益
        strategy_month = strategy_ret[
            (strategy_ret["ym"] >= period["start"]) &
            (strategy_ret["ym"] <= period["end"])
        ]
        
        if len(market_month) > 0 and len(strategy_month) > 0:
            mkt_ret = market_month["market_ret"].mean() * 100
            stg_ret = strategy_month["ret"].mean() * 100
            excess = stg_ret - mkt_ret
            
            print(f"{scenario_name:<25} {mkt_ret:>9.2f}% {stg_ret:>9.2f}% {excess:>9.2f}%")

# ============================================================
# 第8部分: 策略容量分析
# ============================================================
print("\n" + "=" * 70)
print("第8部分: 策略容量分析")
print("=" * 70)

print("\n策略容量估算:")
print("-" * 50)

# 假设参数
stocks_count = 50  # 持仓股票数
avg_daily_volume = 5000000  # 平均日成交量（股）
avg_price = 20  # 平均股价（元）
impact_rate = 0.001  # 冲击成本率

# 计算每日可交易金额
daily_tradable = stocks_count * avg_daily_volume * avg_price
monthly_tradable = daily_tradable * 20  # 每月20个交易日

# 假设每月换仓50%
monthly_capacity = monthly_tradable * 0.5

print(f"持仓股票数: {stocks_count}")
print(f"平均日成交量: {avg_daily_volume/10000:.0f} 万股")
print(f"平均股价: {avg_price} 元")
print(f"每日可交易金额: {daily_tradable/100000000:.2f} 亿元")
print(f"每月可交易金额: {monthly_tradable/100000000:.2f} 亿元")
print(f"策略容量估算: {monthly_capacity/100000000:.2f} 亿元")

print("\n容量限制因素:")
print("  1. 个股流动性：小盘股成交量有限")
print("  2. 冲击成本：大额交易会影响价格")
print("  3. 换手率：高换手率需要更多流动性")

# ============================================================
# 总结报告
# ============================================================
print("\n" + "=" * 70)
print("策略全面评估总结")
print("=" * 70)

if strategy_result:
    print("\n📊 核心指标:")
    print(f"  年化收益: {strategy_result['annual_return']*100:.2f}%")
    print(f"  最大回撤: {strategy_result['max_drawdown']*100:.2f}%")
    print(f"  夏普比率: {strategy_result['sharpe']:.2f}")
    print(f"  Calmar比率: {strategy_result['calmar']:.2f}")
    print(f"  胜率: {strategy_result['win_rate']*100:.1f}%")
    
    print("\n📈 风险指标:")
    print(f"  年化波动率: {strategy_result['annual_vol']*100:.2f}%")
    print(f"  VaR (95%): {var_95*100:.2f}%")
    print(f"  CVaR (95%): {cvar_95*100:.2f}%")
    print(f"  偏度: {skewness:.2f}")
    print(f"  峰度: {kurtosis:.2f}")
    
    print("\n⚠️ 风险提示:")
    print("  1. 历史回测不代表未来表现")
    print("  2. 策略在不同市场环境下表现差异较大")
    print("  3. 需要持续监控因子有效性衰减")
    print("  4. 实盘交易需考虑滑点、手续费、冲击成本")
    print("  5. 策略容量有限，大资金需要分散投资")
    
    print("\n💡 建议:")
    print("  1. 先进行模拟交易验证（3-6个月）")
    print("  2. 从小资金开始，逐步增加仓位")
    print("  3. 定期监控因子IC和策略表现")
    print("  4. 每季度重新训练ML模型")
    print("  5. 设置止损线（如-15%）")

# 保存评估结果
evaluation_results = {
    "core_metrics": {
        "annual_return": round(strategy_result['annual_return']*100, 2),
        "max_drawdown": round(strategy_result['max_drawdown']*100, 2),
        "sharpe": round(strategy_result['sharpe'], 2),
        "calmar": round(strategy_result['calmar'], 2),
        "win_rate": round(strategy_result['win_rate']*100, 1)
    },
    "risk_metrics": {
        "annual_vol": round(strategy_result['annual_vol']*100, 2),
        "var_95": round(var_95*100, 2),
        "cvar_95": round(cvar_95*100, 2),
        "skewness": round(skewness, 2),
        "kurtosis": round(kurtosis, 2)
    },
    "market_env_performance": {},
    "stress_test": {},
    "trading_cost": {
        "commission_rate": commission_rate,
        "slippage_rate": slippage_rate,
        "annual_cost": round(annual_cost*100, 2),
        "net_annual_return": round(net_annual_return*100, 2),
        "net_sharpe": round(net_sharpe, 2)
    }
}

with open(os.path.expanduser("~/quant_strategies/individual_stock_strategy/data/strategy_evaluation.json"), "w") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print("\n✅ 策略全面评估完成！结果已保存")
