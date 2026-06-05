# A股个股多因子量化策略

## 策略表现
- 年化收益: 34.83%
- 最大回撤: -16.78%
- 夏普比率: 1.43
- Calmar比率: 2.08
- 胜率: 67.2%

## 策略组成
1. ML因子: GradientBoosting机器学习模型
2. LLM因子: 流动性情绪指标
3. arXiv因子: 学术论文公式因子
4. 三因子组合: 等权重配置策略

## 使用方法
```bash
# 计算因子
python scripts/llm_factor_part1.py
python scripts/arxiv_factor_part1.py

# 运行回测
python scripts/llm_factor_part2.py
python scripts/arxiv_factor_part2.py

# 策略评估
python scripts/strategy_evaluation.py

# 获取建仓建议
python scripts/today_advice_10w.py
```
