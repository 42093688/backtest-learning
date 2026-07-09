# 📈 Backtest Learning

曾老师量化回测教学项目——量化投研。

## 课程目录

| 课次                | 内容                                                                                       | 知识点                                                           |
| ------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Lesson 1**  | [双均线金叉死叉策略](lesson1/lesson01_sma_crossover.py)                                     | backtrader 入门；模拟数据生成；均线信号回测                      |
| **Lesson 2**  | [基本面选股策略](lesson2/lesson02_fundamental_strategy.py)                                  | ROE/PE/负债率因子；止损止盈；仓位控制                            |
| **Lesson 3**  | [多次随机回测](lesson3/lesson02_fundamental_strategy_multirun.py)                           | 多轮回测；稳定性分析；结果汇总                                   |
| **Lesson 4**  | [回测结果汇总与可视化](lesson4/lesson04_fundamental_strategy_multirun.py)                   | 收益分布图；CSV/Excel 导出；网格参数搜索                         |
| **Lesson 5**  | [研究报告风格输出](lesson5/lesson05_fundamental_strategy_multirun.py)                       | Markdown 报告；参数敏感性分析                                    |
| **Lesson 6**  | [研究级回测框架](lesson6/lesson06_research_style_backtest.py)                               | 基准对比（buy-and-hold）；Sharpe/Sortino/最大回撤；训练集/测试集 |
| **Lesson 7**  | [Walk-Forward 验证](lesson7/lesson07_walkforward_significance.py)                           | 滚动窗口回测；t 检验统计显著性                                   |
| **Lesson 8**  | [真实交易环境模拟](lesson8/lesson08_realistic_trading_environment.py)                       | 佣金、滑点、停牌、价格缺口                                       |
| **Lesson 9**  | [多因子策略](lesson9/lesson09_multifactor_strategy.py)                                      | ROE/PE/负债率/动量因子组合；加权评分                             |
| **Lesson 10** | [因子标准化与权重优化](lesson10/lesson10_factor_standardization_and_weight_optimization.py) | Z-score 标准化；权重搜索                                         |
| **Lesson 11** | [因子 IC 分析](lesson11/lesson11_factor_ic_analysis.py)                                     | Information Coefficient；因子有效性评估                          |
| **Lesson 12** | [机器学习信号](lesson12/lesson12_machine_learning_signal.py)                                | 随机森林分类；模型生成交易信号                                   |
| **Lesson 13** | [时间序列预测](lesson13/lesson13_time_series_forecasting.py)                                | ARIMA 建模；预测转交易信号                                       |
| **Lesson 14** | [稳健时间序列与波动率](lesson14/lesson14_robust_time_series_and_volatility.py)              | 波动率过滤；降低噪声信号                                         |
| **Lesson 15** | [EasyTrader 实盘对接](lesson15-easytrader/)                                                 | 自动交易接入                                                     |

## 学习路线

```
入门 → 基础策略 → 多因子 → 机器学习 → 实盘
  1     2~5      9~11    12~14    15
```

- **Lesson 1~2**：认识回测，搭建基础策略
- **Lesson 3~5**：让回测结果更可信、更易展示
- **Lesson 6~8**：研究级回测框架与真实环境模拟
- **Lesson 9~11**：多因子分析与因子有效性
- **Lesson 12~14**：机器学习与时间序列入门
- **Lesson 15**：对接实盘交易

## 运行环境

- Python 3.8+
- 主要依赖：`backtrader`、`pandas`、`numpy`、`scipy 、scikit-learn、matplotlib`\......

## 快速开始

```bash
# 运行 Lesson 1（双均线策略回测）
cd lesson1 && python3 lesson01_sma_crossover.py

# 运行 Lesson 2（基本面选股策略）
cd lesson2 && python3 lesson02_fundamental_strategy.py
```

## 项目结构

```
backtest-learning/
├── README.md
├── git_commands_guide.txt    # Git 操作指南
├── lessons_1_to_14_summary.md  # 课程总结
├── lesson1/                  # 双均线金叉死叉策略
├── lesson2/                  # 基本面选股策略
├── lesson3/                  # 多次随机回测
├── ...                       # 更多课程
└── lesson15-easytrader/      # EasyTrader 实盘对接
```

---

> 持续更新中…
