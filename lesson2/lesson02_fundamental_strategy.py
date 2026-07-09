"""
======================================================================
第2课：用财务指标（ROE、PE）做选股策略
======================================================================
知识点：
  1. 在股价数据中附加财务指标（ROE、PE、负债率）
  2. 用“基本面 + 风险控制”的方式做买卖决策
  3. 了解什么样的回测才更接近正式量化研究

本课目标：用财务指标（ROE、PE）做选股策略。
======================================================================
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lesson2"
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("第1步：生成带有财务指标的模拟数据")
print("=" * 60)

np.random.seed(42)
days = 500

# ---- (A) 生成股价 ----
start_price = 100.0
daily_returns = np.random.normal(0.0005, 0.02, days)
prices = start_price * np.cumprod(1 + daily_returns)

# ---- (B) 生成财务指标 ----
# 这里模拟两个公司的基本面差异：
# 好公司：ROE高、PE适中、负债率低
# 差公司：ROE低、PE高、负债率高
roe_good = np.random.uniform(0.15, 0.25, days)
roe_bad = np.random.uniform(0.00, 0.08, days)

# 让好公司的股价表现更强，差公司表现更弱
boost_good = np.cumprod(1 + np.random.normal(0.0005, 0.004, days))
penalty_bad = np.cumprod(1 + np.random.normal(-0.0003, 0.004, days))
prices_good = prices * boost_good
prices_bad = prices * penalty_bad

pe_good = np.random.normal(12, 3, days).clip(5, 30)
pe_bad = np.random.normal(50, 20, days).clip(15, 200)
debt_good = np.random.uniform(0.2, 0.4, days)
debt_bad = np.random.uniform(0.5, 0.8, days)


def build_company_data(prices_arr, roe_arr, pe_arr, debt_arr):
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=days, freq='B'),
        'open': prices_arr * np.random.uniform(0.98, 1.02, days),
        'high': prices_arr * np.random.uniform(1.01, 1.05, days),
        'low': prices_arr * np.random.uniform(0.95, 0.99, days),
        'close': prices_arr,
        'volume': np.random.randint(1000000, 10000000, days),
        'roe': roe_arr,
        'pe': pe_arr,
        'debt_ratio': debt_arr,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    return df


df_good = build_company_data(prices_good, roe_good, pe_good, debt_good)
df_bad = build_company_data(prices_bad, roe_bad, pe_bad, debt_bad)

# 保存为 CSV
df_good.to_csv(DATA_DIR / 'company_good.csv', index=False)
df_bad.to_csv(DATA_DIR / 'company_bad.csv', index=False)

print("已生成两个公司的模拟数据：")
print("好公司: ROE 在 15%-25%，PE 在 10-15，负债率 20%-40%")
print("差公司: ROE 在 0%-8%，PE 很高，负债率 50%-80%")
print()

# 对比两个公司的股价走势
plt.figure(figsize=(14, 8))
plt.subplot(2, 2, 1)
plt.plot(df_good['date'], df_good['close'], label='好公司', color='green', linewidth=2)
plt.plot(df_bad['date'], df_bad['close'], label='差公司', color='red', linewidth=2)
plt.title('股价走势对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(df_good['date'], df_good['roe'], label='好公司 ROE', color='green')
plt.plot(df_bad['date'], df_bad['roe'], label='差公司 ROE', color='red')
plt.axhline(y=0.15, color='gray', linestyle='--', label='ROE 15% 基准线')
plt.title('ROE 对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(df_good['date'], df_good['pe'], label='好公司 PE', color='green')
plt.plot(df_bad['date'], df_bad['pe'], label='差公司 PE', color='red')
plt.axhline(y=15, color='gray', linestyle='--', label='PE 15 基准线')
plt.title('PE 对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
plt.plot(df_good['date'], df_good['debt_ratio'], label='好公司负债率', color='green')
plt.plot(df_bad['date'], df_bad['debt_ratio'], label='差公司负债率', color='red')
plt.title('负债率对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'fundamental_data.png', dpi=150)
print("财务指标对比图已保存")
print()

# ============================================================
# 第2步：定义回测策略
# ============================================================

print("=" * 60)
print("第2步：编写回测的策略")
print("=" * 60)


class FundamentalStrategy(bt.Strategy):
    """
    回测的基本面策略：
    - 信号使用上一期财务数据（避免未来函数）
    - 采用仓位控制和止损/止盈
    - 使用市场订单，并考虑手续费与滑点
    """

    params = (
        ('roe_buy', 0.15),
        ('pe_buy', 15),
        ('debt_buy', 0.50),
        ('roe_sell', 0.10),
        ('pe_sell', 30),
        ('debt_sell', 0.60),
        ('max_position_pct', 0.20),
        ('stop_loss_pct', 0.08),
        ('take_profit_pct', 0.12),
    )

    def __init__(self):
        self.roe = self.data.roe
        self.pe = self.data.pe
        self.debt = self.data.debt_ratio
        self.order = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print(f"\n   买入条件：ROE > {self.params.roe_buy * 100:.0f}% 且 "
              f"PE < {self.params.pe_buy} 且 负债率 < {self.params.debt_buy * 100:.0f}%")
        print(f"   卖出条件：ROE < {self.params.roe_sell * 100:.0f}% 或 "
              f"PE > {self.params.pe_sell} 或 负债率 > {self.params.debt_sell * 100:.0f}%")
        print(f"   风险控制：最多持仓 {self.params.max_position_pct * 100:.0f}%，止损 {self.params.stop_loss_pct * 100:.0f}%，止盈 {self.params.take_profit_pct * 100:.0f}%")
        print()

    def next(self):
        # 使用上一期的财务指标，避免使用未来数据
        current_roe = self.roe[-1]
        current_pe = self.pe[-1]
        current_debt = self.debt[-1]

        if self.order:
            return

        if not self.position:
            buy_signal = (
                current_roe > self.params.roe_buy and
                current_pe < self.params.pe_buy and
                current_debt < self.params.debt_buy
            )

            if buy_signal:
                price = self.data.close[0]
                if price > 0:
                    cash = self.broker.getcash()
                    max_investment = cash * self.params.max_position_pct
                    size = int(max_investment / price / 100) * 100
                    if size >= 100:
                        self.order = self.buy(size=size)
                        self.entry_price = price
                        self.stop_price = price * (1 - self.params.stop_loss_pct)
                        self.target_price = price * (1 + self.params.take_profit_pct)
                        print(f"{self.data.datetime.date(0)} 买入，规模={size} 股")

        else:
            current_price = self.data.close[0]
            sell_signal = (
                current_roe < self.params.roe_sell or
                current_pe > self.params.pe_sell or
                current_debt > self.params.debt_sell or
                current_price <= self.stop_price or
                current_price >= self.target_price
            )

            if sell_signal:
                self.order = self.sell(size=self.position.size)
                print(f"{self.data.datetime.date(0)} 卖出，价格={current_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"{self.data.datetime.date(0)} 订单完成：买入 {order.executed.size} 股 价格={order.executed.price:.2f}")
            elif order.issell():
                print(f"{self.data.datetime.date(0)} 订单完成：卖出 {order.executed.size} 股 价格={order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"订单未成交：{order.Status[order.status]}")
        self.order = None


# ============================================================
# 第3步：自定义 Data Feed——让 backtrader 认识财务指标
# ============================================================

class PandasDataFundamental(bt.feeds.PandasData):
    """扩展的 Data Feed，增加财务指标列"""
    lines = ('roe', 'pe', 'debt_ratio')
    params = (
        ('roe', 'roe'),
        ('pe', 'pe'),
        ('debt_ratio', 'debt_ratio'),
    )


# ============================================================
# 第4步：对好公司和差公司分别做回测
# ============================================================

def run_backtest(df, title, initial_cash=100000):
    """对给定的数据运行回测"""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=20)

    data_feed = PandasDataFundamental(
        dataname=df,
        datetime='date',
        open='open', high='high', low='low', close='close', volume='volume',
    )
    cerebro.adddata(data_feed, name=title)

    cerebro.addstrategy(FundamentalStrategy)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.00025)
    cerebro.broker.set_slippage_perc(0.0005)

    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"\n{'=' * 50}")
    print(f"回测：{title}")
    print(f"{'=' * 50}")
    print(f"初始资金: {initial_cash:,.2f} 元")

    results = cerebro.run()
    result = results[0]

    final_value = cerebro.broker.getvalue()
    profit = final_value - initial_cash
    profit_pct = (profit / initial_cash) * 100

    returns_analysis = result.analyzers.returns.get_analysis()
    drawdown_analysis = result.analyzers.drawdown.get_analysis()
    trades_analysis = result.analyzers.trades.get_analysis()

    print(f"\n{'─' * 40}")
    print(f"最终资产: {final_value:,.2f} 元")
    print(f"总收益:   {profit:+,.2f} 元 ({profit_pct:+.2f}%)")
    print(f"最大回撤:  {drawdown_analysis.get('drawdown', 0):.2f}%")
    if returns_analysis.get('rnorm100', None) is not None:
        print(f"年化收益:  {returns_analysis['rnorm100']:.2f}%")
    total_trades = trades_analysis.get('total', {}).get('total', 0)
    print(f"交易次数:  {total_trades} 次")
    if total_trades > 0:
        won = trades_analysis.get('won', {}).get('total', 0)
        lost = trades_analysis.get('lost', {}).get('total', 0)
        win_rate = won / (won + lost) * 100 if (won + lost) > 0 else 0
        print(f"胜率:     {win_rate:.1f}%")
    print(f"{'─' * 40}\n")

    return cerebro


# ============================================================
# 第5步：分别回测
# ============================================================

print("\n" + "=" * 60)
print("第3步：分别回测好公司和差公司")
print("=" * 60)

run_backtest(df_good, "好公司 (ROE高+PE低+负债率低)")
run_backtest(df_bad, "差公司 (ROE低+PE高+负债率高)")

# ============================================================
# 第6步：总结对比
# ============================================================

print("=" * 60)
print("最终对比总结")
print("=" * 60)
print()
print("好公司（基本面优秀） vs  差公司（基本面差）")
print(f"{'─' * 40}")
print(" 好公司策略：更容易在基本面改善时获得正收益")
print(" 差公司策略：即使规则看起来合理，仍会受财务恶化和高估值拖累")
print()
print("关键结论：")
print("  1. 基本面筛选可以提高选股质量，但不能替代风险控制")
print("  2. 只靠财务指标买卖，仍需止损/止盈和仓位管理")
print()
print("已完成回测")
print("=" * 60)
