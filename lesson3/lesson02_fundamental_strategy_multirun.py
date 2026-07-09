"""
======================================================================
第2课扩展：多次随机回测（观察策略稳定性） + 结果汇总表（便于后续研究分析）
======================================================================
目标：
  1. 策略多次随机回测
  2. 汇总不同随机种子下的结果，观察策略稳定性
  3. 输出 CSV 汇总表，方便后续做研究分析
======================================================================
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lesson2"
DATA_DIR.mkdir(exist_ok=True)

NUM_RUNS = 10
INITIAL_CASH = 100000.0


class PandasDataFundamental(bt.feeds.PandasData):
    lines = ('roe', 'pe', 'debt_ratio')
    params = (
        ('roe', 'roe'),
        ('pe', 'pe'),
        ('debt_ratio', 'debt_ratio'),
    )


class FundamentalStrategy(bt.Strategy):
    """基本面策略，带仓位控制、止损、止盈"""

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

    def next(self):
        # 使用上一期财务数据，避免未来函数
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

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            pass
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
        self.order = None


def build_company_data(days, seed, label):
    """生成模拟公司数据"""
    rng = np.random.default_rng(seed)
    start_price = 100.0
    daily_returns = rng.normal(0.0005, 0.02, days)
    prices = start_price * np.cumprod(1 + daily_returns)

    if label == 'good':
        roe_arr = rng.uniform(0.15, 0.25, days)
        pe_arr = rng.normal(12, 3, days).clip(5, 30)
        debt_arr = rng.uniform(0.2, 0.4, days)
        boost = np.cumprod(1 + rng.normal(0.0005, 0.004, days))
        prices_arr = prices * boost
    else:
        roe_arr = rng.uniform(0.00, 0.08, days)
        pe_arr = rng.normal(50, 20, days).clip(15, 200)
        debt_arr = rng.uniform(0.5, 0.8, days)
        penalty = np.cumprod(1 + rng.normal(-0.0003, 0.004, days))
        prices_arr = prices * penalty

    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=days, freq='B'),
        'open': prices_arr * rng.uniform(0.98, 1.02, days),
        'high': prices_arr * rng.uniform(1.01, 1.05, days),
        'low': prices_arr * rng.uniform(0.95, 0.99, days),
        'close': prices_arr,
        'volume': rng.integers(1000000, 10000000, days),
        'roe': roe_arr,
        'pe': pe_arr,
        'debt_ratio': debt_arr,
    })
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    return df


def run_single_backtest(df, title):
    """执行一次回测并返回关键指标"""
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=20)

    data_feed = PandasDataFundamental(
        dataname=df,
        datetime='date',
        open='open', high='high', low='low', close='close', volume='volume',
    )
    cerebro.adddata(data_feed, name=title)

    cerebro.addstrategy(FundamentalStrategy)
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=0.00025)
    cerebro.broker.set_slippage_perc(0.0005)

    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    results = cerebro.run()
    result = results[0]

    final_value = cerebro.broker.getvalue()
    profit = final_value - INITIAL_CASH
    profit_pct = profit / INITIAL_CASH * 100

    returns_analysis = result.analyzers.returns.get_analysis()
    drawdown_analysis = result.analyzers.drawdown.get_analysis()
    trades_analysis = result.analyzers.trades.get_analysis()

    total_trades = trades_analysis.get('total', {}).get('total', 0)
    won = trades_analysis.get('won', {}).get('total', 0)
    lost = trades_analysis.get('lost', {}).get('total', 0)
    win_rate = won / (won + lost) * 100 if (won + lost) > 0 else 0

    return {
        'final_value': final_value,
        'profit': profit,
        'return_pct': profit_pct,
        'max_drawdown': drawdown_analysis.get('drawdown', 0),
        'annualized_return': returns_analysis.get('rnorm100', None),
        'total_trades': total_trades,
        'win_rate': win_rate,
    }


def summarize_results(results_df):
    summary = results_df.groupby('company').agg(
        runs=('seed', 'count'),
        avg_final_value=('final_value', 'mean'),
        std_final_value=('final_value', 'std'),
        avg_return_pct=('return_pct', 'mean'),
        median_return_pct=('return_pct', 'median'),
        min_return_pct=('return_pct', 'min'),
        max_return_pct=('return_pct', 'max'),
        avg_drawdown=('max_drawdown', 'mean'),
        avg_trades=('total_trades', 'mean'),
        avg_win_rate=('win_rate', 'mean'),
    ).reset_index()
    return summary


def print_summary_table(summary_df):
    print("\n" + "=" * 90)
    print("多次随机回测汇总表")
    print("=" * 90)
    print(summary_df.to_string(index=False, float_format=lambda x: f'{x:,.2f}' if abs(x) >= 1 else f'{x:.3f}'))
    print("=" * 90)


if __name__ == '__main__':
    print("=" * 60)
    print("开始多次随机回测")
    print("=" * 60)

    days = 500
    rows = []

    for run_idx in range(NUM_RUNS):
        seed = 1000 + run_idx
        print(f"\n  第 {run_idx + 1}/{NUM_RUNS} 轮，随机种子={seed}")

        df_good = build_company_data(days, seed, 'good')
        df_bad = build_company_data(days, seed + 10000, 'bad')

        good_metrics = run_single_backtest(df_good, f'好公司-Run{run_idx + 1}')
        bad_metrics = run_single_backtest(df_bad, f'差公司-Run{run_idx + 1}')

        for company, metrics in [('good', good_metrics), ('bad', bad_metrics)]:
            row = {'seed': seed if company == 'good' else seed + 10000, 'company': company}
            row.update(metrics)
            rows.append(row)

            print(f"   {company}: 最终资产={metrics['final_value']:,.2f}，收益率={metrics['return_pct']:.2f}%，回撤={metrics['max_drawdown']:.2f}%")

    results_df = pd.DataFrame(rows)
    summary_df = summarize_results(results_df)
    print_summary_table(summary_df)

    out_csv = DATA_DIR / 'multi_run_summary.csv'
    summary_df.to_csv(out_csv, index=False)
    print(f"\n  汇总表已保存到: {out_csv}")

    per_run_csv = DATA_DIR / 'multi_run_results.csv'
    results_df.to_csv(per_run_csv, index=False)
    print(f"  每轮结果已保存到: {per_run_csv}")
