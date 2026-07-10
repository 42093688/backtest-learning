"""
======================================================================
第4课：多次随机回测 + 参数网格搜索 + 策略稳定性（汇总图） + Excel/CSV 输出
======================================================================
目标：
  1. 策略多次随机回测
  2. 输出汇总结果图，直观看出策略稳定性
  3. 输出CSV / Excel 汇总表
  4. 参数网格搜索，找出更优参数组合
======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
LESSON2_DIR = ROOT_DIR / 'lesson2'
LESSON4_DIR = ROOT_DIR / 'lesson4'
LESSON4_DIR.mkdir(exist_ok=True)

NUM_RUNS = 10
INITIAL_CASH = 100000.0
DAYS = 500


class PandasDataFundamental(bt.feeds.PandasData):
    lines = ('roe', 'pe', 'debt_ratio')
    params = (
        ('roe', 'roe'),
        ('pe', 'pe'),
        ('debt_ratio', 'debt_ratio'),
    )


class FundamentalStrategy(bt.Strategy):
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

    def next(self):
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
        self.order = None


def build_company_data(days, seed, label):
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
        roe_arr = rng.uniform(0.0, 0.08, days)
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


def run_single_backtest(df, title, params):
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=20)

    data_feed = PandasDataFundamental(
        dataname=df,
        datetime='date',
        open='open', high='high', low='low', close='close', volume='volume',
    )
    cerebro.adddata(data_feed, name=title)

    cerebro.addstrategy(FundamentalStrategy, **params)
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


def run_multirun_backtest():
    rows = []
    print("=" * 60)
    print("开始多次随机回测")
    print("=" * 60)

    for run_idx in range(NUM_RUNS):
        seed = 1000 + run_idx
        print(f"\n  第 {run_idx + 1}/{NUM_RUNS} 轮，随机种子={seed}")

        df_good = build_company_data(DAYS, seed, 'good')
        df_bad = build_company_data(DAYS, seed + 10000, 'bad')

        default_params = {
            'roe_buy': 0.15,
            'pe_buy': 15,
            'debt_buy': 0.50,
            'roe_sell': 0.10,
            'pe_sell': 30,
            'debt_sell': 0.60,
            'max_position_pct': 0.20,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.12,
        }

        good_metrics = run_single_backtest(df_good, f'好公司-Run{run_idx + 1}', default_params)
        bad_metrics = run_single_backtest(df_bad, f'差公司-Run{run_idx + 1}', default_params)

        for company, metrics in [('good', good_metrics), ('bad', bad_metrics)]:
            row = {'seed': seed if company == 'good' else seed + 10000, 'company': company}
            row.update(metrics)
            rows.append(row)
            print(f"   {company}: 最终资产={metrics['final_value']:,.2f}，收益率={metrics['return_pct']:.2f}%，回撤={metrics['max_drawdown']:.2f}%")

    results_df = pd.DataFrame(rows)
    summary_df = summarize_results(results_df)
    return results_df, summary_df


def summarize_results(results_df):
    return results_df.groupby('company').agg(
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


def plot_summary(results_df, summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    good = results_df[results_df['company'] == 'good']['return_pct']
    bad = results_df[results_df['company'] == 'bad']['return_pct']

    axes[0].bar(['好公司', '差公司'], [good.mean(), bad.mean()], color=['#2ca02c', '#d62728'])
    axes[0].set_title('平均收益率对比')
    axes[0].set_ylabel('收益率 (%)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].boxplot([good, bad], labels=['好公司', '差公司'])
    axes[1].set_title('收益率分布箱线图')
    axes[1].set_ylabel('收益率 (%)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = LESSON4_DIR / 'summary_results_plot.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"汇总结果图已保存: {plot_path}")


def save_tables(results_df, summary_df):
    csv_summary = LESSON4_DIR / 'multi_run_summary.csv'
    csv_results = LESSON4_DIR / 'multi_run_results.csv'
    xlsx_summary = LESSON4_DIR / 'multi_run_summary.xlsx'
    xlsx_results = LESSON4_DIR / 'multi_run_results.xlsx'

    summary_df.to_csv(csv_summary, index=False)
    results_df.to_csv(csv_results, index=False)
    print(f"CSV 汇总表已保存: {csv_summary}")
    print(f"CSV 每轮结果已保存: {csv_results}")

    try:
        summary_df.to_excel(xlsx_summary, index=False, sheet_name='summary')
        results_df.to_excel(xlsx_results, index=False, sheet_name='results')
        print(f"Excel 汇总表已保存: {xlsx_summary}")
        print(f"Excel 每轮结果已保存: {xlsx_results}")
    except Exception as exc:
        print(f"Excel 写入失败，已保留 CSV：{exc}")


def run_parameter_grid_search():
    print("\n" + "=" * 60)
    print("开始策略参数网格搜索")
    print("=" * 60)

    param_grid = []
    for roe_buy in [0.12, 0.15]:
        for pe_buy in [12, 15]:
            for debt_buy in [0.40, 0.50]:
                for roe_sell in [0.08, 0.10]:
                    for pe_sell in [25, 30]:
                        for debt_sell in [0.55, 0.60]:
                            param_grid.append({
                                'roe_buy': roe_buy,
                                'pe_buy': pe_buy,
                                'debt_buy': debt_buy,
                                'roe_sell': roe_sell,
                                'pe_sell': pe_sell,
                                'debt_sell': debt_sell,
                                'max_position_pct': 0.20,
                                'stop_loss_pct': 0.08,
                                'take_profit_pct': 0.12,
                            })

    rows = []
    for idx, params in enumerate(param_grid, start=1):
        df = build_company_data(DAYS, 2000 + idx, 'good')
        metrics = run_single_backtest(df, f'Grid-{idx}', params)
        row = {'combo_id': idx}
        row.update(params)
        row.update(metrics)
        rows.append(row)
        print(f"[{idx}/{len(param_grid)}] params={params} -> 收益率={metrics['return_pct']:.2f}%")

    grid_df = pd.DataFrame(rows)
    grid_df = grid_df.sort_values(by='return_pct', ascending=False)
    grid_df.to_csv(LESSON4_DIR / 'parameter_grid_search.csv', index=False)
    try:
        grid_df.to_excel(LESSON4_DIR / 'parameter_grid_search.xlsx', index=False, sheet_name='grid_search')
    except Exception as exc:
        print(f"参数网格 Excel 写入失败：{exc}")

    print("\n  最优参数组合：")
    print(grid_df[['roe_buy', 'pe_buy', 'debt_buy', 'roe_sell', 'pe_sell', 'debt_sell', 'return_pct', 'max_drawdown', 'total_trades']].head(10).to_string(index=False))
    return grid_df


if __name__ == '__main__':
    results_df, summary_df = run_multirun_backtest()
    plot_summary(results_df, summary_df)
    save_tables(results_df, summary_df)
    run_parameter_grid_search()

    print("\n  所有结果已生成完毕！")
    print(f"   输出目录: {LESSON4_DIR}")
