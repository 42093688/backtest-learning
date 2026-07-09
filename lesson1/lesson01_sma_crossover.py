"""
======================================================================
第1课：双均线金叉死叉策略 — 教学案例：回测入门
======================================================================
知识技能点：
  1. 生成模拟的股票价格数据
  2. 用 backtrader 构建一个简单的策略
  3. 运行回测并查看结果
  4. 理解"赚了多少、最大亏了多少、交易了多少次"

运行方法：
  python3 + pip install backtrader
  python3 lesson01_sma_crossover.py
"""

import os
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt

# ============================================================
# 第1步：生成模拟的股票价格数据
# ============================================================

print("=" * 60)
print("第1步：生成模拟股价数据")
print("=" * 60)

np.random.seed(42)  # 固定随机种子，保证每次运行结果一样

# 参数设置
days = 252  # 生成252个交易日的数据（1年）
start_price = 100.0

# 生成每日收益率（随机游走，年化波动率约20%）
# 日收益率均0.05%，标准差1.26%
daily_returns = np.random.normal(0.0005, 0.0126, days)  

# 从起始价格开始累计
prices = start_price * np.cumprod(1 + daily_returns)

# 构建 OHLCV 数据（开盘、最高、最低、收盘、成交量）
# B = 工作日
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=days, freq='B'), 
    'open': prices * np.random.uniform(0.98, 1.02, days),
    'high': prices * np.random.uniform(1.01, 1.05, days),
    'low': prices * np.random.uniform(0.95, 0.99, days),
    'close': prices,
    'volume': np.random.randint(1000000, 10000000, days)
})

# 修正：确保 high >= open/close, low <= open/close
for i in range(len(data)):
    data.loc[i, 'high'] = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
    data.loc[i, 'low'] = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])

# 保存为 CSV（以后可以直接用真实数据替换这个文件）
csv_path = os.path.expanduser('~/backtest-learning/sample_data.csv')
data.to_csv(csv_path, index=False)
print(f"已生成模拟数据，保存到: {csv_path}")
print(f"   共 {days} 个交易日，起始价 {start_price}，最终价 {prices[-1]:.2f}")
print()

# 画个走势图看看
plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['close'], label='收盘价', color='steelblue')
plt.title('模拟股票价格走势')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/backtest-learning/price_chart.png'))
print("价格走势图已保存: price_chart.png")
print()


# ============================================================
# 第2步：定义策略
# ============================================================
# 在 backtrader 中，策略是一个继承 bt.Strategy 的类
# 我们只需要定义：
#   - __init__()     初始化指标（只算一次）
#   - next()         每天调用一次，决定买还是卖

print("=" * 60)
print("第2步：编写策略 — 双均线金叉死叉")
print("=" * 60)

class SmaCross(bt.Strategy):
    """
    双均线策略（最经典的入门策略）：
    - 计算两条移动平均线：短期（10日均线）和长期（30日均线）
    - 金叉：短期均线上穿长期均线 买入信号
    - 死叉：短期均线下穿长期均线 卖出信号
    """

    # 可调参数（你可以修改这些数字看效果）
    params = (
        ('short_period', 10),   # 短期均线天数
        ('long_period', 30),    # 长期均线天数
    )

    def __init__(self):
        """
        初始化函数：在这里计算指标
        注意：这个函数只会在开始时运行一次
        """
        # 计算两条均线
        # bt.indicators.SMA = Simple Moving Average（简单移动平均）
        self.sma_short = bt.indicators.SMA(
            self.data.close, period=self.params.short_period
        )
        self.sma_long = bt.indicators.SMA(
            self.data.close, period=self.params.long_period
        )

        # 交叉信号（backtrader 自动算"上穿"和"下穿"）
        # bt.indicators.CrossOver 返回：
        #    1  短线上穿长线（金叉，买入信号）
        #   -1  短线下穿长线（死叉，卖出信号）
        #    0  没有交叉
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        self.order = None

        print(f"策略初始化完成：短期均线={self.params.short_period}天，"
              f"长期均线={self.params.long_period}天")

    def next(self):
        """
        核心函数：每个交易日调用一次
        在这里决定：买？卖？还是不动？
        """
        # 如果还有未完成订单，先不再下新单
        if self.order:
            return

        # 如果还没有持仓（手中没有股票）
        if not self.position:
            # crossover == 1 表示金叉，买入
            if self.crossover[0] == 1:
                # self.buy() = 买入，size 参数是买多少股
                # 这里用全部资金买（95%仓位，留手续费），100股的整数倍，并且下取整
                size = int(self.broker.getcash() * 0.95 / self.data.close[0] / 100) * 100
                if size > 0:
                    self.order = self.buy(size=size, exectype=bt.Order.Market)
                    print(f"{self.data.datetime.date(0)} 金叉！下单：次日开盘市价买入 {size}股")

        # 如果手里有股票
        else:
            # crossover == -1 表示死叉 → 卖出
            if self.crossover[0] == -1:
                self.order = self.sell(size=self.position.size, exectype=bt.Order.Market)
                print(f"{self.data.datetime.date(0)} 死叉！下单：次日开盘市价卖出 {self.position.size}股")

    # 订单状态通知监控
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"{self.data.datetime.date(0)} 成交：买入 {order.executed.size}股 @ {order.executed.price:.2f}")
            elif order.issell():
                print(f"{self.data.datetime.date(0)} 成交：卖出 {order.executed.size}股 @ {order.executed.price:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"{self.data.datetime.date(0)} 订单未成交：{order.Status[order.status]}")

        self.order = None


# ============================================================
# 第3步：设置回测引擎并运行
# ============================================================

print()
print("=" * 60)
print("第3步：设置回测引擎并运行")
print("=" * 60)

# 创建回测引擎
cerebro = bt.Cerebro()

# 把数据喂给引擎
# 第1种方式：从 CSV 读（可以用真实数据替换）
data_feed = bt.feeds.PandasData(
    dataname=data,
    datetime='date',   # 日期列名
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
)
cerebro.adddata(data_feed)

# 把策略加入引擎
cerebro.addstrategy(SmaCross)

# 设置初始资金
initial_cash = 1000000.0  # 100万元
cerebro.broker.setcash(initial_cash)

# 设置手续费（A股佣金约为万分之2.5）
cerebro.broker.setcommission(commission=0.00025)

# 添加一个分析器来统计收益
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

print(f"初始资金: {initial_cash:,.2f} 元")
print()

# 运行回测
print("回测中，请稍候...正在回测中...")
results = cerebro.run()

print("回测完成！")
print()

# ============================================================
# 第4步：查看结果
# ============================================================

print("=" * 60)
print("第4步：回测结果分析")
print("=" * 60)

strategy_result = results[0]  # 只有一个策略，所以取 [0]

# 获取分析结果
returns = strategy_result.analyzers.returns.get_analysis()
sharpe = strategy_result.analyzers.sharpe.get_analysis()
drawdown = strategy_result.analyzers.drawdown.get_analysis()
trades = strategy_result.analyzers.trades.get_analysis()

final_value = cerebro.broker.getvalue()
profit = final_value - initial_cash
profit_pct = (profit / initial_cash) * 100

print(f"\n{'─' * 40}")
print(f"  最终资产: {final_value:,.2f} 元")
print(f"  总收益:    {profit:+,.2f} 元  ({profit_pct:+.2f}%)")
print(f"  最大回撤:   {drawdown.get('drawdown', 0):.2f}%")
sharpe_ratio = sharpe.get('sharperatio', None)
print(f"  夏普比率:   {sharpe_ratio:.2f}" if sharpe_ratio is not None else "  夏普比率:   N/A")
print(f"  交易次数:   {trades.get('total', {}).get('total', 0)} 次")
print(f"{'─' * 40}")

# 年化收益率兼容 None
rnorm100 = returns.get('rnorm100', None)
if rnorm100 is not None:
    print(f"年化收益率: {rnorm100:.2f}%")
else:
    print("年化收益率: N/A")
print()

# 交易详情
if trades.get('total', {}).get('total', 0) > 0:
    won = trades.get('won', {}).get('total', 0)
    lost = trades.get('lost', {}).get('total', 0)
    win_rate = won / (won + lost) * 100 if (won + lost) > 0 else 0
    print(f"盈利交易: {won} 次")
    print(f"亏损交易: {lost} 次")
    print(f"胜率:     {win_rate:.1f}%")

    avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
    if avg_loss != 0:
        profit_ratio = abs(avg_win / avg_loss)
        print(f"盈亏比:   {profit_ratio:.2f} (平均赚 {avg_win:.0f} / 平均亏 {avg_loss:.0f})")
print()

# ============================================================
# 第5步：画回测结果图
# ============================================================

print("=" * 60)
print("第5步：生成回测图表")
print("=" * 60)

# backtrader 自带画图功能
fig = cerebro.plot(style='candlestick', volume=True, figsize=(16, 10))
chart_path = os.path.expanduser('~/backtest-learning/backtest_result.png')
# backtrader 的 plot 返回一个列表，里面是 figure 对象
if fig and fig[0]:
    fig[0][0].savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"回测图表已保存: backtest_result.png")

print()
print("=" * 60)
print("Congratulations，完成了第一次回测！")
print("=" * 60)
print(f"回测结果保存在：{chart_path}")
print("=" * 60)