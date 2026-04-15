import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from pathlib import Path
from bs_ops import single_stock_data,get_trading_days
def align_stock_to_calendar(df_stock, calendar):
    # 1. 确保 df_stock 的 index 是 DatetimeIndex
    if not isinstance(df_stock.index, pd.DatetimeIndex):
        df_stock.index = pd.to_datetime(df_stock.index)

    # 2. 确保 calendar 是 DatetimeIndex
    if not isinstance(calendar, pd.DatetimeIndex):
        calendar = pd.to_datetime(calendar)

    # 3. 去重（防止重复日期）
    df_stock = df_stock[~df_stock.index.duplicated(keep='first')]
    # df_stock.to_csv('stock_data.csv')
    # 4. 重新索引到统一日历
    df_aligned = df_stock.reindex(calendar)
    # df_aligned.to_csv('aligned.csv', index=False)

    # 5. 填充：价格类前向填充（限5天），成交量填0
    # price_cols = ['open', 'high', 'low', 'close']
    # if all(col in df_aligned.columns for col in price_cols):
        # df_aligned[price_cols] = df_aligned[price_cols].fillna(method="ffill", limit=5)

    for col in df_stock.columns:    #['date','code']:
        if col in ['date','code']:
            continue
        elif col in ['volume','turn']:         # 成交量数据填充0, 没有交易
            df_aligned[col] = df_aligned[col].fillna(0)
        else:      # 价格数据向前填充，价格无变化,limit放大些，考虑春节休市时间，免得麻烦
            df_aligned[col] = df_aligned[col].ffill(limit=20)
    # df_aligned.to_csv('aligned2.csv', index=False)
    # if 'volume' in df_aligned.columns:
        # df_aligned['volume'] = df_aligned['volume'].fillna(0)

    return df_aligned

class EasyProfitScreener:
    """
    从指定指数成分股中筛选未来5-20天“容易盈利”的股票
    输入: 日线数据 (含 OHLCV, turnover, peTTM, psTTM, pcfNcfTTM, pbMRQ, roe)
    输出: 每个指数内 Composite Score Top N 的股票
    """
    
    def __init__(self, 
                 index_components,
                 benchmark_returns,
                 lookback_days=120,
                 top_n=10):
        """
        :param index_components: dict, {'CSI300': [list of stocks], 'CSI500': [...]}
        :param benchmark_returns: pd.Series, index=date, value=指数日收益率 (用于相对强度)
        :param lookback_days: 计算因子所需的历史天数
        :param top_n: 每个池子选出的股票数量
        """
        self.index_components = index_components
        self.benchmark_returns = benchmark_returns
        self.lookback_days = lookback_days
        self.top_n = top_n
        
        # 定义要计算的因子
        self.factor_names = [
            'momentum_20',      # 动量
            'reversal_5',       # 反转
            'low_volatility_30',# 波动稳定性 (取负，所以叫 low_vol)
            'volume_price_ratio', # 量价配合
            'relative_strength' # 相对强度
        ]
    
    def calculate_factors(self, df_stock):
        """为单只股票计算所有因子"""
        df = df_stock.copy().sort_index()  # 确保按日期排序
        if len(df) < self.lookback_days:
            return pd.Series(index=self.factor_names, dtype=float)
        
        # --- 1. 动量因子: 过去20日累计收益 ---
        returns = df['close'].pct_change()
        mom_20 = returns.rolling(20).sum().iloc[-1]
        
        # --- 2. 反转因子: 近5日跌幅最大但未破位 ---
        # 我们用 -max_drawdown_5 来表示"超跌程度"
        price_5d = df['close'].iloc[-5:]
        max_dd_5 = (price_5d.min() / price_5d.iloc[0]) - 1
        reversal_5 = -max_dd_5  # 越负越好，所以取负
        
        # --- 3. 波动稳定性: 近30日波动率越低越好 ---
        vol_30 = returns.rolling(30).std().iloc[-1]
        low_vol_30 = -vol_30  # 越低越好，取负
        
        # --- 4. 量价配合: 上涨日成交量 vs 下跌日 ---
        recent_ret = returns.iloc[-5:]
        recent_vol = df['volume'].iloc[-5:]
        up_days = recent_ret > 0
        down_days = recent_ret < 0
        
        if up_days.sum() == 0 or down_days.sum() == 0:
            vol_ratio = 0.0
        else:
            up_vol_avg = recent_vol[up_days].mean()
            down_vol_avg = recent_vol[down_days].mean()
            vol_ratio = up_vol_avg / (down_vol_avg + 1e-6)
        volume_price_ratio = vol_ratio
        
        # --- 5. 相对强度: 相对于大盘的超额收益 ---
        # 需要外部传入 benchmark_returns
        stock_return_20 = df['close'].iloc[-1] / df['close'].iloc[-21] - 1
        bench_return_20 = self._get_benchmark_return(df.index[-1])
        relative_strength = stock_return_20 - bench_return_20
        
        return pd.Series({
            'momentum_20': mom_20,
            'reversal_5': reversal_5,
            'low_volatility_30': low_vol_30,
            'volume_price_ratio': volume_price_ratio,
            'relative_strength': relative_strength
        })
    
    def _get_benchmark_return(self, date):
        """获取指定日期对应的20日指数收益"""
        try:
            end_idx = self.benchmark_returns.index.get_loc(date)
            start_idx = end_idx - 20
            if start_idx < 0:
                return 0.0
            cum_return = (1 + self.benchmark_returns.iloc[start_idx:end_idx]).prod() - 1
            return cum_return
        except KeyError:
            return 0.0
    
    def cross_sectional_zscore(self, factor_df):
        """对每个因子做横截面Z-Score标准化"""
        zscore_df = factor_df.copy()
        for col in self.factor_names:
            mean_val = factor_df[col].mean()
            std_val = factor_df[col].std()
            if std_val > 1e-8:
                zscore_df[col] = (factor_df[col] - mean_val) / std_val
            else:
                zscore_df[col] = 0.0
        return zscore_df
    
    def screen(self, all_stocks_data, target_date):
        """
        主筛选函数
        :param all_stocks_data: dict, {stock_code: pd.DataFrame (daily data)}
        :param target_date: str, 'YYYY-MM-DD', 筛选该日期的数据
        :return: dict, {'CSI300': [top stocks], 'CSI500': [top stocks]}
        """
        results = {}
        
        for index_name, stock_list in self.index_components.items():
            print(f"\n🔍 正在处理 {index_name} 股票池...")
            
            # 收集该指数内所有股票的因子
            factor_records = []
            valid_stocks = []
            
            for stock in stock_list:
                if stock not in all_stocks_data:
                    continue
                df = all_stocks_data[stock]
                if target_date not in df.index:
                    continue
                
                # 获取截至 target_date 的数据
                df_until = df.loc[:target_date]
                if len(df_until) < self.lookback_days:
                    continue
                
                factors = self.calculate_factors(df_until)
                if factors.isna().all():
                    continue
                    
                factor_records.append(factors)
                valid_stocks.append(stock)
            
            if not factor_records:
                print(f"  ⚠️  {index_name} 无有效数据")
                results[index_name] = []
                continue
            
            # 构建因子DataFrame
            factor_df = pd.DataFrame(factor_records, index=valid_stocks)
            
            # 横截面Z-Score标准化
            zscore_df = self.cross_sectional_zscore(factor_df)
            
            # 等权合成综合得分
            zscore_df['composite_score'] = zscore_df[self.factor_names].mean(axis=1)
            
            # 选出Top N
            top_stocks = zscore_df.nlargest(self.top_n, 'composite_score')
            results[index_name] = top_stocks.index.tolist()
            
            # print(f"  ✅ {index_name} Top {self.top_n}:")
            for i, stock in enumerate(results[index_name], 1):
                score = zscore_df.loc[stock, 'composite_score']
                print(f"    {i}. {stock} (Score: {score:.4f})")
        
        return results


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # --- 1. 准备数据 (你需要替换成你的真实数据) ---
    
    csi300 = pd.read_csv('csi300_list.csv')
    csi300_list = csi300['code'].to_list()
    csi500 = pd.read_csv('csi500_list.csv')
    csi500_list = csi500['code'].to_list()

    # 示例：指数成分股
    index_components = {
        # 'CSI300': csi300_list,          #['000001.SZ', '600000.SH', '601318.SH', ...],  # 你的沪深300成分股列表
        'CSI500': csi500_list           #['002475.SZ', '300750.SZ', '688981.SH', ...]   # 你的中证500成分股列表
    }
    
    today = datetime.now().strftime('%Y-%m-%d')

    # 示例：指数日收益率 (你需要计算)
    # benchmark_returns = pd.read_csv('csi300_daily_return.csv', index_col='date', parse_dates=True)['return']
    # 假设我们有一个包含所有日期的Series
    dates = pd.date_range(start='2020-01-01', end=today, freq='D')
    # 这里用随机数据代替，实际请用真实指数收益率
    np.random.seed(42)
    
    csi500 = single_stock_data("sh.000905", '2020-01-01', today)
    csi500.set_index(csi500['date'], inplace=True)
    # benchmark_returns_csi300 = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)
    benchmark_returns_csi500 = csi500['pctChg']
    
    # 示例：所有股票的日线数据
    # all_stocks_data = {
    #     '000001.SZ': pd.read_csv('000001.csv', index_col='date', parse_dates=True),
    #     '600000.SH': pd.read_csv('600000.csv', index_col='date', parse_dates=True),
    #     ...
    # }
    # 为演示，我们创建模拟数据
    all_stocks_data = {}
#    for stock in index_components['CSI300'] + index_components['CSI500']:
#        n_days = 200
#        close = 10 * np.cumprod(1 + np.random.normal(0, 0.02, n_days))
#        df = pd.DataFrame({
#            'open': close * (1 + np.random.normal(0, 0.005, n_days)),
#            'high': close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
#            'low': close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
#            'close': close,
#            'volume': np.random.randint(1e6, 1e8, n_days),
#            'turnover': np.random.uniform(0.5, 5.0, n_days),
#            'peTTM': np.random.uniform(10, 50, n_days),
#            'psTTM': np.random.uniform(1, 10, n_days),
#            'pcfNcfTTM': np.random.uniform(5, 30, n_days),
#            'pbMRQ': np.random.uniform(0.8, 5.0, n_days),
#            'roe': np.random.uniform(0.05, 0.25, n_days)
#        }, index=pd.date_range(end='2026-04-11', periods=n_days, freq='D'))
#        all_stocks_data[stock] = df

    calendar = get_trading_days(start_date='2020-01-01', end_date=today)
    df_trading_days = calendar[calendar['calendar_date'] >= "2020-01-01"].copy()
    print(df_trading_days.head())
    d = df_trading_days.loc[df_trading_days['is_trading_day']=="1"]['calendar_date']
    print(d.head())
    all_trading_days = pd.to_datetime(d).sort_values().unique()
    
    for stock in index_components['CSI500']:
        n_days = 200
        csv = Path("working") / f"{stock}.csv"
        df = pd.read_csv(csv, index_col='date', parse_dates=True)
        # 这里要做填充，把停牌等非交易的日期填充为前一个交易日的收盘价，参照交易所日历
        df_aligned = align_stock_to_calendar(df, all_trading_days)
        all_stocks_data[stock] = df_aligned
    
    # --- 2. 初始化筛选器 ---
    screener = EasyProfitScreener(
        index_components=index_components,
        benchmark_returns=benchmark_returns_csi500,
        lookback_days=120,
        top_n=10
    )
    
    # --- 3. 执行筛选 (假设今天是2026-04-11，为下周选股) ---
    target_date = '2026-02-27'
    top_picks = screener.screen(all_stocks_data, target_date)
    
    # --- 4. 输出结果 ---
    print("\n" + "="*50)
    print("🎯 最终选股结果 (供下周参考)")
    print("="*50)
    for index, stocks in top_picks.items():
        print(f"\n{index} Top 10:")
        for stock in stocks:
            print(f"  - {stock}")