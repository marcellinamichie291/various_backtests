import pandas as pd
import keys
from binance import Client
from pathlib import Path
from decimal import Decimal, getcontext
# from pushbullet import Pushbullet
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import itertools as it
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import statistics as stats
import time
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 4)

df = pd.read_pickle('results.pkl')
# df_v = pd.read_pickle('vwma_results.pkl')
# df_c = pd.read_pickle('close_results.pkl')
# df_a = pd.read_pickle('results_all_tfs.pkl')
# df = pd.concat([df_v, df_c, df_a])


def calc_pnl(pnls, risk_pct):
    """takes a list of trade PnLs denominated in R, and calculates the cumulative account balance from them"""
    start_bal = bal = 100
    bal_evo = [bal]
    for i in pnls:
        # print(bal)
        pnl_factor = 1 + (risk_pct * i / 100)
        bal = bal * pnl_factor
        bal_evo.append(bal)
    # print(f"final bal: {bal:.3f}, pnl = {(bal / start_bal) - 1:.1%}")
    return bal_evo


def lin_returns(series):
    """calculates percentage change of strategy balance from period to period"""

    return series.pct_change().fillna(0)


def log_returns(df, column):
    """
    Calculates the log returns for a given column in a dataframe.
    :param df: The dataframe containing the column.
    :param column: The column to be used.
    :return: The series with the log returns column.
    """
    return np.log(df[column] / df[column].shift(1, fill_value=0)).fillna(0)


def max_drawdown(series):
    """takes the cumulative balance series of the strategy and calculates the largest percentage drawdown that occurs"""
    cum_max = series.cummax()
    dd = 1 - (series/cum_max)

    return dd.max()


def sharpe_ratio(series):
    """
    Calculates the Sharpe ratio for a given column in a dataframe.
    :param series: The column to be used.
    :return: The Sharpe ratio.
    """
    expected_ret = series.reset_index(drop=True).drop(0).mean() * 365
    volatility = series.reset_index(drop=True).drop(0).std() * np.sqrt(365)

    return expected_ret/volatility


def sortino_ratio(series):
    """
    Calculates the Sortino ratio for a given column in a dataframe.
    :param series: The column to be used.
    :return: The Sortino ratio.
    """
    expected_ret = series.reset_index(drop=True).drop(0).mean() * 365
    volatility = series[series<0].reset_index(drop=True).drop(0).std() * np.sqrt(365)

    return expected_ret/volatility


def calmar_ratio(series, dd):
    return series.reset_index(drop=True).drop(0).mean() * 365 / dd


def get_sqn(pnls_r):
    """takes the average PnL (denominated in R) multiplied by the square root of the number of trades (capped at 100)
    and divides that by the standard deviation of the PnLs. The idea is that higher average PnL and larger number of
    trades in the backtest are both good and should increase the score, but higher standard deviation of PnLs is a sign
    of an unreliable system and should therefore decrease the score"""
    r_exp = stats.mean(pnls_r)
    r_std = stats.stdev(pnls_r)
    scalar = min(100, len(pnls_r)) ** 0.5

    return scalar * r_exp / r_std


def get_modified_sqn(pnls_r):
    """same calculation as standard sqn except all winning trades > 1R are capped at 1R in the stdev calculation.
    since stdev is inversely proportional to sqn score, lots of big wins will lower the score"""
    mod_pnls = [min(1, r) for r in pnls_r] # clipping r values at 1 so stdev doesnt penalise big wins
    r_exp = stats.mean(pnls_r)
    r_std = stats.stdev(mod_pnls)
    scalar = min(100, len(pnls_r)) ** 0.5

    return scalar * r_exp / r_std


def get_win_rate(pnls_r):
    wins = [r for r in pnls_r if r > 0]
    return len(wins) / len(pnls_r)


def avg_win_avg_loss(pnls_r):
    pass


def modified_winrate(pnls_r):
    pass

def convert_tf(tf):
    timeframes = {
        '1min': '1m',
        '5min': '5m',
        '15min': '15m',
        '30min': '30m',
        '1h': '1h',
        '4h': '4h',
        '12h': '12h',
    }
    return timeframes[tf]

res_dict = {}
res_count = 0
for row in df.itertuples():
    if len(row.all_rs) > 50:
        all_rs = row.all_rs
        all_bals = calc_pnl(all_rs, 1)
        all_rs = [0] + all_rs
        df_2 = pd.DataFrame({'r_series': all_rs, 'bal_series': all_bals})
        df_2['log_pnls'] = log_returns(df_2, 'bal_series')
        sharpe = sharpe_ratio(df_2.log_pnls)
        sortino = sortino_ratio(df_2.log_pnls)
        max_dd = max_drawdown(pd.Series(all_bals))
        calmar = calmar_ratio(df_2.log_pnls, max_dd)
        sqn = get_sqn(all_rs)
        sqn2 = get_modified_sqn(all_rs)
        win_rate = get_win_rate(all_rs)
        res_dict[res_count] = {'pair': row.pair,
                               'tf': convert_tf(row.timeframe),
                               'bias_lb': row.ema_window,
                               'bias_roc_lb': row.lookback,
                               'source': row.source,
                               'bars': row.bars,
                               'mult': row.mult,
                               'z': row.z_score,
                               'width': row.width,
                               'trades': len(all_rs),
                               'pnl_pct': ((all_bals[-1] / all_bals[0]) - 1) * 100,
                               'sharpe': sharpe,
                               'sortino': sortino,
                               'max_dd': max_dd,
                               'calmar': calmar,
                               'sqn': sqn,
                               'sqn_modified': sqn2,
                               'winrate': win_rate}
        res_count += 1

res_df = pd.DataFrame.from_dict(res_dict, orient='index')



res_df['score'] = (res_df.pnl_pct.rank() +
                   res_df.sharpe.rank() +
                   res_df.sortino.rank() +
                   res_df.max_dd.rank(ascending=False) +
                   res_df.calmar.rank() +
                   res_df.sqn.rank() +
                   res_df.sqn_modified.rank() +
                   res_df.winrate.rank()
                   / 8)
# print(res_df.sort_values('score', ascending=False).reset_index(drop=True).head(100))
settings = (res_df.sort_values('score', ascending=False).reset_index(drop=True).head(20)
.loc[:, ['pair', 'tf', 'bias_lb', 'bias_roc_lb', 'source', 'bars', 'mult', 'z', 'width']])
print(settings)
settings.to_pickle('settings.pkl')
