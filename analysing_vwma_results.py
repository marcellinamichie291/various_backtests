import pandas as pd
import keys
from binance import Client
from pathlib import Path
from decimal import Decimal, getcontext
from pushbullet import Pushbullet
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

df = pd.read_pickle('vwma_results.pkl')


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
    r_exp = stats.mean(pnls_r)
    r_std = stats.stdev(pnls_r)
    scalar = len(pnls_r) ** 0.5

    return scalar * r_exp / r_std


def get_win_rate(pnls_r):
    wins = [r for r in pnls_r if r > 0]
    return len(wins) / len(pnls_r)


res_dict = {}
res_count = 0
for row in df.itertuples():
    if len(row.all_rs) > 50:
        all_rs = row.all_rs
        all_bals = calc_pnl(all_rs, 1.5)
        all_rs = [0] + all_rs
        df_2 = pd.DataFrame({'r_series': all_rs, 'bal_series': all_bals})
        df_2['log_pnls'] = log_returns(df_2, 'bal_series')
        sharpe = sharpe_ratio(df_2.log_pnls)
        sortino = sortino_ratio(df_2.log_pnls)
        max_dd = max_drawdown(pd.Series(all_bals))
        calmar = calmar_ratio(df_2.log_pnls, max_dd)
        sqn = get_sqn(all_rs)
        win_rate = get_win_rate(all_rs)
        res_dict[res_count] = {'timeframe': row.timeframe,
                               'z score': row.z_score,
                               'bars': row.bars,
                               'mult': row.mult,
                               'ema': row.ema_window,
                               'lookback': row.lookback,
                               'atr': row.atr,
                               'pnl': (all_bals[-1] / all_bals[0]) - 1,
                               'sharpe': sharpe,
                               'sortino': sortino,
                               'max_dd': max_dd,
                               'calmar': calmar,
                               'sqn': sqn,
                               'winrate': win_rate}
        res_count += 1

res_df = pd.DataFrame.from_dict(res_dict, orient='index')
print(res_df.sort_values('sqn', ascending=False).head(30))
