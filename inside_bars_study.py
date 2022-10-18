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

# CONSTANTS
client = Client(keys.bPkey, keys.bSkey)
# pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12
ohlc_path = Path("/mnt/pi_2/ohlc_binance_1m")
not_pairs = ['GBPUSDT', 'AUDUSDT', 'BUSDUSDT', 'EURUSDT', 'TUSDUSDT',
             'USDCUSDT', 'PAXUSDT', 'COCOSUSDT', 'SUSDUSDT', 'USDPUSDT',
             'USTUSDT']


# FUNCTIONS

def get_pairs(quote: str = 'USDT', market: str = 'SPOT') -> List[str]:
    """returns all active pairs for a given quote currency. possible values for
    quote are USDT, BTC, BNB etc. possible values for market are SPOT or CROSS"""

    if market == 'SPOT':
        info = client.get_exchange_info()
        symbols = info.get('symbols')
        pairs = []
        for sym in symbols:
            right_quote = sym.get('quoteAsset') == quote
            right_market = market in sym.get('permissions')
            trading = sym.get('status') == 'TRADING'
            allowed = sym.get('symbol') not in not_pairs
            if right_quote and right_market and trading and allowed:
                pairs.append(sym.get('symbol'))
    elif market == 'CROSS':
        pairs = []
        info = client.get_margin_all_pairs()
        for i in info:
            if i.get('quote') == quote:
                pairs.append(i.get('symbol'))
    return pairs


def get_ohlc(pair):
    try:
        pair_path = ohlc_path / f"{pair}.pkl"
        return pd.read_pickle(pair_path)
    except FileNotFoundError:
        klines = client.get_historical_klines(symbol='BTCUSDT',
                                              interval=Client.KLINE_INTERVAL_1MINUTE,
                                              start_str="1 year ago UTC")
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = df['timestamp'] * 1000000
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
                 'taker buy quote vol', 'ignore'], axis=1, inplace=True)

        return df


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """resamples a dataframe and resets the datetime index"""

    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum',
                                                     'vwma': 'last'
                                                     # 'sma3': 'last',
                                                     # 'hidden_flow': 'last',
                                                     # 'hf_upper': 'last',
                                                     # 'hf_lower': 'last',
                                                     # 'hidden_flow_hi': 'last',
                                                     # 'hidden_flow_lo': 'last'
                                                     })
    df = df.dropna(how='any')
    df = df.reset_index()  # don't use drop=True because i want the
    # timestamp index back as a column

    return df


def hidden_flow(df, lookback):
    df['hlc3'] = df[['high', 'low', 'close']].mean(axis=1)
    df['sma3'] = df.hlc3.rolling(lookback).mean()
    df['rolling_pricevol'] = (df.hlc3 * df.volume).rolling(lookback).sum()
    df['rolling_vol'] = df.volume.rolling(lookback).sum()
    df['vwap'] = df.rolling_pricevol / df.rolling_vol

    df['hidden_flow'] = df.vwap / df.sma3
    df['hf_avg'] = df.hidden_flow.rolling(lookback * 5).mean()
    df['hf_std'] = df.hidden_flow.rolling(lookback * 5).std()
    df['hf_upper'] = df.hf_avg + df.hf_std
    df['hf_lower'] = df.hf_avg - df.hf_std

    df['hidden_flow_hi'] = df.hidden_flow < df.hf_upper
    df['hidden_flow_lo'] = df.hidden_flow > df.hf_lower

    return df


def atr(df: pd.DataFrame, lb: int) -> None:
    '''calculates the average true range on an ohlc dataframe'''
    df['tr1'] = df.high - df.low
    df['tr2'] = abs(df.high - df.close.shift(1))
    df['tr3'] = abs(df.low - df.close.shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df[f'atr-{lb}'] = df['tr'].ewm(lb).mean()
    df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    return df


def vwma(df, lookback):
    df['hlc3'] = df[['high', 'low', 'close']].mean(axis=1)
    df['rolling_pricevol'] = (df.hlc3 * df.volume).rolling(lookback).sum()
    df['rolling_vol'] = df.volume.rolling(lookback).sum()
    df['vwma'] = df.rolling_pricevol / df.rolling_vol

    return df


def williams_fractals(df, spacing=0):
    """calculates williams fractals either on the highs and lows or spaced according to average true range.
    if the spacing value is left at the default 0, no atr spacing will be implemented. if spacing is set to an integer
    above 0, the atr will be calculated with a lookback length equal to the spacing value, and the resulting atr values
    will then be multiplied by one tenth of the spacing value. eg if spacing is set to 5, a 5 period atr series will be
    calculated, and the fractals will be spaced 0.5*atr from the highs and lows of the ohlc candles"""

    if spacing:
        df = atr(df, spacing)
        mult = spacing / 10
        df['fractal_high'] = np.where(df.high == df.high.rolling(5, center=True).max(),
                                      df.high + (mult * df[f'atr-{spacing}']), np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(5, center=True).min(),
                                     df.low - (mult * df[f'atr-{spacing}']), np.nan)
    else:
        df['fractal_high'] = np.where(df.high == df.high.rolling(5, center=True).max(), df.high, np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(5, center=True).min(), df.low, np.nan)

    df['inval_high'] = df.fractal_high.interpolate('pad')
    df['inval_low'] = df.fractal_low.interpolate('pad')

    return df


def inside_bars(df):
    df['inside_bar'] = (df.high < df.high.shift(1)) & (df.low > df.low.shift(1))

    return df


def trend_rate(df, z, bars, mult, source):
    """returns True for any ohlc period which follows a strong trend as defined by the rate-of-change and bars params.
    if price has moved at least a certain percentage within the set number of bars, it meets the criteria"""

    df[f"roc_{bars}"] = df[f"{source}"].pct_change(bars)
    m = df[f"roc_{bars}"].abs().rolling(bars * mult).mean()
    s = df[f"roc_{bars}"].abs().rolling(bars * mult).std()
    df['thresh'] = m + (z * s)

    df['trend_up'] = df[f"roc_{bars}"] > df['thresh']
    df['trend_down'] = df[f"roc_{bars}"] < 0 - df['thresh']

    return df


def trend_consec_bars(df, bars):
    """returns True for any ohlc period which follows a strong trend as defined by a sequence of consecutive periods
    that all move in the same direction"""

    df['up_bar'] = df.close.pct_change() > 0
    df['dir_diff'] = df.up_bar.diff().cumsum()
    df['trend_consec'] = df.dir_diff.groupby(df.dir_diff).cumcount()
    df = df.drop(0)
    df = df.reset_index(drop=True)
    df.trend_consec = df.trend_consec.astype(int) + 1

    df['trend_up'] = df.up_bar & (df.trend_consec >= bars)
    df['trend_down'] = ~df.up_bar & (df.trend_consec >= bars)

    return df


def ema_breakout(df, length, lookback):
    df[f"ema_{length}"] = df.close.ewm(length).mean()
    df['ema_high'] = df[f"ema_{length}"].shift(1).rolling(lookback).max()
    df['ema_low'] = df[f"ema_{length}"].shift(1).rolling(lookback).min()

    df['ema_up'] = df[f"ema_{length}"] > df.ema_high
    df['ema_down'] = df[f"ema_{length}"] < df.ema_low

    return df


def ema_trend(df, length, lookback):
    df[f"ema_{length}"] = df.close.ewm(length).mean()

    df['ema_up'] = df[f"ema_{length}"] > df[f"ema_{length}"].shift(lookback)
    df['ema_down'] = df[f"ema_{length}"] < df[f"ema_{length}"].shift(lookback)

    return df


def entry_signals(df):
    df['long_signal'] = df.inside_bar & df.trend_down & df.ema_up & (df.inval_low < df.low)
    df['short_signal'] = df.inside_bar & df.trend_up & df.ema_down & (df.inval_high > df.high)

    return df


def test_next_bar(df):
    df['roc1'] = df.close.pct_change()

    df['short_shift'] = df.short_signal.shift(1)
    df['short_mb_hi_shift'] = df.high.shift(2)
    df['short_r'] = (df.open - df.short_mb_hi_shift) / df.open
    df['short_pnl_r'] = (df.roc1 / df.short_r).clip(lower=-1.2)

    df['long_shift'] = df.long_signal.shift(1)
    df['long_mb_lo_shift'] = df.low.shift(2)
    df['long_r'] = (df.open - df.long_mb_lo_shift) / df.open
    df['long_pnl_r'] = (df.roc1 / df.long_r).clip(lower=-1.2)

    df = df.drop(0)
    df = df.reset_index(drop=True)

    return df


def process_trade(df, group, direction):
    i, trade = group
    orig_last = trade.index[-1]
    trade = trade.reset_index(drop=True)
    new_last = trade.index[-1]
    entry = trade.at[0, 'open']
    inval = trade.at[0, f'{direction}_stops']
    ex = trade.at[new_last, f'{direction}_stops']
    r_val = abs(entry - inval)
    if direction == 'long':
        trade_r = (ex - entry) / r_val
        trade[f'{direction}_pnl_evo'] = (trade.close - entry) / r_val
    else:
        trade_r = (entry - ex) / r_val
        trade[f'{direction}_pnl_evo'] = (entry - trade.close) / r_val
    df.at[orig_last, f'{direction}_pnls'] = trade_r
    trade.at[new_last, f'{direction}_pnls'] = trade_r
    # print(f"{i} - {direction} pnl(R): {trade_r:.2f}, entry: {entry}, inval: {inval}, exit: {ex}")
    return df, trade


def calc_dd(group, direction):
    group['pnl_hi'] = group[f'{direction}_pnl_evo'].cummax()
    group['drawdown'] = group.pnl_hi - group[f'{direction}_pnl_evo']
    return group.drawdown.max()


def test_frac_swing(df):
    df['long_shift'] = df.long_signal.shift(1, fill_value=False)
    df['long_r'] = ((df.open - df.inval_low) / df.open).clip(lower=0.00001)
    df['long_inval'] = df.inval_low

    df['short_shift'] = df.short_signal.shift(1, fill_value=False)
    df['short_r'] = ((df.open - df.inval_high) / df.open).clip(lower=0.00001)
    df['short_inval'] = df.inval_high

    in_long, long_idx, long_stop, long_stops = False, [], np.nan, []
    in_short, short_idx, short_stop, short_stops = False, [], np.nan, []

    for row in df.itertuples():
        if in_long:
            long_stop = max(long_stop, row.long_inval)
        if in_short:
            short_stop = min(short_stop, row.short_inval)

        if row.long_shift:
            in_long = True
            long_stop = row.long_inval
        elif row.close < long_stop:
            in_long = False

        if row.short_shift:
            in_short = True
            short_stop = row.short_inval
        elif row.close > short_stop:
            in_short = False

        long_idx.append(in_long)
        short_idx.append(in_short)

        if in_long:
            long_stops.append(long_stop)
        else:
            long_stops.append(np.nan)
        if in_short:
            short_stops.append(short_stop)
        else:
            short_stops.append(np.nan)

    df['in_long'] = long_idx
    df['long_stops'] = long_stops
    df['in_short'] = short_idx
    df['short_stops'] = short_stops

    df['long_pnls'] = np.nan
    df['long_pnl_evo'] = np.nan
    df['short_pnls'] = np.nan
    df['short_pnl_evo'] = np.nan

    longs = df.groupby(df.in_long.diff().loc[df.in_long].cumsum())
    shorts = df.groupby(df.in_short.diff().loc[df.in_short].cumsum())

    drawdowns = []

    for trade in longs:
        df, group = process_trade(df, trade, 'long')
        drawdowns.append(calc_dd(group, 'long'))
    for trade in shorts:
        df, group = process_trade(df, trade, 'short')
        drawdowns.append(calc_dd(group, 'short'))

    # if len(drawdowns) > 2:
    #     print(f"avg drawdown: {stats.mean(drawdowns):.3f}")
    # else:
    #     print(f"drawdowns: {drawdowns}")

    # need to work out how to get the pnl evolutions back into the original dataframe from the groups
    df['long_pnl_r'] = longs['long_pnls'].transform(lambda x: x)
    df['short_pnl_r'] = shorts['short_pnls'].transform(lambda x: x)

    return df


def ib_signals(df, trend_type, z, bars, mult, source, ema_len, ema_lb, atr_val=0):
    df = inside_bars(df)

    if trend_type == 'trend':
        df = ema_trend(df, ema_len, ema_lb)
    elif trend_type == 'breakout':
        df = ema_breakout(df, ema_len, ema_len)

    df = trend_rate(df, z, bars, mult, source)
    df = williams_fractals(df, atr_val)
    df = entry_signals(df)

    return df


def plot_chart(df, trend_type, length, roc_bars, tail_bars, head_bars):
    # df = (df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    #       .resample('1D', on='timestamp')
    #       .agg({'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #             'volume': 'sum'}))
    # df = df.dropna(how='any')
    # df = df.reset_index()

    df = df.tail(tail_bars)
    # df = df.reset_index(drop=True)
    if head_bars:
        df = df.head(head_bars)
    df = df.reset_index(drop=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.6, 0.2, 0.2])
    # add OHLC trace
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 showlegend=False
                                 ))

    short_index = list(df.loc[df.in_short].index)
    fig.add_trace(go.Candlestick(x=short_index,
                                 open=df.open.iloc[short_index],
                                 high=df.high.iloc[short_index],
                                 low=df.low.iloc[short_index],
                                 close=df.close.iloc[short_index],
                                 increasing={'line': {'color': 'yellow'}},
                                 decreasing={'line': {'color': 'yellow'}},
                                 name='short signals',
                                 # showlegend=False
                                 ))

    long_index = list(df.loc[df.in_long].index)
    fig.add_trace(go.Candlestick(x=long_index,
                                 open=df.open.iloc[long_index],
                                 high=df.high.iloc[long_index],
                                 low=df.low.iloc[long_index],
                                 close=df.close.iloc[long_index],
                                 increasing={'line': {'color': 'purple'}},
                                 decreasing={'line': {'color': 'purple'}},
                                 name='long signals',
                                 # showlegend=False
                                 ))

    ema_colours = ['green' if x else 'red' for x in list(df.ema_up)]
    fig.add_trace(go.Scatter(x=df.index,
                             y=df[f"ema_{length}"],
                             opacity=0.7,
                             mode='markers',
                             marker=dict(color=ema_colours, size=2),
                             name=f"ema_{length}"
                             ))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.short_stops,
                             opacity=0.7,
                             mode='markers',
                             marker=dict(color='red', size=5),
                             name='short stop'))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.long_stops,
                             opacity=0.7,
                             mode='markers',
                             marker=dict(color='green', size=5),
                             name='long stop'))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.vwma,
                             opacity=0.7,
                             line=dict(color='white', width=2),
                             name=f"volume weighted avg"))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df[f"roc_{roc_bars}"],
                             opacity=0.7,
                             line=dict(color='blue', width=2),
                             name=f"roc_{roc_bars}"),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.thresh,
                             opacity=0.7,
                             line=dict(color='orange', width=2),
                             name='thresh'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=0 - df.thresh,
                             opacity=0.7,
                             line=dict(color='orange', width=2),
                             name='thresh'),
                  row=2, col=1)

    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.hidden_flow,
    #                          opacity=0.7,
    #                          mode='lines',
    #                          line=dict(color='black', width=2),
    #                          name='hidden flow'),
    #               row=3, col=1)
    #
    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.hf_upper,
    #                          opacity=0.7,
    #                          mode='lines',
    #                          line=dict(color='red', width=2),
    #                          name='upper bandw'),
    #               row=3, col=1)
    #
    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.hf_lower,
    #                          opacity=0.7,
    #                          mode='lines',
    #                          line=dict(color='red', width=2),
    #                          name='lower band'),
    #               row=3, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


def calc_pnl(pnls):
    """takes a list of trade PnLs denominated in R, and calculates the cumulative profit"""
    start_bal = bal = 100
    risk_pct = 10
    for i in pnls:
        print(bal)
        pnl_factor = 1 + (risk_pct * i / 100)
        bal = bal * pnl_factor
    print(f"final bal: {bal:.3f}, pnl = {(bal / start_bal) - 1:.1%}")
    return (bal / start_bal) - 1


def projected_time(counter):
    if counter % round(100) == 0:
        current_time = time.perf_counter() - start
        projected_total = round((num_tests * current_time / (counter + 1)))
        projected_time = round(projected_total - current_time)
        proj_tot = f"{projected_total // 3600}h {(projected_total // 60) % 60}m {projected_total % 60}s"
        proj_str = f"{projected_time // 3600}h {(projected_time // 60) % 60}m {projected_time % 60}s"
        print(f"{counter / num_tests:.1%} completed, estimated time remaining: {proj_str} ({proj_tot} total)")


# MAIN

#TODO when this test is finished, copy the results dataframe to the laptop, change tr_source to 'close', and run it again

if __name__ == '__main__':

    pair = 'BTCUSDT'

    t_type = 'trend' # trend type ('trend' or 'breakout')
    tr_source = 'vwma' # trend rate source ('close' or 'vwma')

    # timeframes = {'3min': 3, '5min': 5, '10min': 10, '30min': 30}
    timeframes = {'3min': 3}
    z_scores = [2, 2.5, 3]
    # z_scores = [3]
    bars = list(range(8, 19, 2))
    # bars = [7]
    windows = [350, 450, 550, 650, 750]
    # windows = [550]
    lookbacks = range(20)  # [10, 15, 20, 25]
    # lookbacks = [20]
    mults = range(5, 11)
    # mults = [10]
    atr_vals = range(0, 5)
    # atr_vals = [0]

    results = {}
    counter = 0

    df_orig = get_ohlc(pair)

    start = time.perf_counter()

    num_tests = (len(timeframes.keys()) * len(z_scores) * len(bars)
                 * len(mults) * len(windows) * len(lookbacks) * len(atr_vals))
    print(f"number of tests: {num_tests}")

    for tf in timeframes.keys():
        data = df_orig.copy()
        data = vwma(data, timeframes[tf])
        # data = hidden_flow(data, 100)
        data = resample(data, tf)

        for z, bar, mult, window, lb, atr_val in it.product(z_scores, bars, mults,
                                                                    windows, lookbacks, atr_vals):
            # print(f"{tf = } {tt = } {z = } {bar = } {mult = } {window = } {lb = } {atr_val = }")
            df = ib_signals(data, t_type, z, bar, mult, tr_source, window, lb, atr_val)
            df = test_frac_swing(df)
            # plot_fractals(data, 1440)

            # long_rs = df.long_pnl_r.loc[df.long_shift]
            long_rs = df.long_pnl_r.dropna()
            mean_long = long_rs.mean()
            med_long = long_rs.median()
            tot_long = long_rs.sum()

            # short_rs = df.short_pnl_r.loc[df.short_shift]
            short_rs = df.short_pnl_r.dropna()
            mean_short = short_rs.mean()
            med_short = short_rs.median()
            tot_short = short_rs.sum()

            all_rs = list(pd.concat([long_rs, short_rs]).sort_index())
            # print(all_rs)

            mean_r = (mean_long + mean_short) / 2
            med_r = (med_long + med_short) / 2
            tot_r = tot_long + tot_short

            df_signals = df.loc[(df.long_signal) | (df.short_signal), :]
            len_1, len_2 = len(df), len(df_signals)

            results[counter] = {'timeframe': tf, 'type': t_type, 'source': tr_source, 'z_score': z, 'bars': bar, 'mult': mult, 'ema_window': window,
                                'lookback': lb, 'atr': atr_val, 'num_signals': len_2,
                                'mean_long_r': mean_long, 'med_long_r': med_long, 'total_long_r': tot_long,
                                'mean_short_r': mean_short, 'med_short_r': med_short, 'total_short_r': tot_short,
                                'mean_r': mean_r, 'med_r': med_r, 'total_r': tot_r, 'all_rs': all_rs}
            counter += 1
            projected_time(counter)

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_pickle('results.pkl')
    print(f"Number of tests: {counter + 1}")
    print(results_df.loc[(results_df.num_signals > 50) & (results_df.med_r > 0)]
          .sort_values('total_r', ascending=False).head(30))
    # plot_chart(df, tt, window, bar, 6000, 1500)

    end = time.perf_counter()
    elapsed = round(end - start)
    print(f"Time taken: {elapsed // 60}m {elapsed % 60}s")
