import datetime
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

# from pandarallel import pandarallel
# pandarallel.initialize()

# CONSTANTS
client = Client(keys.bPkey, keys.bSkey)
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
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
    print('runnning get_ohlc')
    try:
        # pair_path = ohlc_path / f"{pair}.pkl"
        pair_path = f"{pair}_1m.pkl"
        return pd.read_pickle(pair_path)
    except (FileNotFoundError, OSError):
        klines = client.get_historical_klines(pair, Client.KLINE_INTERVAL_1MINUTE, '1 year ago UTC')
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
                'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = df['timestamp'] * 1000000
        df = df.astype(float)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
                      'taker buy quote vol', 'ignore'], axis=1)

        return df


def update_ohlc(pair: str, timeframe: str, old_df: pd.DataFrame) -> pd.DataFrame:
    print('runnning update_ohlc')
    """takes an ohlc dataframe, works out when the data ends, then requests from
    binance all data from the end to the current moment. It then joins the new
    data onto the old data and returns the updated dataframe"""

    # client = Client(keys.bPkey, keys.bSkey)
    tf = {'1m': Client.KLINE_INTERVAL_1MINUTE,
          '5m': Client.KLINE_INTERVAL_5MINUTE,
          '15m': Client.KLINE_INTERVAL_15MINUTE,
          '30m': Client.KLINE_INTERVAL_30MINUTE,
          '1h': Client.KLINE_INTERVAL_1HOUR,
          '4h': Client.KLINE_INTERVAL_4HOUR,
          '6h': Client.KLINE_INTERVAL_6HOUR,
          '8h': Client.KLINE_INTERVAL_8HOUR,
          '12h': Client.KLINE_INTERVAL_12HOUR,
          '1d': Client.KLINE_INTERVAL_1DAY,
          '3d': Client.KLINE_INTERVAL_3DAY,
          '1w': Client.KLINE_INTERVAL_1WEEK,
          }
    old_end = int(old_df.at[len(old_df) - 1, 'timestamp'].timestamp()) * 1000
    klines = client.get_klines(symbol=pair, interval=tf.get(timeframe),
                               startTime=old_end)
    cols = ['timestamp', 'open', 'high', 'low', 'close', 'base vol', 'close time',
            'volume', 'num trades', 'taker buy base vol', 'taker buy quote vol', 'ignore']
    df = pd.DataFrame(klines, columns=cols)
    df['timestamp'] = df['timestamp'] * 1000000
    df = df.astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(['base vol', 'close time', 'num trades', 'taker buy base vol',
             'taker buy quote vol', 'ignore'], axis=1, inplace=True)

    df_new = pd.concat([old_df[:-1], df], copy=True, ignore_index=True)
    return df_new


def ohlc_1yr(pair):
    fp = Path(f"{pair}_1m.pkl")
    if fp.exists():
        df = pd.read_pickle(fp)
        df = update_ohlc(pair, '1m', df)
        df = df.tail(525600)
        df = df.reset_index(drop=True)
    else:
        df = get_ohlc(pair)

    df.to_pickle(fp)

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


def williams_fractals(df, frac_width, atr_spacing=0):
    """calculates williams fractals either on the highs and lows or spaced according to average true range.
    if the spacing value is left at the default 0, no atr spacing will be implemented. if spacing is set to an integer
    above 0, the atr will be calculated with a lookback length equal to the spacing value, and the resulting atr values
    will then be multiplied by one tenth of the spacing value. eg if spacing is set to 5, a 5 period atr series will be
    calculated, and the fractals will be spaced 0.5*atr from the highs and lows of the ohlc candles
    frac_width determines how many candles are used to decide if the current candle is a local high/low, so a frac_width
    of five will look at the current candle, the two previous candles, and the two subsequent ones"""

    if atr_spacing:
        df = atr(df, atr_spacing)
        mult = atr_spacing / 10
        df['fractal_high'] = np.where(df.high == df.high.rolling(frac_width, center=True).max(),
                                      df.high + (mult * df[f'atr-{atr_spacing}']), np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(frac_width, center=True).min(),
                                     df.low - (mult * df[f'atr-{atr_spacing}']), np.nan)
    else:
        df['fractal_high'] = np.where(df.high == df.high.rolling(frac_width, center=True).max(), df.high, np.nan)
        df['fractal_low'] = np.where(df.low == df.low.rolling(frac_width, center=True).min(), df.low, np.nan)

    df['inval_high'] = df.fractal_high.interpolate('pad')
    df['inval_low'] = df.fractal_low.interpolate('pad')

    return df


def fractal_density(df, lookback, frac_width):
    """a way of detecting when price is trending based on how frequently williams fractals are printed, since they are
    much more common during choppy conditions and spaced further apart during trending conditions.
    the calculation is simply the total number of fractals (high+low) in a given lookback period divided by the lookback.
    i could modify it by working out how to normalise the output to a range of 0-1, but for now the range of possible
    values is from 0 to some number between 0 and 1, since the most fractals you could possibly have in any lookback
    period is going to be significantly less than the period itself and dependent on the frac_width parameter"""

    df = williams_fractals(df, frac_width)

    return (df.fractal_high.rolling(lookback).count() + df.fractal_low.rolling(lookback).count()) / lookback


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
    df['long_signal'] = df.inside_bar & df.trend_down & df.ema_up# & (df.inval_low < df.low)
    df['short_signal'] = df.inside_bar & df.trend_up & df.ema_down# & (df.inval_high > df.high)

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

    curr_stdev = trade.close.std()
    min_inval = curr_stdev / 100

    entry = trade.at[0, 'open']
    inval = trade.at[0, f'{direction}_stops']
    ex = trade.at[new_last, f'{direction}_stops']
    r_val = max(abs(entry - inval), min_inval)
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

    df = williams_fractals(df, width)

    # calculate R and invalidation for each signal
    df['long_shift'] = df.long_signal.shift(1, fill_value=False)
    df['long_r'] = ((df.open - df.inval_low) / df.open).clip(lower=0.00001)
    df['long_inval'] = df.inval_low
    df['long_entry_inval'] = df.low.shift(1).rolling(2).min()

    df['short_shift'] = df.short_signal.shift(1, fill_value=False)
    df['short_r'] = ((df.open - df.inval_high) / df.open).clip(lower=0.00001)
    df['short_inval'] = df.inval_high
    df['short_entry_inval'] = df.high.shift(1).rolling(2).max()

    # map out trades by trailing stops using the inval columns
    in_long, long_idx, long_stop, long_stops = False, [], np.nan, []
    in_short, short_idx, short_stop, short_stops = False, [], np.nan, []

    for row in df.itertuples():
        if in_long:
            long_stop = max(long_stop, row.long_inval)
        if in_short:
            short_stop = min(short_stop, row.short_inval)

        if row.long_shift:
            in_long = True
            long_stop = min(row.long_inval, row.long_entry_inval)
        elif row.close < long_stop:
            in_long = False

        if row.short_shift:
            in_short = True
            short_stop = max(row.short_inval, row.short_entry_inval)
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

    # put trade data back into dataframe
    df['in_long'] = long_idx
    df['long_stops'] = long_stops
    df['in_short'] = short_idx
    df['short_stops'] = short_stops

    # initialise pnl columns
    df['long_pnls'] = np.nan
    df['long_pnl_evo'] = np.nan
    df['short_pnls'] = np.nan
    df['short_pnl_evo'] = np.nan

    # groupby individual trades
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


def ib_signals(df, trend_type, z, bars, mult, source, ema_len, ema_lb, width):
    df = inside_bars(df)

    if trend_type == 'trend':
        df = ema_trend(df, ema_len, ema_lb)
    elif trend_type == 'breakout':
        df = ema_breakout(df, ema_len, ema_len)

    df = trend_rate(df, z, bars, mult, source)
    df = entry_signals(df)

    return df


def plot_chart(df, pair, trend_type, length, roc_bars, tail_bars, head_bars, show):
    # df = (df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'in_long', 'in_short', 'bal_evo']]
    #       .resample('1D', on='timestamp')
    #       .agg({'open': 'first',
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last',
    #             'volume': 'sum',
    #             'in_long': 'max',
    #             'in_short': 'max',
    #             'bal_evo': 'last'}))
    # df = df.dropna(how='any')
    df = df.reset_index()

    df = df.tail(tail_bars)
    # df = df.reset_index(drop=True)
    if head_bars:
        df = df.head(head_bars)
    df = df.reset_index(drop=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.8, 0.2])
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

    # ema_colours = ['green' if x else 'red' for x in list(df.ema_up)]
    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df[f"ema_{length}"],
    #                          opacity=0.7,
    #                          mode='markers',
    #                          marker=dict(color=ema_colours, size=2),
    #                          name=f"ema_{length}"
    #                          ))

    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.inval_low,
    #                          opacity=0.7,
    #                          mode='markers',
    #                          marker=dict(color='orange', size=5),
    #                          name='inval low'))
    #
    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.inval_high,
    #                          opacity=0.7,
    #                          mode='markers',
    #                          marker=dict(color='blue', size=5),
    #                          name='inval high'))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.short_stops,
                             opacity=0.7,
                             mode='markers',
                             marker=dict(color='red', size=5),
                             name='short stops'))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.long_stops,
                             opacity=0.7,
                             mode='markers',
                             marker=dict(color='green', size=5),
                             name='long stops'))

    # fig.add_trace(go.Scatter(x=df.index,
    #                          y=df.vwma,
    #                          opacity=0.7,
    #                          line=dict(color='white', width=2),
    #                          name=f"volume weighted avg"))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.bal_evo,
                             opacity=0.7,
                             line=dict(color='blue', width=2),
                             name='PnL (R)'),
                  row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False,
                      title={
                          'text': pair,
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top'}
                      )
    if show:
        fig.show()
    # else:
    #     fig.to_image(f"{pair}_results.png",
    #                  format='png',
    #                  width=1920,
    #                  height=1080,
    #                  scale=1)


def calc_pnl(pnls, risk_pct):
    """takes a list of trade PnLs denominated in R, and calculates the cumulative profit"""
    start_bal = bal = 100
    for i in pnls:
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


def calc_pnl_series(pnls, risk_pct):
    """takes a list of trade PnLs denominated in R, and calculates the cumulative account balance from them"""
    bal = 100
    bal_evo = []
    for i in pnls:
        pnl_factor = 1 + (risk_pct * i / 100)
        bal = bal * pnl_factor
        bal_evo.append(bal)
    return pd.Series(bal_evo)


def process(data, source, z, bar, mult, window, lb, width, show_plot):
    # print(f"{tf = } {source = } {z = } {bar = } {mult = } {window = } {lb = } {atr_val = }")
    df = ib_signals(data, t_type, z, bar, mult, source, window, lb, width)
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

    ar_series = pd.concat([long_rs, short_rs]).sort_index()
    df['all_rs'] = ar_series.loc[~ar_series.index.duplicated()].reindex(df.index)
    all_rs = list(pd.concat([long_rs, short_rs]).sort_index())

    mean_r = ar_series.mean()
    med_r = ar_series.median()
    tot_r = ar_series.sum()

    df_signals = df.loc[df.long_signal | df.short_signal, :]
    len_1, len_2 = len(df), len(df_signals)

    pos_r = [r for r in all_rs if r > 0]
    neg_r = [r for r in all_rs if r <= 0]
    if pos_r:
        avg_win = stats.mean(pos_r)
    else:
        avg_win = 0
    if neg_r:
        avg_loss = abs(stats.mean(neg_r))
    else:
        avg_loss = 1

    profit_factor = avg_win / avg_loss
    if all_rs:
        win_rate = len(pos_r) / len(all_rs)
        exp_return = profit_factor * win_rate
    else:
        win_rate = 0
        exp_return = 0

    df['bal_evo'] = calc_pnl_series(df.all_rs.fillna(0), 1)
    plot_chart(df, pair, t_type, window, bar, 6000, 1500, show=show_plot)

    return {'pair': pair, 'timeframe': tf, 'type': t_type, 'source': source, 'z_score': z,
                        'bars': bar, 'mult': mult, 'ema_window': window, 'lookback': lb, 'width': width,
                        'num_signals': len_2, 'mean_long_r': mean_long, 'med_long_r': med_long,
                        'total_long_r': tot_long, 'mean_short_r': mean_short, 'med_short_r': med_short,
                        'total_short_r': tot_short, 'mean_r': mean_r, 'med_r': med_r, 'total_r': tot_r,
                        'win_rate': win_rate, 'profit_factor': profit_factor, 'all_rs': all_rs}


# MAIN

# TODO the main pc takes ~0.4s per test. i should split the results into different timeframes and study each one to see
#  if the best results are falling off the edge of the test ranges

# TODO i really need to walk-forward test all of it

if __name__ == '__main__':

    # pairs = ['BTCUSDT', 'ETHUSDT', 'ETHBTC', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'DOTUSDT',
    #          'MATICUSDT']
    pairs = ['BTCUSDT']

    t_type = 'trend'  # trend type ('trend' or 'breakout')

    # timeframes = {'5min': 5, '15min': 15, '30min': 30, '1h': 60}
    timeframes = {'5min': 5}
    # tr_sources = ['close', 'vwma']
    tr_sources = ['close']
    # z_scores = [1, 1.5, 2, 2.5, 3]
    z_scores = [2]
    # bars = list(range(8, 19, 2))
    bars = [10]
    # mults = range(5, 11)
    mults = [9]
    # windows = [200, 400, 600, 800, 1000]
    windows = [600]
    # lookbacks = range(2, 15, 3)
    lookbacks = [8]
    # wf_widths = [3, 5, 7]
    wf_widths = [5]

    results = {}
    counter = 0

    num_tests = (len(pairs) * len(timeframes.keys()) * len(tr_sources) * len(z_scores)
                 * len(bars) * len(mults) * len(windows) * len(lookbacks) * len(wf_widths))
    print(f"number of tests: {num_tests}")

    start = time.perf_counter()

    for pair in pairs:
        print(pair)
        df_orig = ohlc_1yr(pair)
        for tf in timeframes.keys():
            data = df_orig.copy()
            # data = hidden_flow(data, 100)
            data = vwma(data, timeframes[tf])
            data = resample(data, tf)
            for source, z, bar, mult, window, lb, width in it.product(tr_sources, z_scores, bars, mults, windows,
                                                                        lookbacks, wf_widths):
                results[counter] = process(data, source, z, bar, mult, window, lb, width, show_plot=True)
                counter += 1
                projected_time(counter)

    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_pickle('results.pkl')
    print(results_df.loc[results_df.num_signals > 50].sort_values('total_r', ascending=False).head(30))


    end = time.perf_counter()
    elapsed = round(end - start)
    print(f"Time taken: {elapsed//3600}h {(elapsed//60)%60}m {elapsed%60}s ({elapsed/num_tests:.3f}s/test)")
