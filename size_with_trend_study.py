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

#TODO size with trend didn't really work but i want to generalise this concept to 'size with bias' so i can use any
# method of quantifying a bias and then sizing according to that. the next bias-forming method i should try is proximity
# to horizontal levels

pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)

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
    pair_path = ohlc_path / f"{pair}.pkl"
    return pd.read_pickle(pair_path)


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """resamples a dataframe and resets the datetime index"""

    df = df.resample(timeframe, on='timestamp').agg({'open': 'first',
                                                     'high': 'max',
                                                     'low': 'min',
                                                     'close': 'last',
                                                     'volume': 'sum'})
    df = df.dropna(how='any')
    df = df.reset_index()  # don't use drop=True because i want the
    # timestamp index back as a column

    return df


def add_emas(df, lengths):
    for l in lengths:
        df[f"ema_{l}"] = df.close.ewm(l).mean()

    return df


def remove_emas(df, lengths):

    df = df.drop(list(range(lengths[0])))

    ema_cols = [f"ema_{y}" for y in lengths]
    df = df.drop(ema_cols, axis=1)
    df = df.reset_index(drop=True)

    return df


def trend_index(df, lengths, smoothing):
    score_num = 0

    new_cols = {}

    for i in lengths:
        new_cols[f"score_{score_num}"] = (df[f"ema_{i}"] > df[f"ema_{i}"].shift(1)).astype('int8')
        score_num += 1

    for a, b in it.combinations(lengths, 2):
        new_cols[f"score_{score_num}"] = (df[f"ema_{a}"] > df[f"ema_{b}"]).astype('int8')
        score_num += 1

    score_df = pd.DataFrame(new_cols)
    df['trend_index'] = (score_df.mean(axis=1) * 2 - 1).ewm(smoothing).mean()

    return df


def trend_accelleration(df, lengths):
    accel_cols = {i: df[f"ema_{i}"].pct_change().pct_change().rank(pct=True) for i in lengths}

    accel_df = pd.DataFrame(accel_cols)

    df['trend_accel'] = accel_df.mean(axis=1)

    return df


def size_with_trend(df, threshold=0, fee=0.00075):
    cash = 10000
    asset = 0
    pf_evo = []
    q_sizes = []

    for row in df.itertuples():
        # pre-adjustment values
        asset_val = asset * row.close
        pf_val = cash + asset_val
        allocation = asset_val / pf_val
        adjust = row.trend_index - allocation

        if adjust > threshold:  # INCREASE LONG / DECREASE SHORT
            # print('Buy')
            quote_size = pf_val * adjust
            q_sizes.append(quote_size)
            base_size = quote_size / row.close * (1 - fee)
            cash -= quote_size
            asset += base_size

        elif adjust < (-1 * threshold):  # DECREASE LONG INCREASE SHORT
            # print('Sell')
            quote_size = pf_val * adjust * (1 - fee)
            q_sizes.append(quote_size)
            base_size = quote_size / row.close
            cash -= quote_size
            asset += base_size

        # post-adjustment recalculation
        allocation += adjust
        asset_val = asset * row.close
        pf_val = cash + asset_val  # - liability
        if pf_val < 0:
            break

        pf_evo.append(pf_val)

        # print(f"price: {row.close:.3f}, adjust: {adjust:.3f}, cash: {cash:3f}, asset_val: {asset_val:.3f}, asset qty: {base_size}, pf_val: {pf_val:.3f}, size: {allocation:.3f}")# , liability: {liability}

    df['pf_evo'] = pf_evo

    return df


def plot_chart(df, bars):
    df = (df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trend_index', 'pf_evo']]
          .resample('1D', on='timestamp')
          .agg({'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trend_index': 'last',
                'pf_evo': 'last'}))
    df = df.dropna(how='any')
    df = df.reset_index()

    df = df.tail(bars)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.01,
                        row_heights=[0.3, 0.2, 0.3, 0.2])
    # add OHLC trace
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 showlegend=False))

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.trend_index,
                             opacity=0.7,
                             line=dict(color='blue', width=2),
                             name='Trend Index'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.pf_evo,
                             opacity=0.7,
                             line=dict(color='green', width=2),
                             name='Equity Curve'),
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=df.index,
                             y=df.trend_accel,
                             opacity=0.7,
                             line=dict(color='red', width=2),
                             name='trend acceleration'),
                  row=4, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()


# MAIN

if __name__ == '__main__':

    # timeframes = ['1min', '5min', '15min', '30min', '1H', '2H', '4H', '8H', '12H', '1D']
    timeframes = ['1min']

    # num_tests = 10
    # scalar_low = 20
    # scalar_hi = 120
    # div = num_tests / (scalar_hi - scalar_low)
    # scales = [i / div for i in range(int(scalar_low * div), int((scalar_hi * div) + 1))]
    scales = [90]

    num_emas = 10
    power_low = 2
    power_hi = 13


    for tf, scale in it.product(timeframes, scales):
        pow_div = num_emas / (power_hi - power_low)
        powers = [i / pow_div for i in range(int(power_low * pow_div), int((power_hi * pow_div) + 1))]
        emas = [round((2 ** p) * scale) for p in powers]
        scores = []
        # pairs = [p for p in get_pairs(market='CROSS') if p not in not_pairs]
        pairs = ['BTCUSDT']
        for pair in pairs:
            df = get_ohlc(pair)
            df = resample(df, tf)
            # df = df.tail(200)
            # df = df.reset_index(drop=True)
            df = add_emas(df, emas)
            df = trend_index(df, emas, 5)
            df = trend_accelleration(df, emas)
            df = remove_emas(df, emas)
            try:
                df = size_with_trend(df, 0.2)
            except ValueError:
                # print(f"{pair} liquidated")
                scores.append(0)
                continue

            hodl_gain = (df.at[len(df) - 1, 'close'] / df.at[0, 'close']) - 1
            gain = (df.at[len(df) - 1, 'pf_evo'] / df.at[0, 'pf_evo']) - 1

            df['cum_max'] = df.pf_evo.cummax()
            df['drawdown'] = (df.pf_evo - df.cum_max) / df.cum_max
            avg_dd = 0 - (df.drawdown.mean())
            max_dd = 0 - (df.drawdown.min())

            df['returns'] = (df.pf_evo - df.pf_evo.shift(1)) / df.pf_evo.shift(1)

            df['pf_evo_s'] = df.pf_evo.shift(1)
            log_returns = [math.log((row.pf_evo / row.pf_evo_s)) for row in df.itertuples()]
            df['log_returns'] = log_returns

            # print(f"{pair}, {len(df) = }, Gain: {gain:.1%}, lin std dev: {df.returns.std():.4f}, log std dev: {df.log_returns.std():.4f}, mean dd: {avg_dd:.1%}, max dd: {max_dd:.1%}")

            print(f"gain: {gain:.1%}, hodl: {hodl_gain:.1%}, avg dd: {avg_dd:.1%}, max dd: {max_dd:.1%}")
            score = ((gain - hodl_gain) - (avg_dd + max_dd)) / df.returns.std()
            scores.append(score)

        mean_score = round(stats.mean(scores))
        median_score = round(stats.median(scores))
        try:
            spread = abs(median_score - mean_score) / ((median_score + mean_score) / 2)
        except ZeroDivisionError:
            spread = 0
        # print(f"Timeframe: {tf}, {len(df) = }, EMA range: {emas[0]}-{emas[-1]}, scale: {scale:.1f}, avg score: {mean_score}, spread: {spread:.1%}")
        print(f"Timeframe: {tf}, {len(df) = }, EMA range: {emas[0]}-{emas[-1]}, scale: {scale:.1f}, score: {mean_score}")


plot_chart(df, 200000)
