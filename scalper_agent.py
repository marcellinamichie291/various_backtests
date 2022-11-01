import datetime
import keys
import pandas as pd
from binance import Client
from pushbullet import Pushbullet as pb
from decimal import Decimal, getcontext
import json
import websocket
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import statistics as stats
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import threading
import time
from pprint import pprint, pformat
from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager


class Agent():

    def __init__(self, params, live):

        self.pair = params['pair']
        self.timeframe = params['tf']
        self.stream = f"{self.pair.lower()}@kline_{self.timeframe}"
        self.bias_lb = params['bias_lb']
        self.bias_roc_lb = params['bias_roc_lb']
        self.source = params['source']
        self.bars = params['bars']
        self.mult = params['mult']
        self.z = params['z']
        self.width = params['width']
        self.name = f"{self.pair}_{self.timeframe}_{self.bias_lb}_{self.source}_{self.bars}_{self.mult}_{self.z}_{self.width}"

        if live:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey)
        else:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey, testnet=True)

        # max_lb = the highest of bias_lb and bars*mult, but not more than 1000
        self.max_lb = min(max(self.bias_lb, (self.bars * self.mult)), 1000)

        print(f"init agent: {self.name}")

    def make_dataframe(self, ohlc_data):
        self.df = pd.DataFrame(ohlc_data)
        self.df['timestamp'] = pd.to_datetime(self.df.timestamp, utc=True)

    def inside_bars(self):
        self.df['inside_bar'] = (self.df.high < self.df.high.shift(1)) & (self.df.low > self.df.low.shift(1))
        last_idx = self.df.index[-1]
        if self.df.at[last_idx, 'inside_bar']:
            print(f"{self.stream} inside bar detected. {datetime.datetime.now().strftime('%d/%m/%y %H:%M')}")

    def ema_trend(self):
        length = self.bias_lb
        span = self.bias_roc_lb
        self.df[f"ema_{length}"] = self.df.close.ewm(length).mean()
        self.df['ema_up'] = self.df[f"ema_{length}"] > self.df[f"ema_{length}"].shift(span)
        self.df['ema_down'] = self.df[f"ema_{length}"] < self.df[f"ema_{length}"].shift(span)

    def trend_rate(self):
        """returns True for any ohlc period which follows a strong trend as defined by the rate-of-change and bars params.
        if price has moved at least a certain percentage within the set number of bars, it meets the criteria"""

        self.df[f"roc_{self.bars}"] = self.df[f"{self.source}"].pct_change(self.bars)
        m = self.df[f"roc_{self.bars}"].abs().rolling(self.bars * self.mult).mean()
        s = self.df[f"roc_{self.bars}"].abs().rolling(self.bars * self.mult).std()
        self.df['thresh'] = m + (self.z * s)

        self.df['trend_up'] = self.df[f"roc_{self.bars}"] > self.df['thresh']
        self.df['trend_down'] = self.df[f"roc_{self.bars}"] < 0 - self.df['thresh']

    def calc_atr(self, lb) -> None:
        """calculates the average true range on an ohlc dataframe"""
        self.df['tr1'] = self.df.high - self.df.low
        self.df['tr2'] = abs(self.df.high - self.df.close.shift(1))
        self.df['tr3'] = abs(self.df.low - self.df.close.shift(1))
        self.df['tr'] = self.df[['tr1', 'tr2', 'tr3']].max(axis=1)
        self.df[f'atr-{lb}'] = self.df['tr'].ewm(lb).mean()
        self.df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1, inplace=True)

    def williams_fractals(self):
        """calculates williams fractals on the highs and lows.
        frac_width determines how many candles are used to decide if the current candle is a local high/low, so a frac_width
        of five will look at the current candle, the two previous candles, and the two subsequent ones"""

        self.df['fractal_high'] = np.where(self.df.high == self.df.high.rolling(self.width, center=True).max(),
                                           self.df.high, np.nan)
        self.df['fractal_low'] = np.where(self.df.low == self.df.low.rolling(self.width, center=True).min(),
                                          self.df.low, np.nan)

        self.df['inval_high'] = self.df.fractal_high.interpolate('pad')
        self.df['inval_low'] = self.df.fractal_low.interpolate('pad')

    def entry_signals(self):
        self.df['long_signal'] = (self.df.inside_bar & self.df.trend_down &
                                  self.df.ema_up & (self.df.inval_low < self.df.low))
        self.df['short_signal'] = (self.df.inside_bar & self.df.trend_up &
                                   self.df.ema_down & (self.df.inval_high > self.df.high))

        last_idx = self.df.index[-1]
        if self.df.at[last_idx, 'long_signal']:
            print(f"{self.name} long signal generated. {datetime.datetime.now().strftime('%d/%m/%y %H:%M')}")
        if self.df.at[last_idx, 'short_signal']:
            print(f"{self.name} short signal generated. {datetime.datetime.now().strftime('%d/%m/%y %H:%M')}")

    def run_calcs(self, data, ohlc_data):
        # print(1)
        self.make_dataframe(ohlc_data)
        # print(2)
        self.inside_bars()
        # print(3)
        self.ema_trend()
        # print(4)
        self.trend_rate()
        # print(5)
        self.williams_fractals()
        # print(6)
        self.entry_signals()
        # print(7)

        last = self.df.to_dict('records')[-1]
        ib = 'inside bar' if last['inside_bar'] else 'no inside bar'
        eu = 'long ema' if last['ema_up'] else 'short ema'
        if last['trend_up']:
            tu = 'short trend'
        elif last['trend_down']:
            tu = 'long trend'
        else:
            tu = 'no trend'
        il = 'long inval ok' if last['inval_low'] < last['low'] else 'no long inval'
        ih = 'short inval ok' if last['inval_high'] > last['high'] else 'no short inval'
        if last['long_signal']:
            signal = 'long signal'
        elif last['short_signal']:
            signal = 'short signal'
        else:
            signal = 'no signal'

        print(f"{ib}, {eu}, {tu}, {ih}, {il}, {signal}")

        if last['long_signal']:
            now = datetime.datetime.now().strftime('%d/%m/%y %H:%M')
            note = (f"{self.pair} {self.timeframe} long signal, "
                    f"invalidation: {last['inval_low']}, {last['vol_delta'] = }")
            print(now, note)
            pb.push_note(title=now, body=note)
        if last['short_signal']:
            now = datetime.datetime.now().strftime('%d/%m/%y %H:%M')
            note = (f"{self.pair} {self.timeframe} short signal, "
                    f"invalidation: {last['inval_high']}, {last['vol_delta'] = }")
            print(now, note)
            pb.push_note(title=now, body=note)