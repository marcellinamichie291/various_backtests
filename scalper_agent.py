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
        self.stream = f"{self.pair}@kline_{self.timeframe}"
        self.bias_lb = params['bias_lb']
        self.bias_roc_lb = params['bias_roc_lb']
        self.source = params['source']
        self.bars = params['bars']
        self.mult = params['mult']
        self.z = params['z']
        self.width = params['width']

        if live:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey)
        else:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey, testnet=True)

        # max_lb = the highest of bias_lb and bars*mult, but not more than 1000
        self.max_lb = min(max(self.bias_lb, (self.bars * self.mult)), 1000)

        print(f"init agent: {self.pair} {self.timeframe} {self.max_lb}")





    def make_dataframe(self, ohlc_data):
        self.df = pd.DataFrame(ohlc_data)
        self.df['timestamp'] = pd.to_datetime(self.df.timestamp, utc=True)

    def inside_bars(self):
        self.df['inside_bar'] = (self.df.high < self.df.high.shift(1)) & (self.df.low > self.df.low.shift(1))

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
        '''calculates the average true range on an ohlc dataframe'''
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

    def run_calcs(self, data, ohlc_data):
        self.make_dataframe(ohlc_data)
        self.inside_bars()
        self.ema_trend()
        self.trend_rate()
        self.williams_fractals()
        self.entry_signals()

        last = self.df.to_dict('records')[-1]
        if last['long_signal']:
            now = datetime.datetime.now().strftime('%d/%m/%y %H:%M')
            note = (f"{self.pair} {self.timeframe} long signal, "
                    f"invalidation: {last['inval_low']}, {last['vol_delta'] = }")
            pb.push_note(now, note)
            print(now, note)
        if last['short_signal']:
            now = datetime.datetime.now().strftime('%d/%m/%y %H:%M')
            note = (f"{self.pair} {self.timeframe} short signal, "
                    f"invalidation: {last['inval_high']}, {last['vol_delta'] = }")
            pb.push_note(now, note)
            print(now, note)