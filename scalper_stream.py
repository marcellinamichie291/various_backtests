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


def volume_delta(msg):
    volume = float(msg['k']['q'])
    taker_vol = float(msg['k']['Q'])
    return (2 * taker_vol) - volume

class Ohlc_Stream():

    def __init__(self, params, max_lb, live):
        self.id = params[0]
        self.pair = params[1]
        self.timeframe = params[2]

        if live:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey)
        else:
            self.client = Client(api_key=keys.bPkey, api_secret=keys.bSkey, testnet=True)

        self.max_lb = max_lb
        self.ohlc = self.preload_ohlc(self.pair, self.timeframe)
        self.latest = self.ohlc.pop()
        self.prices = [self.latest['open'], self.latest['high'], self.latest['low']]
        self.volumes = [0] * 3

        print('init stream:', self.id, self.pair, self.timeframe, self.max_lb)

    def preload_ohlc(self, pair, timeframe):
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
        klines = self.client.get_klines(symbol=self.pair.upper(), interval=tf[self.timeframe], limit=self.max_lb)

        return [{'timestamp': int(x[0]) * 1000000,
                 'open': float(x[1]),
                 'high': float(x[2]),
                 'low': float(x[3]),
                 'close': float(x[4]),
                 'volume': float(x[7]),
                 'vwap': (float(x[1]) + float(x[2]) + float(x[3]) + float(x[4])) / 4,
                 # using ohlc4 to approximate vwap for old data
                 'vol_delta': (2 * float(x[10])) - float(x[7])
                 } for x in klines]

    def update_ohlc(self, msg):
        self.ohlc.append(
            {'timestamp': msg['k']['t' ] *1000000,
             'open': float(msg['k']['o']),
             'high': float(msg['k']['h']),
             'low': float(msg['k']['l']),
             'close': float(msg['k']['c']),
             'volume': float(msg['k']['q']),
             'vwap': self.vwap(),
             'vol_delta': volume_delta(msg)
             }
        )
        self.prices.clear()
        self.volumes.clear()
        self.ohlc = self.ohlc[-self.max_lb:]

    def vwap(self):
        data = pd.DataFrame({'price': self.prices, 'volume': self.volumes})
        data['vol_diff'] = data.volume.diff().fillna(data.volume[0])
        data['vwap'] = (data.price * data.vol_diff).cumsum() / data.volume

        return data.at[len(data) - 1, 'vwap']

