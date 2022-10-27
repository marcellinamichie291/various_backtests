import datetime
import keys
import pandas as pd
from pushbullet import Pushbullet
from decimal import Decimal, getcontext
import json
import websocket
import numpy as np
from binance import Client
from scalper_agent import Agent
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

# CONFIGURATION
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)


# FUNCTIONS


def on_open(ws):
    print('connection opened')


def on_close(ws):
    print('connection closed')


def on_message(ws, msg):
    details = parse_msg(msg)
    agent = agents[details['agent']]
    agent.prices.append(details['price'])
    agent.volumes.append(details['volume'])
    if details['close']:
        agent.run_calcs(details['data'])


def parse_msg(msg):
    message = json.loads(msg)

    if len(agents.keys()) == 1:
        stream_id = f"{message['k']['s'].lower()}@kline_{message['k']['i']}"
        data = message
    else:
        stream_id = message['stream']
        data = message['data']

    return {'agent': stream_id,
            'data': data,
            'price': float(data['k']['c']),
            'volume': float(data['k']['q']),
            'time': data['k']['t'] * 1000000,
            'close': data['k']['x']}


def set_stream(settings, live):
    if live:
        base_ep = 'wss://stream.binance.com:9443/ws/'
        stream_ep = 'wss://stream.binance.com/stream'
    else:
        base_ep = 'wss://testnet.binance.vision/ws/'
        stream_ep = 'wss://testnet.binance.vision/stream'

    if len(settings) == 1:
        stream = f'{base_ep}{settings[0][0]}@kline_{settings[0][1]}'
    else:
        elements = [f"{i[0]}@kline_{i[1]}" for i in settings]
        stream = f'{stream_ep}?streams=' + '/'.join(elements)

    print(stream)

    return stream


# CONSTANTS
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12

# Settings
agent_params = [
    {'pair': 'btcusdt',
     'tf': '1m',
     'bias_lb': 450,  # long-term ema trend for bullish/bearish bias
     'bias_roc_lb': 8,  # lookback for judging if the long-term ema is moving up or down
     'source': 'vwap',  # timeseries source for trend_rate calculation
     'bars': 10,  # lookback for the ROC that is applied to the source in trend_rate
     'mult': 9,  # bars * mult gives the lookback window for finding the rolling mean and stdev of the ROC series
     'z': 2,  # z-score that the threshold is set to, to decide what is True or False in trend_rate
     'width': 5  # for a high/low to qualify as a williams fractal, it must be the highest/lowest of this many bars
     },
    {'pair': 'btcusdt',
     'tf': '5m',
     'bias_lb': 450,  # long-term ema trend for bullish/bearish bias
     'bias_roc_lb': 8,  # lookback for judging if the long-term ema is moving up or down
     'source': 'vwap',  # timeseries source for trend_rate calculation
     'bars': 10,  # lookback for the ROC that is applied to the source in trend_rate
     'mult': 9,  # bars * mult gives the lookback window for finding the rolling mean and stdev of the ROC series
     'z': 2,  # z-score that the threshold is set to, to decide what is True or False in trend_rate
     'width': 5  # for a high/low to qualify as a williams fractal, it must be the highest/lowest of this many bars
     },
    {'pair': 'btcusdt',
     'tf': '15m',
     'bias_lb': 450,  # long-term ema trend for bullish/bearish bias
     'bias_roc_lb': 8,  # lookback for judging if the long-term ema is moving up or down
     'source': 'vwap',  # timeseries source for trend_rate calculation
     'bars': 10,  # lookback for the ROC that is applied to the source in trend_rate
     'mult': 9,  # bars * mult gives the lookback window for finding the rolling mean and stdev of the ROC series
     'z': 2,  # z-score that the threshold is set to, to decide what is True or False in trend_rate
     'width': 5  # for a high/low to qualify as a williams fractal, it must be the highest/lowest of this many bars
     },
]

live = False
streams = [(agent['pair'], agent['tf']) for agent in agent_params]
stream = set_stream(streams, live)

# Run Program
agents = {f"{params['pair']}@kline_{params['tf']}": Agent(params, live) for params in agent_params}

ws = websocket.WebSocketApp(stream, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()

# FUNCTIONS


# def is_empty_message(message):
#     if message is False:
#         return True
#     if '"result":null' in message:
#         return True
#     if '"result":None' in message:
#         return True
#     return False
#
#
# def handle_price_change(symbol, timestamp, price):
#     print(f"Handle price change for symbol: {symbol}, timestamp: {timestamp}, price: {price}")
#
#
# def process_stream_data(binance_websocket_api_manager):
#     while True:
#         if binance_websocket_api_manager.is_manager_stopping():
#             exit(0)
#         oldest_data = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
#         is_empty = is_empty_message(oldest_data)
#         if is_empty:
#             time.sleep(0.01)
#         else:
#             oldest_data_dict = json.loads(oldest_data)
#             data = oldest_data_dict['data']
#             #  Handle price change
#             handle_price_change(symbol=data['s'], timestamp=data['T'], price=data['p'])
#
#
# def start_websocket_listener():
#     binance_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance")
#     channels = {'trade', }
#     binance_websocket_api_manager.create_stream(channels, markets='BTCUSDT', api_key=keys.bPkey, api_secret=keys.bSkey)
#     # Start a worker process to move the received stream_data from the stream_buffer to a print function
#     worker_thread = threading.Thread(target=process_stream_data, args=(binance_websocket_api_manager,))
#     worker_thread.start()
#
#
# def compare_server_times():
#     server_time = client.get_server_time()
#     aa = str(server_time)
#     bb = aa.replace("{'serverTime': ", "")
#     aa = bb.replace("}", "")
#     gg = int(aa)
#     ff = gg - 10799260
#     uu = ff / 1000
#     yy = int(uu)
#     tt = time.localtime(yy)
#     print(f"Binance Server time: {tt}")
#     print(f"Local time: {time.localtime()}")
#

#  Compare server and local times
# compare_server_times()
# start_websocket_listener()
