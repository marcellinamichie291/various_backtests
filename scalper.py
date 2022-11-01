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
from scalper_stream import Ohlc_Stream
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

all_start = time.perf_counter()

# CONFIGURATION
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)


# FUNCTIONS


def on_open(ws):
    print('connection opened')


def on_close(ws, close_status_code, close_msg):
    # Because on_close was triggered, we know the opcode = 8
    print("connection closed")
    if close_status_code or close_msg:
        print("close status code: " + str(close_status_code))
        print("close message: " + str(close_msg))


def on_message(ws, msg):
    start = time.perf_counter()
    details = parse_msg(msg)
    stream = streams[details['stream']]
    stream.prices.append(details['price'])
    stream.volumes.append(details['volume'])
    if details['close']:
        stream.update_ohlc(details['data'])
        for agent_id in agents.keys():
            if details['stream'] in agent_id:
                agent = agents[agent_id]
                # print(f"running calcs for {agent_id}")
                agent.run_calcs(details['data'], stream.ohlc)
    # end = time.perf_counter()
    # print(round(end-start, 6))


def on_ping(ws, msg):
    ping_time = time.perf_counter()
    measure = round(ping_time - all_start)
    # print(f"Pinged at {measure//3600}h {(measure//60)%60}m {measure%60}s")


def on_error(ws, err):
    print(err)


def parse_msg(msg):
    message = json.loads(msg)

    if len(streams) == 1:
        stream_id = f"{message['k']['s'].lower()}@kline_{message['k']['i']}"
        data = message
    else:
        stream_id = message['stream']
        data = message['data']

    return {'stream': stream_id,
            'data': data,
            'price': float(data['k']['c']),
            'volume': float(data['k']['q']),
            'time': data['k']['t'] * 1000000,
            'close': data['k']['x']}


def build_feed(settings, live):
    if live:
        base_ep = 'wss://stream.binance.com:9443/ws/'
        stream_ep = 'wss://stream.binance.com/stream'
    else:
        base_ep = 'wss://testnet.binance.vision/ws/'
        stream_ep = 'wss://testnet.binance.vision/stream'

    if len(settings) == 1:
        feed_str = f"{base_ep}{settings[0][0]}"
    else:
        elements = [i[0] for i in settings]
        feed_str = f'{stream_ep}?streams=' + '/'.join(elements)

    return feed_str


def init_streams(streams_list, agents_dict, live):
    streams = {}
    for s in streams_list:
        max_lbs = []
        for agent in agents_dict.values():
            if s[0] == agent.stream:
                max_lbs.append(agent.max_lb)
        streams[s[0]] = Ohlc_Stream(s, max(max_lbs), live)

    return streams


# CONSTANTS
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12

# Settings
# agent_params = pd.read_pickle('settings.pkl').to_dict('records')

agent_params = [
    {'pair': 'btcusdt',
     'tf': '1m',
     'bias_lb': 200,  # long-term ema trend for bullish/bearish bias
     'bias_roc_lb': 8,  # lookback for judging if the long-term ema is moving up or down
     'source': 'vwap',  # timeseries source for trend_rate calculation
     'bars': 5,  # lookback for the ROC that is applied to the source in trend_rate
     'mult': 5,  # bars * mult gives the lookback window for finding the rolling mean and stdev of the ROC series
     'z': 1,  # z-score that the threshold is set to, to decide what is True or False in trend_rate
     'width': 3  # for a high/low to qualify as a williams fractal, it must be the highest/lowest of this many bars
     }
]

live = False

# Run Program
print(f"Started at {datetime.datetime.now().strftime('%d/%m/%y %H:%M')}\n")
agents = {f"{params['pair'].lower()}@kline_{params['tf']}_{x:02}": Agent(params, live) for x, params in enumerate(agent_params)}
# print('agents:')
# pprint(agents)

streams_list = list(set([(f"{agent['pair'].lower()}@kline_{agent['tf']}", agent['pair'].lower(), agent['tf']) for agent in agent_params]))
ws_feed = build_feed(streams_list, live)
print(ws_feed)

streams = init_streams(streams_list, agents, live)
# print('streams:')
# pprint(streams)


# streams = {s[0]: Ohlc_Stream(s, live) for s in streams_list}

ws = websocket.WebSocketApp(ws_feed, on_open=on_open, on_close=on_close,
                            on_message=on_message, on_ping=on_ping,
                            on_error=on_error)
ws.run_forever()

all_end = time.perf_counter()
elapsed = round(all_end - all_start)
print(f"Time taken: {elapsed//3600}h {(elapsed//60)%60}m {elapsed%60}s")

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
