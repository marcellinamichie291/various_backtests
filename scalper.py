import pandas as pd
import numpy as np
from pathlib import Path
from pushbullet import Pushbullet
from decimal import Decimal, getcontext
import keys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import statistics as stats
from typing import Union, List, Tuple, Dict, Set, Optional, Any
import json
import threading
import time
from binance import Client
from pprint import pformat
from unicorn_binance_websocket_api.manager import BinanceWebSocketApiManager

# CONFIGURATION
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.precision', 3)

# CONSTANTS
pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')
ctx = getcontext()
ctx.prec = 12
TEST_API_KEY = '<your API Key>'
TEST_API_SECRET = '<your API secret>'
base_endpoint = 'https://api.binance.us' # Select the Binance API endpoint for your exchange
client = Client(api_key=TEST_API_KEY, api_secret=TEST_API_SECRET, testnet=True, tld='us')

# FUNCTIONS


def is_empty_message(message):
    if message is False:
        return True
    if '"result":null' in message:
        return True
    if '"result":None' in message:
        return True
    return False


def handle_price_change(symbol, timestamp, price):
    print(f"Handle price change for symbol: {symbol}, timestamp: {timestamp}, price: {price}")


def process_stream_data(binance_websocket_api_manager):
    while True:
        if binance_websocket_api_manager.is_manager_stopping():
            exit(0)
        oldest_data = binance_websocket_api_manager.pop_stream_data_from_stream_buffer()
        is_empty = is_empty_message(oldest_data)
        if is_empty:
            time.sleep(0.01)
        else:
            oldest_data_dict = json.loads(oldest_data)
            data = oldest_data_dict['data']
            #  Handle price change
            handle_price_change(symbol=data['s'], timestamp=data['T'], price=data['p'])


def start_websocket_listener():
    binance_us_websocket_api_manager = BinanceWebSocketApiManager(exchange="binance.us")
    channels = {'trade', }
    binance_us_websocket_api_manager.create_stream(channels, markets=lc_symbols, api_key=TEST_API_KEY, api_secret=TEST_API_SECRET)
    # Start a worker process to move the received stream_data from the stream_buffer to a print function
    worker_thread = threading.Thread(target=process_stream_data, args=(binance_us_websocket_api_manager,))
    worker_thread.start()


def compare_server_times():
    server_time = client.get_server_time()
    aa = str(server_time)
    bb = aa.replace("{'serverTime': ", "")
    aa = bb.replace("}", "")
    gg = int(aa)
    ff = gg - 10799260
    uu = ff / 1000
    yy = int(uu)
    tt = time.localtime(yy)
    print(f"Binance Server time: {tt}")
    print(f"Local time: {time.localtime()}")


#  Initialize binance client


#  Compare server and local times
compare_server_times()

start_websocket_listener()
