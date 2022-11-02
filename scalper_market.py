import datetime
import keys
import pandas as pd
from binance import Client
from pushbullet import Pushbullet
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

pb = Pushbullet('o.H4ZkitbaJgqx9vxo5kL2MMwnlANcloxT')

class Market():
    """a class to represent the trading pair so that i can keep track of how many positions i have in each pair and
    therefore how many stop-loss orders. if i don't keep track of that, i could run into problems with the exchange
    sending back errors. this could also be used to keep track of other information like precision and size limits"""
    def __init__(self, pair):
        self.pair = pair
        self.max_orders = 5
        self.orders = 0