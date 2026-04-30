# r3v01velvetcore.py: vNN counts only inside the `velvetcore` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; base stack.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import math

class Trader:

    STRIKE = 5200

    def run(self, state: TradingState):
        result = {}

        underlying_depth = state.order_depths["VELVETFRUIT_EXTRACT"]
        best_bid = max(underlying_depth.buy_orders.keys())
        best_ask = min(underlying_depth.sell_orders.keys())
        S = (best_bid + best_ask) / 2

        option = "VEV_5200"
        depth = state.order_depths.get(option)

        if depth:
            best_bid_o = max(depth.buy_orders.keys())
            best_ask_o = min(depth.sell_orders.keys())
            mid_o = (best_bid_o + best_ask_o) / 2

            model = max(0, S - self.STRIKE) + 5

            orders = []

            if mid_o < model:
                orders.append(Order(option, best_ask_o, 10))
                orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -6))

            elif mid_o > model:
                orders.append(Order(option, best_bid_o, -10))
                orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, 6))

            result[option] = orders

        return result, 0, ""