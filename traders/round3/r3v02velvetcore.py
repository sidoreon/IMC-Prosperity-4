# r3v02velvetcore.py: vNN counts only inside the `velvetcore` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; base stack.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

class Trader:

    STRIKE = 5200
    THRESH = 5

    def run(self, state: TradingState):
        result = {}

        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return result, 0, ""

        u_depth = state.order_depths["VELVETFRUIT_EXTRACT"]
        best_bid = max(u_depth.buy_orders.keys())
        best_ask = min(u_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        prev = getattr(self, "prev_mid", mid)
        move = abs(mid - prev)
        self.prev_mid = mid

        option = "VEV_5200"
        if option not in state.order_depths:
            return result, 0, ""

        o_depth = state.order_depths[option]
        best_bid_o = max(o_depth.buy_orders.keys())
        best_ask_o = min(o_depth.sell_orders.keys())
        mid_o = (best_bid_o + best_ask_o) / 2

        orders = []

        if move > self.THRESH:

            model = max(0, mid - self.STRIKE)

            if mid_o < model:
                orders.append(Order(option, best_ask_o, 10))
                orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -6))

            elif mid_o > model:
                orders.append(Order(option, best_bid_o, -10))
                orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, 6))

        result[option] = orders
        return result, 0, ""