# r4v02vevcore.py: vNN counts only inside the `vevcore` family (same round can have other r4v01… files with different tags).
# VEV ladder / voucher fair core.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

class Trader:

    def run(self, state: TradingState):
        result = {}

        atm = "VEV_5200"
        wings = ["VEV_6000", "VEV_6500"]

        if atm not in state.order_depths:
            return result, 0, ""

        atm_depth = state.order_depths[atm]
        best_bid_a = max(atm_depth.buy_orders.keys())
        best_ask_a = min(atm_depth.sell_orders.keys())
        mid_a = (best_bid_a + best_ask_a) / 2
        spread_a = best_ask_a - best_bid_a

        if spread_a > 3:
            return result, 0, ""

        orders = []

        for w in wings:
            if w not in state.order_depths:
                continue

            d = state.order_depths[w]
            best_bid_w = max(d.buy_orders.keys())
            best_ask_w = min(d.sell_orders.keys())
            mid_w = (best_bid_w + best_ask_w) / 2

            if mid_w > mid_a * 0.2:
                orders.append(Order(w, best_bid_w, -10))
                orders.append(Order(atm, best_ask_a, 10))

        result[atm] = orders
        return result, 0, ""