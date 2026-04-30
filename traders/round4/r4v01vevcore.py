# r4v01vevcore.py: vNN counts only inside the `vevcore` family (same round can have other r4v01… files with different tags).
# VEV ladder / voucher fair core.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

class Trader:

    def run(self, state: TradingState):
        result = {}

        atm = "VEV_5200"
        wings = ["VEV_6000", "VEV_6500"]

        orders = []

        if atm in state.order_depths:
            d = state.order_depths[atm]
            best_ask = min(d.sell_orders.keys())
            orders.append(Order(atm, best_ask, 10))

        for w in wings:
            if w in state.order_depths:
                d = state.order_depths[w]
                best_bid = max(d.buy_orders.keys())
                orders.append(Order(w, best_bid, -10))

        result[atm] = orders

        return result, 0, ""