# r3v01hvoptcore.py: vNN counts only inside the `hvoptcore` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV; core fair + spread engine without extra cue.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import json

class Trader:

    LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
    }

    EDGE = 6
    SIZE = 20
    OFI_THRESHOLD = 0.2
    SKEW_K = 3

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            orders = []

            if product == "VELVETFRUIT_EXTRACT":
                orders = self.trade_ofi(depth, pos)

            elif product == "HYDROGEL_PACK":
                orders = self.passive_mm(depth, pos, product)

            result[product] = orders

        return result, 0, ""

    def compute_ofi(self, depth: OrderDepth):
        bid = sum(depth.buy_orders.values())
        ask = sum(abs(v) for v in depth.sell_orders.values())
        if bid + ask == 0:
            return 0
        return (bid - ask) / (bid + ask)

    def trade_ofi(self, depth: OrderDepth, pos: int):
        orders = []
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        ofi = self.compute_ofi(depth)
        skew = self.SKEW_K * ofi

        bid = int(mid - self.EDGE - skew)
        ask = int(mid + self.EDGE - skew)

        if ofi > self.OFI_THRESHOLD:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, 20))
        elif ofi < -self.OFI_THRESHOLD:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -20))

        orders.append(Order("VELVETFRUIT_EXTRACT", bid, 10))
        orders.append(Order("VELVETFRUIT_EXTRACT", ask, -10))

        return orders

    def passive_mm(self, depth, pos, product):
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        return [
            Order(product, int(mid - 8), 10),
            Order(product, int(mid + 8), -10)
        ]