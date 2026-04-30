# r3v03hvoptinv.py: vNN counts only inside the `hvoptinv` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; inventory skew and reduce logic.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

class Trader:

    EDGE = 5
    SIZE = 20
    SKEW_K = 4
    INV_K = 0.05

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            orders = []

            if product == "VELVETFRUIT_EXTRACT":
                orders = self.trade(depth, pos)

            elif product == "HYDROGEL_PACK":
                orders = self.passive(depth, pos, product)

            result[product] = orders

        return result, 0, ""

    def compute_ofi(self, depth: OrderDepth):
        bid = sum(depth.buy_orders.values())
        ask = sum(abs(v) for v in depth.sell_orders.values())
        if bid + ask == 0:
            return 0
        return (bid - ask) / (bid + ask)

    def trade(self, depth: OrderDepth, pos: int):
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        ofi = self.compute_ofi(depth)

        skew = self.SKEW_K * ofi
        inv_penalty = self.INV_K * pos

        bid = int(mid - self.EDGE - skew - inv_penalty)
        ask = int(mid + self.EDGE - skew - inv_penalty)

        orders = [
            Order("VELVETFRUIT_EXTRACT", bid, 15),
            Order("VELVETFRUIT_EXTRACT", ask, -15),
        ]

        if ofi > 0.3:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, 10))
        elif ofi < -0.3:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -10))

        return orders

    def passive(self, depth, pos, product):
        best_bid = max(depth.buy_orders.keys())
        best_ask = min(depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        return [
            Order(product, int(mid - 8), 10),
            Order(product, int(mid + 8), -10),
        ]