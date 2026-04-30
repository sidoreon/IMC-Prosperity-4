# r4v01assetcore.py: vNN counts only inside the `assetcore` family (same round can have other r4v01… files with different tags).
# Cross-asset base core.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List

class Trader:

    EDGE = 6
    BASE_SIZE = 20
    INV_K = 0.1

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            inv_penalty = self.INV_K * pos

            bid = int(mid - self.EDGE - inv_penalty)
            ask = int(mid + self.EDGE - inv_penalty)

            size = int(self.BASE_SIZE * min(2, spread / 5))

            orders = [
                Order(product, bid, size),
                Order(product, ask, -size),
            ]

            if best_ask < mid - 2:
                orders.append(Order(product, best_ask, 10))
            if best_bid > mid + 2:
                orders.append(Order(product, best_bid, -10))

            result[product] = orders

        return result, 0, ""