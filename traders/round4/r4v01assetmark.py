# r4v01assetmark.py: vNN counts only inside the `assetmark` family (same round can have other r4v01… files with different tags).
# Cross-asset Mark-sensitive layer.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List
import json

class Trader:

    def classify(self, party):
        if party == "Mark 67":
            return "toxic"
        elif party in ["Mark 22", "Mark 49"]:
            return "alpha"
        return "neutral"

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            orders = []
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())

            for trade in state.market_trades.get(product, []):
                tag = self.classify(trade.buyer)

                if tag == "alpha":
                    orders.append(Order(product, best_ask, 20))

                elif tag == "toxic":

                    continue

            mid = (best_bid + best_ask) / 2
            orders.append(Order(product, int(mid - 6), 10))
            orders.append(Order(product, int(mid + 6), -10))

            result[product] = orders

        return result, 0, ""