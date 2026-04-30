# r3v03velvetcore.py: vNN counts only inside the `velvetcore` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; base stack.
#
from datamodel import OrderDepth, Order, TradingState

class Trader:

    BASE_SIZE = 20
    INV_K = 0.08

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            inv = int(self.INV_K * pos)

            edge = 2 if product == "VELVETFRUIT_EXTRACT" else 6

            bid = int(mid - edge - inv)
            ask = int(mid + edge - inv)

            size = int(self.BASE_SIZE * (1 + min(1, spread / 5)))

            orders = [
                Order(product, bid, size),
                Order(product, ask, -size)
            ]

            if spread > edge:
                orders.append(Order(product, best_ask, 6))
                orders.append(Order(product, best_bid, -6))

            result[product] = orders

        return result, 0, ""