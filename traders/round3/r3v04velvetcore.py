# r3v04velvetcore.py: vNN counts only inside the `velvetcore` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; base stack.
#
from datamodel import OrderDepth, Order, TradingState

class Trader:

    STRIKE = 5200

    def run(self, state: TradingState):
        result = {}

        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return result, 0, ""

        u = state.order_depths["VELVETFRUIT_EXTRACT"]
        ubid, uask = max(u.buy_orders), min(u.sell_orders)
        mid = (ubid + uask) / 2

        prev = getattr(self, "prev", mid)
        move = abs(mid - prev)
        self.prev = mid

        option = "VEV_5200"
        if option not in state.order_depths:
            return result, 0, ""

        d = state.order_depths[option]
        bid, ask = max(d.buy_orders), min(d.sell_orders)
        mid_o = (bid + ask) / 2

        model = max(0, mid - self.STRIKE)

        size = 4

        if move > 1:
            size = 10
        elif move > 0.5:
            size = 7

        orders = []

        if mid_o < model:
            orders.append(Order(option, ask, size))
            orders.append(Order("VELVETFRUIT_EXTRACT", ubid, -int(0.6 * size)))

        elif mid_o > model:
            orders.append(Order(option, bid, -size))
            orders.append(Order("VELVETFRUIT_EXTRACT", uask, int(0.6 * size)))

        result[option] = orders
        return result, 0, ""