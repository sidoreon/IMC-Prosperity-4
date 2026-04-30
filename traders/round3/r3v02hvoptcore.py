# r3v02hvoptcore.py: vNN counts only inside the `hvoptcore` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV; core fair + spread engine without extra cue.
#
from datamodel import OrderDepth, Order, TradingState

class Trader:

    SIZE = 15
    SKEW_K = 2.5
    INV_K = 0.03

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)

            if product == "VELVETFRUIT_EXTRACT":
                result[product] = self.trade_vf(depth, pos)
            elif product == "HYDROGEL_PACK":
                result[product] = self.trade_hg(depth, pos)
            else:
                result[product] = []

        return result, 0, ""

    def ofi(self, depth):
        bid = sum(depth.buy_orders.values())
        ask = sum(abs(v) for v in depth.sell_orders.values())
        return (bid - ask) / (bid + ask) if bid + ask else 0

    def trade_vf(self, depth, pos):
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        ofi = self.ofi(depth)
        skew = int(self.SKEW_K * ofi)
        inv = int(self.INV_K * pos)

        bid = best_bid + max(0, skew - inv)
        ask = best_ask - max(0, -skew + inv)

        orders = [
            Order("VELVETFRUIT_EXTRACT", bid, self.SIZE),
            Order("VELVETFRUIT_EXTRACT", ask, -self.SIZE)
        ]

        if ofi > 0.4:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, 8))
        elif ofi < -0.4:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -8))

        return orders

    def trade_hg(self, depth, pos):
        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)

        return [
            Order("HYDROGEL_PACK", best_bid, 10),
            Order("HYDROGEL_PACK", best_ask, -10)
        ]