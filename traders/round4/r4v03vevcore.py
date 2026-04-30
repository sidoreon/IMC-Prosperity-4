# r4v03vevcore.py: vNN counts only inside the `vevcore` family (same round can have other r4v01… files with different tags).
# VEV ladder / voucher fair core.
#
from datamodel import OrderDepth, Order, TradingState

class Trader:

    def run(self, state: TradingState):
        result = {}

        atm = "VEV_5200"
        wings = ["VEV_6000", "VEV_6500"]

        if atm not in state.order_depths:
            return result, 0, ""

        atm_d = state.order_depths[atm]
        abid, aask = max(atm_d.buy_orders), min(atm_d.sell_orders)
        mid_a = (abid + aask) / 2

        orders = []

        for w in wings:
            if w not in state.order_depths:
                continue

            d = state.order_depths[w]
            wbid, wask = max(d.buy_orders), min(d.sell_orders)
            mid_w = (wbid + wask) / 2

            if mid_w > mid_a * 0.12:
                orders.append(Order(w, wbid, -6))
                orders.append(Order(atm, aask, 6))

        result[atm] = orders
        return result, 0, ""