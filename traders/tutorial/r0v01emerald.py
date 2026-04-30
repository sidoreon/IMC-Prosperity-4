# r0v01emerald.py: vNN counts only inside the `emerald` family (round 0; other families reuse v01 independently).
# Light last-mid memory and simple two-sided quotes on EMERALDS + TOMATOES.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def __init__(self):
        self.position_limit = {
            "TOMATOES": 20,
            "EMERALDS": 20
        }

        # very light memory for tiny skew
        self.last_mid = {}

    def run(self, state: TradingState):

        result = {}

        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            position = state.position.get(product, 0)
            limit = self.position_limit.get(product, 20)

            # =========================
            # VERY LIGHT ALPHA (OPTIONAL)
            # =========================
            prev_mid = self.last_mid.get(product, mid)
            self.last_mid[product] = mid

            momentum = mid - prev_mid
            skew = int(momentum * 0.2)  # very small

            # =========================
            # TOP-OF-BOOK QUOTING
            # =========================

            buy_price = best_bid + 1
            sell_price = best_ask - 1

            # If spread collapses, fallback
            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            # Apply tiny skew
            buy_price += skew
            sell_price += skew

            # =========================
            # INVENTORY CONTROL
            # =========================

            # shift quotes based on position
            inv_skew = int(-0.3 * position)

            buy_price += inv_skew
            sell_price += inv_skew

            # =========================
            # AGGRESSIVE SIZE
            # =========================

            base_size = 20  # max participation

            buy_size = min(base_size, limit - position)
            sell_size = min(base_size, position + limit)

            # =========================
            # PLACE ORDERS
            # =========================

            if buy_size > 0:
                orders.append(Order(product, buy_price, buy_size))

            if sell_size > 0:
                orders.append(Order(product, sell_price, -sell_size))

            result[product] = orders

        return result, 0, ""