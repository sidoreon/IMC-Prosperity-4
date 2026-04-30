# r0v03imcprac.py: vNN counts only inside the `imcprac` family (round 0; other families reuse v01 independently).
# EWMA + momentum + inventory penalty 0.15; single quoting path for EMERALDS and TOMATOES.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def __init__(self):
        # EWMA
        self.ewma = {}
        self.prev_ewma = {}

        # Parameters
        self.alpha = 0.1
        self.momentum_weight = 0.3
        self.inventory_penalty = 0.15

        # Position limits
        self.position_limit = {
            "TOMATOES": 20,
            "EMERALDS": 20
        }

    # =========================
    # EWMA
    # =========================
    def compute_ewma(self, product, price):
        if product not in self.ewma:
            self.ewma[product] = price
        else:
            self.ewma[product] = (
                self.alpha * price + (1 - self.alpha) * self.ewma[product]
            )
        return self.ewma[product]

    # =========================
    # MAIN LOOP
    # =========================
    def run(self, state: TradingState):

        result = {}

        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            # Best prices
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # =========================
            # SIGNAL
            # =========================
            fair = self.compute_ewma(product, mid)

            dev = mid - fair

            prev = self.prev_ewma.get(product, fair)
            momentum = fair - prev
            self.prev_ewma[product] = fair

            signal = -dev + self.momentum_weight * momentum

            # =========================
            # INVENTORY
            # =========================
            position = state.position.get(product, 0)
            limit = self.position_limit.get(product, 20)

            signal -= self.inventory_penalty * position

            # =========================
            # AGGRESSIVE QUOTING
            # =========================

            # Base: compete at top of book
            buy_price = best_bid + 1
            sell_price = best_ask - 1

            # If spread is too tight, fall back
            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            # Apply alpha skew
            skew = int(signal)

            buy_price += skew
            sell_price += skew

            # =========================
            # SIZE LOGIC (IMPORTANT)
            # =========================

            base_size = 10  # much more aggressive than before

            buy_size = min(base_size, limit - position)
            sell_size = min(base_size, position + limit)

            # =========================
            # EXECUTION
            # =========================

            # Always quote both sides (market making)
            if buy_size > 0:
                orders.append(Order(product, buy_price, buy_size))

            if sell_size > 0:
                orders.append(Order(product, sell_price, -sell_size))

            # =========================
            # OPTIONAL: TAKE WHEN EDGE IS BIG
            # =========================

            # If strong signal, cross spread intentionally
            if signal > spread:
                take_size = min(5, limit - position)
                orders.append(Order(product, best_ask, take_size))

            elif signal < -spread:
                take_size = min(5, position + limit)
                orders.append(Order(product, best_bid, -take_size))

            result[product] = orders

        return result, 0, ""