# r0v02imcprac.py: vNN counts only inside the `imcprac` family (round 0; other families reuse v01 independently).
# Hybrid MM + alpha layer with inventory penalty on fills (IMC prosperity trader v2).
#
# =========================
# IMC PROSPERITY TRADER (HYBRID MM + ALPHA)
# =========================

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def __init__(self):
        # EWMA state
        self.ewma = {}
        self.prev_ewma = {}

        # Parameters
        self.alpha = 0.1
        self.momentum_weight = 0.3
        self.inventory_penalty = 0.1

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
    # MAIN
    # =========================
    def run(self, state: TradingState):

        result = {}

        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Skip if empty
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = orders
                continue

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # =========================
            # SIGNALS
            # =========================

            fair = self.compute_ewma(product, mid_price)

            dev = mid_price - fair

            prev = self.prev_ewma.get(product, fair)
            momentum = fair - prev
            self.prev_ewma[product] = fair

            # Combined alpha signal
            signal = -dev + self.momentum_weight * momentum

            # =========================
            # POSITION CONTROL
            # =========================

            position = state.position.get(product, 0)
            limit = self.position_limit.get(product, 20)

            # Inventory penalty (pushes you toward flat)
            signal -= self.inventory_penalty * position

            # =========================
            # MARKET MAKING LOGIC
            # =========================

            # Base quote width
            base_spread = max(1, spread // 2)

            # Alpha skew
            skew = int(signal)

            # Quotes
            buy_price = int(fair - base_spread + skew)
            sell_price = int(fair + base_spread + skew)

            # =========================
            # EXECUTION
            # =========================

            # Passive BUY
            if position < limit:
                volume = min(5, limit - position)
                orders.append(Order(product, buy_price, volume))

            # Passive SELL
            if position > -limit:
                volume = min(5, position + limit)
                orders.append(Order(product, sell_price, -volume))

            result[product] = orders

        return result, 0, ""