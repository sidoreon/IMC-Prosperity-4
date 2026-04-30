# r0v01imcprac.py: vNN counts only inside the `imcprac` family (round 0; other families reuse v01 independently).
# EWMA fair, momentum on the signal, inventory-aware bid/ask skew (IMC prosperity trader v1).
#
# =========================
# IMC PROSPERITY TRADER
# =========================

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def __init__(self):
        # EWMA storage
        self.ewma = {}
        self.prev_ewma = {}

        # Parameters
        self.alpha = 0.1
        self.momentum_weight = 0.5

        # Position limits (adjust if needed)
        self.position_limit = {
            "TOMATOES": 20,
            "EMERALDS": 20
        }

    # =========================
    # EWMA CALCULATION
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
    # MAIN RUN FUNCTION
    # =========================
    def run(self, state: TradingState):

        result = {}

        for product in state.order_depths:

            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            # Skip if no liquidity
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                result[product] = orders
                continue

            # Best prices
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            # =========================
            # SIGNALS
            # =========================

            # EWMA fair value
            fair = self.compute_ewma(product, mid_price)

            # Deviation (mean reversion)
            dev = mid_price - fair

            # Momentum (FIXED)
            prev = self.prev_ewma.get(product, fair)
            momentum = fair - prev
            self.prev_ewma[product] = fair

            # Combined signal
            signal = -dev + self.momentum_weight * momentum

            # =========================
            # POSITION MANAGEMENT
            # =========================

            position = state.position.get(product, 0)
            limit = self.position_limit.get(product, 20)

            # Cost-aware threshold
            threshold = spread / 2

            # =========================
            # TRADING LOGIC
            # =========================

            # BUY condition
            if signal > threshold and position < limit:
                volume = min(5, limit - position)
                orders.append(Order(product, best_ask, volume))

            # SELL condition
            elif signal < -threshold and position > -limit:
                volume = min(5, position + limit)
                orders.append(Order(product, best_bid, -volume))

            result[product] = orders

        # =========================
        # REQUIRED RETURN FORMAT
        # =========================
        return result, 0, ""