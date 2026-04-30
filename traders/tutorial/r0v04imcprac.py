# r0v04imcprac.py: vNN counts only inside the `imcprac` family (round 0; other families reuse v01 independently).
# EMERALDS: dedicated tight quotes; TOMATOES: vol-adjusted spread and stronger alpha skew vs r0v03imcprac.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:

    def __init__(self):
        self.ewma = {}
        self.prev_ewma = {}

        # Parameters
        self.alpha = 0.1
        self.momentum_weight = 0.3
        self.inventory_penalty = 0.15

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

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())

            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            position = state.position.get(product, 0)
            limit = self.position_limit.get(product, 20)

            # =========================
            # SPECIAL: EMERALDS
            # =========================
            if product == "EMERALDS":

                buy_price = best_bid + 1
                sell_price = best_ask - 1

                if buy_price >= sell_price:
                    buy_price = best_bid
                    sell_price = best_ask

                size = 20

                if position < limit:
                    orders.append(Order(product, buy_price, min(size, limit - position)))

                if position > -limit:
                    orders.append(Order(product, sell_price, -min(size, position + limit)))

                result[product] = orders
                continue

            # =========================
            # SIGNAL (TOMATOES)
            # =========================
            fair = self.compute_ewma(product, mid)

            dev = mid - fair

            prev = self.prev_ewma.get(product, fair)
            momentum = fair - prev
            self.prev_ewma[product] = fair

            signal = -dev + self.momentum_weight * momentum

            # Inventory adjustment
            signal -= self.inventory_penalty * position

            # =========================
            # DYNAMIC SPREAD
            # =========================
            vol = abs(momentum) + abs(dev)
            spread_adjust = int(vol * 2)

            buy_price = best_bid + 1 - spread_adjust
            sell_price = best_ask - 1 + spread_adjust

            if buy_price >= sell_price:
                buy_price = best_bid
                sell_price = best_ask

            # =========================
            # ALPHA SKEW (FIXED)
            # =========================
            skew = signal * 2

            buy_price += int(skew)
            sell_price += int(skew)

            # =========================
            # ADAPTIVE SIZE
            # =========================
            base_size = 10 + int(abs(signal) * 5)

            buy_size = min(base_size, limit - position)
            sell_size = min(base_size, position + limit)

            # =========================
            # PASSIVE ORDERS
            # =========================
            if buy_size > 0:
                orders.append(Order(product, buy_price, buy_size))

            if sell_size > 0:
                orders.append(Order(product, sell_price, -sell_size))

            # =========================
            # SMART TAKING
            # =========================
            edge = abs(dev)

            if signal > 0 and edge > spread:
                take_size = min(5, limit - position)
                orders.append(Order(product, best_ask, take_size))

            elif signal < 0 and edge > spread:
                take_size = min(5, position + limit)
                orders.append(Order(product, best_bid, -take_size))

            result[product] = orders

        return result, 0, ""