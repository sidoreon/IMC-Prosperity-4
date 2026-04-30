# r2v01osmiuminv.py: vNN counts only inside the `osmiuminv` family (same round can have other r2v01… files with different tags).
# Osmium book; inventory / penny / passive mix.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json
import math
from collections import deque

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    EMA_ALPHA = 0.72

    TAKE_Z = 0.6
    STRONG_TAKE_Z = 1.2

    PASSIVE_SIZE = 8
    AGGRESSIVE_SIZE = 12
    MAX_AGGRESSIVE = 16

    FAIR_BLEND = 0.95

    INVENTORY_SKEW = 0.06
    REDUCE_THRESHOLD = 50

    def __init__(self):
        self._ema = None

    def _update_ema(self, mid: float):
        if self._ema is None:
            self._ema = mid
        else:
            self._ema = self.EMA_ALPHA * mid + (1 - self.EMA_ALPHA) * self._ema
        return self._ema

    def _volatility(self, history: deque) -> float:
        # Return std dev of log-returns, not price levels.
        if len(history) < 5:
            return 0.00037
        log_rets = [math.log(history[i] / history[i-1])
                    for i in range(1, len(history))]
        mean = sum(log_rets) / len(log_rets)
        var = sum((r - mean) ** 2 for r in log_rets) / len(log_rets)
        return max(math.sqrt(var), 1e-8)

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def run(self, order_depth: OrderDepth, position: int,
            history: deque, timestamp: int):

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0.0, 0.0

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        half_spread = spread / 2

        ema = self._update_ema(mid)
        fair = self.FAIR_BLEND * mid + (1 - self.FAIR_BLEND) * ema

        vol = self._volatility(history)
        if vol > 0 and abs(mid - fair) > 0:
            z = (mid - fair) / (vol * mid)
        else:
            z = 0.0

        z = max(-4.0, min(4.0, z))

        imbalance = self._imbalance(order_depth)

        buy_cap = self.POSITION_LIMIT - position
        sell_cap = self.POSITION_LIMIT + position

        orders: List[Order] = []

        take_size = self.AGGRESSIVE_SIZE
        if abs(z) > self.STRONG_TAKE_Z:
            take_size = self.MAX_AGGRESSIVE

        if z < -self.TAKE_Z and buy_cap > 0:
            qty = min(abs(order_depth.sell_orders[best_ask]),
                      buy_cap, take_size)
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_ask, qty))

        if z > self.TAKE_Z and sell_cap > 0:
            qty = min(order_depth.buy_orders[best_bid],
                      sell_cap, take_size)
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_bid, -qty))

        reduce_size = max(4, abs(position) // 3)

        if position < -self.REDUCE_THRESHOLD and buy_cap > 0:

            qty = min(buy_cap, reduce_size,
                      abs(order_depth.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_ask, qty))

        if position > self.REDUCE_THRESHOLD and sell_cap > 0:

            qty = min(sell_cap, reduce_size,
                      order_depth.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_bid, -qty))

        if spread >= 2:
            skew_ticks = int(position * self.INVENTORY_SKEW)

            my_bid = best_bid + 1
            my_ask = best_ask - 1

            my_bid -= skew_ticks
            my_ask -= skew_ticks

            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)

            passive_size = self.PASSIVE_SIZE
            if abs(position) > 40:
                passive_size = max(2, passive_size // 2)

            if buy_cap > 0:
                orders.append(Order(self.PRODUCT, my_bid,
                                  min(passive_size, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(self.PRODUCT, my_ask,
                                  -min(passive_size, sell_cap)))

        history.append(mid)
        while len(history) > 10:
            history.popleft()

        print(f"[OSMIUM] t={timestamp} mid={mid:.1f} fair={fair:.1f} "
              f"z={z:.2f} pos={position} spread={spread} "
              f"imb={imbalance:.2f} n_orders={len(orders)}")

        return orders, fair, z

class Trader:
    def __init__(self):
        self.osmium = OsmiumStrategy()

    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except:
            data = {}

        history = deque(data.get("osmium_history", []), maxlen=10)

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == "ASH_COATED_OSMIUM":
                orders, fair, z = self.osmium.run(
                    order_depth, position, history, state.timestamp
                )
                orders_by_product[product] = orders
            else:
                orders_by_product[product] = []

        trader_data = json.dumps({
            "osmium_history": list(history)
        })

        return orders_by_product, 0, trader_data