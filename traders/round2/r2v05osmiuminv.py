# r2v05osmiuminv.py: vNN counts only inside the `osmiuminv` family (same round can have other r2v01… files with different tags).
# Osmium book; inventory / penny / passive mix.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json
import math

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_WINDOW = 220
    MIN_HISTORY = 25
    EMA_ALPHA = 0.18
    FAIR_BLEND = 0.85

    TAKE_EDGE = 2
    STRONG_TAKE_EDGE = 5

    MIN_PENNY_SPREAD = 4
    PASSIVE_OFFSET = 1
    BASE_PASSIVE_SIZE = 12
    BIG_PASSIVE_SIZE = 12

    REDUCE_TRIGGER = 30
    REDUCE_SIZE = 4

    INVENTORY_SKEW = 0.04
    VOL_FLOOR = 1.0

    LOW_SIGNAL_Z = 0.3
    IMBALANCE_THRESHOLD = 0.2
    TIGHT_SPREAD = 3

    @staticmethod
    def _ema(values: List[float], alpha: float) -> float:
        ema = values[0]
        for x in values[1:]:
            ema = alpha * x + (1 - alpha) * ema
        return ema

    @staticmethod
    def _stdev(values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return max(math.sqrt(var), 1.0)

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = -sum(order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def run(self, order_depth: OrderDepth, position: int, history: List[float], timestamp: int):

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0.0, 0.0

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        ema = mid
        if len(history) >= self.MIN_HISTORY:
            ema = self._ema(history[-self.HISTORY_WINDOW:], self.EMA_ALPHA)
        fair = self.FAIR_BLEND * mid + (1 - self.FAIR_BLEND) * ema

        vol = self._stdev(history[-self.HISTORY_WINDOW:]) if history else 1.0
        z = (mid - ema) / max(vol, self.VOL_FLOOR)

        imbalance = self._imbalance(order_depth)

        buy_cap = max(0, self.POSITION_LIMIT - position)
        sell_cap = max(0, self.POSITION_LIMIT + position)

        orders: List[Order] = []

        fair_px = int(round(fair))

        if position < -self.REDUCE_TRIGGER and best_bid > fair:
            qty = min(-position, self.REDUCE_SIZE, buy_cap)
            if qty > 0:
                orders.append(Order(self.PRODUCT, min(fair_px, best_ask - 1), qty))

        if position > self.REDUCE_TRIGGER and best_ask < fair:
            qty = min(position, self.REDUCE_SIZE, sell_cap)
            if qty > 0:
                orders.append(Order(self.PRODUCT, max(fair_px, best_bid + 1), -qty))

        aggressive_mode = spread < self.TIGHT_SPREAD

        dynamic_edge = max(1, int(vol / 2))

        if aggressive_mode or abs(z) > self.LOW_SIGNAL_Z:

            edge = min(self.TAKE_EDGE, dynamic_edge)

            size = 6
            if abs(z) > self.STRONG_TAKE_EDGE:
                size = 16

            if best_ask < fair and buy_cap > 0:
                qty = min(-order_depth.sell_orders[best_ask], buy_cap, size)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_ask, qty))

            if best_bid > fair and sell_cap > 0:
                qty = min(order_depth.buy_orders[best_bid], sell_cap, size)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_bid, -qty))

        if spread >= self.MIN_PENNY_SPREAD:

            skew = int(position * self.INVENTORY_SKEW)

            passive_size = self.BASE_PASSIVE_SIZE
            if abs(z) > self.STRONG_TAKE_EDGE:
                passive_size = self.BIG_PASSIVE_SIZE

            if imbalance > self.IMBALANCE_THRESHOLD:
                bid_price = best_bid + 1
                ask_price = best_ask - self.PASSIVE_OFFSET
            elif imbalance < -self.IMBALANCE_THRESHOLD:
                bid_price = best_bid + self.PASSIVE_OFFSET
                ask_price = best_ask - 1
            else:
                bid_price = best_bid + self.PASSIVE_OFFSET
                ask_price = best_ask - self.PASSIVE_OFFSET

            bid_price -= skew
            ask_price -= skew

            bid_price = min(bid_price, best_ask - 1)
            ask_price = max(ask_price, best_bid + 1)

            if abs(position) > self.REDUCE_TRIGGER:
                passive_size //= 2

            if buy_cap > 0:
                orders.append(Order(self.PRODUCT, bid_price, min(passive_size, buy_cap)))

            if sell_cap > 0:
                orders.append(Order(self.PRODUCT, ask_price, -min(passive_size, sell_cap)))

        history.append(mid)
        if len(history) > self.HISTORY_WINDOW:
            history.pop(0)

        print(f"[OSMIUM] t={timestamp} mid={mid:.2f} fair={fair:.2f} z={z:.2f} pos={position} spread={spread} imb={imbalance:.2f}")

        return orders, fair, z

class Trader:
    def __init__(self):
        self.osmium = OsmiumStrategy()

    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except:
            data = {}

        history = data.get("osmium_history", [])

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == "ASH_COATED_OSMIUM":
                orders, fair, z = self.osmium.run(order_depth, position, history, state.timestamp)
                orders_by_product[product] = orders
            else:
                orders_by_product[product] = []

        trader_data = json.dumps({
            "osmium_history": history[-220:]
        })

        return orders_by_product, 0, trader_data