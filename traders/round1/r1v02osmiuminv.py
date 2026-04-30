# r1v02osmiuminv.py: vNN counts only inside the `osmiuminv` family (same round can have other r1v01… files with different tags).
# Osmium book; inventory / penny / passive mix.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json
import math

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_WINDOW = 100
    MIN_HISTORY = 25
    EMA_ALPHA = 0.18
    FAIR_BLEND = 0.85

    TAKE_EDGE = 3
    STRONG_TAKE_EDGE = 3.5

    MIN_PENNY_SPREAD = 4
    PASSIVE_OFFSET = 1
    BASE_PASSIVE_SIZE = 12
    BIG_PASSIVE_SIZE = 16

    REDUCE_TRIGGER = 20
    REDUCE_SIZE = 3
    HARD_REDUCE_TRIGGER = 50
    HARD_REDUCE_SIZE = 15

    INVENTORY_SKEW = 0.06
    VOL_FLOOR = 1.0

    LOW_SIGNAL_Z = 0.3
    IMBALANCE_THRESHOLD = 0.2
    TIGHT_SPREAD = 4

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

    def run(self, order_depth: OrderDepth, position: int, history: List[float], timestamp: int, market_trades: list = []):

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0.0, 0.0

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        fair = mid
        if len(history) >= self.MIN_HISTORY:
            ema = self._ema(history[-self.HISTORY_WINDOW:], self.EMA_ALPHA)
            fair = self.FAIR_BLEND * mid + (1 - self.FAIR_BLEND) * ema

        vol = self._stdev(history[-self.HISTORY_WINDOW:]) if history else 1.0
        if len(history) >= self.MIN_HISTORY:
            rolling_mean = sum(history[-self.HISTORY_WINDOW:]) / min(len(history), self.HISTORY_WINDOW)
        else:
            rolling_mean = mid
        z = (mid - rolling_mean) / max(vol, self.VOL_FLOOR)

        trade_pressure = sum(t.quantity for t in market_trades)

        imbalance = self._imbalance(order_depth)

        buy_cap = max(0, self.POSITION_LIMIT - position)
        sell_cap = max(0, self.POSITION_LIMIT + position)

        orders: List[Order] = []

        fair_px = int(round(fair))

        if position < -self.HARD_REDUCE_TRIGGER:
            qty = min(-position, self.HARD_REDUCE_SIZE, buy_cap)
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_ask, qty))

        elif position < -self.REDUCE_TRIGGER and best_bid > fair:
            qty = min(-position, self.REDUCE_SIZE, buy_cap)
            if qty > 0:
                orders.append(Order(self.PRODUCT, min(fair_px, best_ask - 1), qty))

        if position > self.HARD_REDUCE_TRIGGER:
            qty = min(position, self.HARD_REDUCE_SIZE, sell_cap)
            if qty > 0:
                orders.append(Order(self.PRODUCT, best_bid, -qty))

        elif position > self.REDUCE_TRIGGER and best_ask < fair:
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

            if best_ask <= fair - edge and buy_cap > 0:
                confirmed_size = size + (4 if trade_pressure > 0 else 0)
                qty = min(-order_depth.sell_orders[best_ask], buy_cap, confirmed_size)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_ask, qty))

            if best_bid >= fair + edge and sell_cap > 0:
                confirmed_size = size + (4 if trade_pressure < 0 else 0)
                qty = min(order_depth.buy_orders[best_bid], sell_cap, confirmed_size)
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

        print(f"[OSMIUM] t={timestamp} mid={mid:.2f} fair={fair:.2f} mean={rolling_mean:.2f} z={z:.2f} pos={position} spread={spread} imb={imbalance:.2f} pressure={trade_pressure}")

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
                market_trades = state.market_trades.get(product, [])
                orders, fair, z = self.osmium.run(order_depth, position, history, state.timestamp, market_trades)
                orders_by_product[product] = orders
            else:
                orders_by_product[product] = []

        trader_data = json.dumps({
            "osmium_history": history[-220:]
        })

        return orders_by_product, 0, trader_data