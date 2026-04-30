# r1v09osmiuminv.py: vNN counts only inside the `osmiuminv` family (same round can have other r1v01… files with different tags).
# Osmium book; inventory / penny / passive mix.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json
import math

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_WINDOW = 150
    MIN_HISTORY = 25
    EMA_ALPHA = 0.25
    VOL_WINDOW_SIZE = 4
    FAIR_BLEND = 0.75

    TAKE_EDGE = 1
    STRONG_TAKE_EDGE = 3.0

    MIN_PENNY_SPREAD = 4
    PASSIVE_OFFSET = 2
    BASE_PASSIVE_SIZE = 18
    BIG_PASSIVE_SIZE = 18

    REDUCE_TRIGGER = 60
    REDUCE_SIZE = 3

    INVENTORY_SKEW = 0.02
    VOL_FLOOR = 1.0

    QUEUE_P85 = 20
    QUEUE_P95 = 27

    OU_BETA = -1.002487

    LAG1_AUTOCORR = -0.494354

    LOW_SIGNAL_Z = 0.6
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

    def run(self, order_depth: OrderDepth, position: int, history: List[float], timestamp: int):

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0.0, 0.0

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

        bid_qty = order_depth.buy_orders.get(best_bid, 0)
        ask_qty = abs(order_depth.sell_orders.get(best_ask, 0))

        fair = mid
        if len(history) >= self.MIN_HISTORY:
            ema = self._ema(history[-self.HISTORY_WINDOW:], self.EMA_ALPHA)
            fair = self.FAIR_BLEND * mid + (1 - self.FAIR_BLEND) * ema

        vol_window = history[-self.VOL_WINDOW_SIZE:] if history else [mid]
        vol = self._stdev(vol_window) if len(vol_window) >= 2 else 1.0
        rolling_mean = sum(vol_window) / len(vol_window)
        z = (mid - rolling_mean) / max(vol, self.VOL_FLOOR)

        if len(history) >= 1:
            last_move = mid - history[-1]
            predicted_next = self.LAG1_AUTOCORR * last_move
        else:
            predicted_next = 0.0
        autocorr_lean = int(round(predicted_next))

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

        expected_reversion = abs(self.OU_BETA * (mid - rolling_mean))
        ou_size_factor = expected_reversion / max(vol, 1.0)
        take_size = max(4, min(int(ou_size_factor * 10), 20))

        if aggressive_mode or abs(z) > self.LOW_SIGNAL_Z:

            edge = min(self.TAKE_EDGE, dynamic_edge)

            if best_ask <= fair - edge and buy_cap > 0:
                ts = min(int(take_size * 1.5), buy_cap) if ask_qty >= self.QUEUE_P95 else take_size
                qty = min(ask_qty, buy_cap, ts)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_ask, qty))

            if best_bid >= fair + edge and sell_cap > 0:
                ts = min(int(take_size * 1.5), sell_cap) if bid_qty >= self.QUEUE_P95 else take_size
                qty = min(bid_qty, sell_cap, ts)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_bid, -qty))

        if spread >= self.MIN_PENNY_SPREAD:

            skew = int(position * self.INVENTORY_SKEW)

            passive_size = self.BASE_PASSIVE_SIZE
            if abs(z) > self.STRONG_TAKE_EDGE:
                passive_size = self.BIG_PASSIVE_SIZE

            bid_price = best_bid + (2 if bid_qty >= self.QUEUE_P95 else 1 if bid_qty >= self.QUEUE_P85 else self.PASSIVE_OFFSET)
            ask_price = best_ask - (2 if ask_qty >= self.QUEUE_P95 else 1 if ask_qty >= self.QUEUE_P85 else self.PASSIVE_OFFSET)

            bid_price += autocorr_lean
            ask_price += autocorr_lean

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

        print(f"[OSMIUMv6] t={timestamp} mid={mid:.2f} fair={fair:.2f} z={z:.2f} ts={take_size} acl={autocorr_lean} pos={position} spread={spread}")

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
