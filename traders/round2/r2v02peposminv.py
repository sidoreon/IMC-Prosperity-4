# r2v02peposminv.py: vNN counts only inside the `peposminv` family (same round can have other r2v01… files with different tags).
# Pepper (+ osmium); inventory / paired risk.
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
    EMA_ALPHA = 0.35
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

        fair = mid
        if len(history) >= self.MIN_HISTORY:
            ema = self._ema(history[-self.HISTORY_WINDOW:], self.EMA_ALPHA)
            fair = self.FAIR_BLEND * mid + (1 - self.FAIR_BLEND) * ema

        vol = self._stdev(history[-self.HISTORY_WINDOW:]) if history else 1.0
        z = (mid - fair) / max(vol, self.VOL_FLOOR)

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

            if best_ask <= fair - edge and buy_cap > 0:
                qty = min(-order_depth.sell_orders[best_ask], buy_cap, size)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, best_ask, qty))

            if best_bid >= fair + edge and sell_cap > 0:
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

        return orders, fair, z

class PepperRootStrategy:
    PRODUCT = "INTARIAN_PEPPER_ROOT"
    HARD_LIMIT = 80

    IMBALANCE_THRESHOLD = 0.5
    FRONT_RUN_TICKS = 1

    STOP_DRAWDOWN = 750

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def _market_trade_pressure(self, market_trades: List) -> int:
        net = 0
        for trade in market_trades:
            net += trade.quantity
        return net

    def run(
        self,
        order_depth: OrderDepth,
        position: int,
        market_trades: List,
        state_blob: Dict,
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        peak = state_blob.get("peak_mid", mid)
        if mid > peak:
            peak = mid
        state_blob["peak_mid"] = peak

        stopped = state_blob.get("stopped", False)
        if not stopped and (peak - mid) >= self.STOP_DRAWDOWN:
            stopped = True
            state_blob["stopped"] = True

        if stopped:

            if position <= 0:
                return []
            orders: List[Order] = []
            room = position
            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if room <= 0:
                    break
                qty = min(order_depth.buy_orders[bid_px], room)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, bid_px, -qty))
                    room -= qty
            if room > 0:
                orders.append(Order(self.PRODUCT, best_ask - 1, -room))
            return orders

        imbalance = self._imbalance(order_depth)
        if imbalance < self.IMBALANCE_THRESHOLD:
            return []

        pressure = self._market_trade_pressure(market_trades)
        if pressure < 0 and imbalance < 0.65:
            return []

        best_competing_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        front_run_bid = best_competing_bid + self.FRONT_RUN_TICKS

        orders = []
        room = self.HARD_LIMIT - position
        if room <= 0:
            return []

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if room <= 0:
                break
            available = abs(order_depth.sell_orders[ask_price])
            buy_size = min(available, room)
            orders.append(Order(self.PRODUCT, ask_price, buy_size))
            room -= buy_size

        if room > 0 and front_run_bid > 0:
            orders.append(Order(self.PRODUCT, front_run_bid, room))

        return orders

class Trader:
    def __init__(self):
        self.osmium = OsmiumStrategy()
        self.pepper = PepperRootStrategy()

    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        history = data.get("osmium_history", [])
        pepper_state = data.get("pepper", {})

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == OsmiumStrategy.PRODUCT:
                orders, _, _ = self.osmium.run(order_depth, position, history, state.timestamp)
                orders_by_product[product] = orders

            elif product == PepperRootStrategy.PRODUCT:
                orders_by_product[product] = self.pepper.run(
                    order_depth,
                    position,
                    state.market_trades.get(product, []),
                    pepper_state,
                )

            else:
                orders_by_product[product] = []

        trader_data = json.dumps({
            "osmium_history": history[-220:],
            "pepper": pepper_state,
        })

        return orders_by_product, 0, trader_data
