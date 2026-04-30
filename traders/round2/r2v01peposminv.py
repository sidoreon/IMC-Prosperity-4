# r2v01peposminv.py: vNN counts only inside the `peposminv` family (same round can have other r2v01… files with different tags).
# Pepper (+ osmium); inventory / paired risk.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    FAIR = 10000.5

    TAKE_BUY_PX   = 9999
    TAKE_SELL_PX  = 10002

    STRONG_BUY_PX  = 9999
    STRONG_SELL_PX = 10004

    PASSIVE_BID = 10000
    PASSIVE_ASK = 10001

    MAX_TAKE_SIZE     = 20
    STRONG_TAKE_SIZE  = POSITION_LIMIT
    PASSIVE_SIZE      = 15
    STRONG_PASSIVE    = 20

    SOFT_POS   = 50
    DANGER_POS = 55

    def run(self, order_depth: OrderDepth, position: int, timestamp: int) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        buy_cap  = self.POSITION_LIMIT - position
        sell_cap = self.POSITION_LIMIT + position

        orders: List[Order] = []

        if buy_cap > 0:
            for ask_px in sorted(order_depth.sell_orders.keys()):
                if ask_px > self.TAKE_BUY_PX:
                    break
                avail = -order_depth.sell_orders[ask_px]
                size_cap = (
                    self.STRONG_TAKE_SIZE if ask_px <= self.STRONG_BUY_PX
                    else self.MAX_TAKE_SIZE
                )
                qty = min(avail, buy_cap, size_cap)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, ask_px, qty))
                    buy_cap -= qty
                if buy_cap <= 0:
                    break

        if sell_cap > 0:
            for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_px < self.TAKE_SELL_PX:
                    break
                avail = order_depth.buy_orders[bid_px]
                size_cap = (
                    self.STRONG_TAKE_SIZE if bid_px >= self.STRONG_SELL_PX
                    else self.MAX_TAKE_SIZE
                )
                qty = min(avail, sell_cap, size_cap)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, bid_px, -qty))
                    sell_cap -= qty
                if sell_cap <= 0:
                    break

        bid_px = self.PASSIVE_BID
        ask_px = self.PASSIVE_ASK

        if bid_px >= best_ask:
            bid_px = best_ask - 1
        if ask_px <= best_bid:
            ask_px = best_bid + 1

        if position >= self.SOFT_POS:

            ask_px = min(ask_px, int(self.FAIR))
        elif position <= -self.SOFT_POS:

            bid_px = max(bid_px, int(self.FAIR) + 1)

        bid_size = self.PASSIVE_SIZE
        ask_size = self.PASSIVE_SIZE

        if position >= self.DANGER_POS:
            bid_size = 0
            ask_size = self.STRONG_PASSIVE
        elif position <= -self.DANGER_POS:
            ask_size = 0
            bid_size = self.STRONG_PASSIVE

        bid_size = min(bid_size, buy_cap)
        ask_size = min(ask_size, sell_cap)

        if bid_size > 0:
            orders.append(Order(self.PRODUCT, bid_px, bid_size))
        if ask_size > 0:
            orders.append(Order(self.PRODUCT, ask_px, -ask_size))

        return orders

class PepperRootStrategy:
    PRODUCT = "INTARIAN_PEPPER_ROOT"
    POSITION_LIMIT = 80

    STOP_DRAWDOWN = 750

    EARLY_FRACTION = 0.25

    EARLY_TICKS = 250_000

    def run(
        self,
        order_depth: OrderDepth,
        position: int,
        state_blob: Dict,
        timestamp: int,
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

        first_ts = state_blob.get("first_ts", timestamp)
        if "first_ts" not in state_blob:
            state_blob["first_ts"] = timestamp
        is_early = (timestamp - first_ts) < self.EARLY_TICKS

        orders: List[Order] = []
        buy_cap  = self.POSITION_LIMIT - position
        sell_cap = self.POSITION_LIMIT + position

        if not stopped:

            room = buy_cap
            ask_levels = sorted(order_depth.sell_orders.keys())
            level_cap = len(ask_levels) if is_early else min(2, len(ask_levels))

            for ask_px in ask_levels[:level_cap]:
                if room <= 0:
                    break
                avail = -order_depth.sell_orders[ask_px]
                qty = min(avail, room)
                if qty > 0:
                    orders.append(Order(self.PRODUCT, ask_px, qty))
                    room -= qty

            if room > 0:
                orders.append(Order(self.PRODUCT, best_bid + 1, room))

        else:

            if position > 0 and sell_cap > 0:
                room = position
                for bid_px in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if room <= 0:
                        break
                    avail = order_depth.buy_orders[bid_px]
                    qty = min(avail, room)
                    if qty > 0:
                        orders.append(Order(self.PRODUCT, bid_px, -qty))
                        room -= qty
                if room > 0:
                    orders.append(Order(self.PRODUCT, best_ask - 1, -room))

        return orders

class Trader:
    def __init__(self):
        self.osmium = OsmiumStrategy()
        self.pepper = PepperRootStrategy()

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        pepper_state = data.get("pepper", {})

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == OsmiumStrategy.PRODUCT:
                orders_by_product[product] = self.osmium.run(
                    order_depth, position, state.timestamp
                )
            elif product == PepperRootStrategy.PRODUCT:
                orders_by_product[product] = self.pepper.run(
                    order_depth, position, pepper_state, state.timestamp
                )
            else:
                orders_by_product[product] = []

        trader_data = json.dumps({"pepper": pepper_state})
        return orders_by_product, 0, trader_data
