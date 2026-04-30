# r2v01osmiumema.py: vNN counts only inside the `osmiumema` family (same round can have other r2v01… files with different tags).
# Osmium book; EMA fair.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json

PRODUCT = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80

EMA_ALPHA    = 0.10
MIN_HISTORY  = 20

PASSIVE_SIZE   = 18
PASSIVE_OFFSET = 5

TAKE_EDGE = 1

class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        ema   = data.get("ema",   None)
        ticks = data.get("ticks", 0)

        order_depth = state.order_depths.get(PRODUCT)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return {PRODUCT: []}, 0, json.dumps({"ema": ema, "ticks": ticks})

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2

        ema    = EMA_ALPHA * mid + (1 - EMA_ALPHA) * (ema if ema is not None else mid)
        ticks += 1
        fair   = ema if ticks >= MIN_HISTORY else mid

        position = state.position.get(PRODUCT, 0)
        buy_cap  = POSITION_LIMIT - position
        sell_cap = POSITION_LIMIT + position

        orders: List[Order] = []

        if best_ask <= fair - TAKE_EDGE and buy_cap > 0:
            qty = min(-order_depth.sell_orders[best_ask], buy_cap)
            orders.append(Order(PRODUCT, best_ask, qty))

        if best_bid >= fair + TAKE_EDGE and sell_cap > 0:
            qty = min(order_depth.buy_orders[best_bid], sell_cap)
            orders.append(Order(PRODUCT, best_bid, -qty))

        bid_px = min(int(fair) - PASSIVE_OFFSET, best_ask - 1)
        ask_px = max(int(fair) + PASSIVE_OFFSET, best_bid + 1)

        if buy_cap > 0:
            orders.append(Order(PRODUCT, bid_px, min(PASSIVE_SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order(PRODUCT, ask_px, -min(PASSIVE_SIZE, sell_cap)))

        trader_data = json.dumps({"ema": ema, "ticks": ticks})
        return {PRODUCT: orders}, 0, trader_data
