# r5v01assetvol.py: vNN counts only inside the `assetvol` family (same round can have other r5v01… files with different tags).
# Cross-asset volatility or edge scaling (non-HV-only basket).
#
import json
import math
from typing import Any

from datamodel import (
    Listing, Observation, Order, OrderDepth,
    ProsperityEncoder, Symbol, Trade, TradingState,
)

POS_LIMIT = 50

MID_WINDOW = 50

VOL_K = 1.0
VOL_TRIGGER_MULT = 2.0

DRIFT_K = 4.0

CFG: dict[str, dict] = {
    "TRANSLATOR_ECLIPSE_CHARCOAL": {"size": 6, "min_half": 2, "inv_skew": 4},
    "TRANSLATOR_ASTRO_BLACK":      {"size": 6, "min_half": 2, "inv_skew": 4},
    "ROBOT_LAUNDRY":               {"size": 6, "min_half": 2, "inv_skew": 4},
    "SNACKPACK_PISTACHIO":         {"size": 8, "min_half": 4, "inv_skew": 6},
    "SNACKPACK_RASPBERRY":         {"size": 8, "min_half": 4, "inv_skew": 6},
}

def microprice(od: OrderDepth) -> float | None:
    if not od.buy_orders or not od.sell_orders:
        return None
    best_bid = max(od.buy_orders.keys())
    best_ask = min(od.sell_orders.keys())
    bid_vol = od.buy_orders[best_bid]
    ask_vol = abs(od.sell_orders[best_ask])
    total = bid_vol + ask_vol
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_bid * ask_vol + best_ask * bid_vol) / total

class Trader:
    def bid(self) -> int:
        return 0

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        orders: dict[str, list[Order]] = {}

        try:
            ts_state = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, ValueError):
            ts_state = {}
        mid_buf: dict[str, list[float]] = ts_state.get("mid_buf", {})

        for sym, cfg in CFG.items():
            od = state.order_depths.get(sym)
            if od is None:
                continue
            fair = microprice(od)
            if fair is None:
                continue
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            book_spread = best_ask - best_bid
            if book_spread <= 0:
                continue

            position = state.position.get(sym, 0)

            inv_shift = cfg["inv_skew"] * position / POS_LIMIT
            skewed_fair = fair - inv_shift

            half = max(cfg["min_half"], book_spread / 4.0)

            our_bid_px = math.floor(skewed_fair - half)
            our_ask_px = math.ceil(skewed_fair + half)

            if our_ask_px <= our_bid_px:
                our_ask_px = our_bid_px + 1

            buy_capacity = POS_LIMIT - position
            sell_capacity = POS_LIMIT + position

            buy_qty = min(cfg["size"], max(0, buy_capacity))
            sell_qty = min(cfg["size"], max(0, sell_capacity))

            ords: list[Order] = []

            ask_take_qty = 0
            if best_ask <= skewed_fair - half and buy_qty > 0:
                avail = abs(od.sell_orders[best_ask])
                ask_take_qty = min(avail, buy_qty)
                if ask_take_qty > 0:
                    ords.append(Order(sym, best_ask, ask_take_qty))

            bid_take_qty = 0
            if best_bid >= skewed_fair + half and sell_qty > 0:
                avail = od.buy_orders[best_bid]
                bid_take_qty = min(avail, sell_qty)
                if bid_take_qty > 0:
                    ords.append(Order(sym, best_bid, -bid_take_qty))

            quote_buy = max(0, buy_qty - ask_take_qty)
            quote_sell = max(0, sell_qty - bid_take_qty)

            if quote_buy > 0:
                ords.append(Order(sym, our_bid_px, quote_buy))
            if quote_sell > 0:
                ords.append(Order(sym, our_ask_px, -quote_sell))

            if ords:
                orders[sym] = ords

        new_trader_data = json.dumps({"mid_buf": mid_buf}, separators=(",", ":"))
        logger.flush(state, orders, 0, new_trader_data)
        return orders, 0, new_trader_data

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders), conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders), conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state, trader_data):
        return [state.timestamp, trader_data, self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position, self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations):
        co = {}
        for product, obs in observations.conversionObservations.items():
            co[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                           obs.exportTariff, obs.importTariff,
                           obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, co]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value):
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value, max_length):
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()
