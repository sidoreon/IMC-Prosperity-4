# r3v01hydroinv.py: vNN counts only inside the `hydroinv` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; inventory / reduce first.
#
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import math
import json
from typing import Any, Optional

import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp])
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice, observation.askPrice, observation.transportFees,
                observation.exportTariff, observation.importTariff,
                observation.sugarPrice, observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            encoded_candidate = json.dumps(candidate)
            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out

logger = Logger()

TIMESTAMPS_PER_DAY = 1_000_000
TICKS_PER_DAY = 10_000

HP   = "HYDROGEL_PACK"

POSITION_LIMITS = {
    HP: 200,
}

LIMIT_HYDROGEL       = 200
AS_GAMMA             = 0.15
MICRO_DECAY          = 0.5
MICRO_SKEW_GAIN      = 0.7
HP_BASE_SPREAD       = 3.0
AS_VAR_WINDOW        = 50
VPIN_WINDOW          = 50
VPIN_TOXIC           = 0.70
VPIN_TOXIC_SPREAD_X  = 4.0

class Trader:

    def _clamp(self, v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def _ts_in_day(self, timestamp: int) -> int:
        # Return the tick position within the current day (0 – 999900).
        return timestamp % TIMESTAMPS_PER_DAY

    def _day_progress(self, timestamp: int) -> float:
        # Fraction of the current day elapsed: 0.0 (open) → 1.0 (close).
        return self._ts_in_day(timestamp) / TIMESTAMPS_PER_DAY

    def _restore_state(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}

    def _serialize_state(self, data: dict) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return "{}"

    def _kalman_update(self, data: dict, symbol: str, measurement: float,
                       Q: float = 0.01, R: float = 4.0) -> float:
        # 1-D Kalman update; returns updated state estimate.
        key_x = f"kf_x_{symbol}"
        key_p = f"kf_p_{symbol}"
        x = data.get(key_x, measurement)
        p = data.get(key_p, 100.0)
        p_pred = p + Q
        k = p_pred / (p_pred + R)
        x_new = x + k * (measurement - x)
        p_new = (1 - k) * p_pred
        data[key_x] = x_new
        data[key_p] = p_new
        return x_new

    def _obi(self, od: OrderDepth) -> float:
        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total = max(1, bid_vol + ask_vol)
        return (bid_vol - ask_vol) / total

    def _micro_price(self, od: OrderDepth, decay: float = MICRO_DECAY) -> tuple:
        # Return (micro, delta_signal) where micro is depth-weighted across the top 3 levels and delta_signal = micro - naive...
        if not od.buy_orders or not od.sell_orders:
            return 0.0, 0.0
        bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])[:3]
        asks = sorted(od.sell_orders.items(), key=lambda x:  x[0])[:3]
        levels = min(len(bids), len(asks))
        if levels == 0:
            return 0.0, 0.0
        num = 0.0
        den = 0.0
        for i in range(levels):
            w = math.exp(-decay * i)
            bp, bq = bids[i][0], abs(bids[i][1])
            ap, aq = asks[i][0], abs(asks[i][1])
            num += w * (aq * bp + bq * ap)
            den += w * (aq + bq)
        mid = 0.5 * (bids[0][0] + asks[0][0])
        if den <= 0:
            return mid, 0.0
        micro = num / den
        return micro, micro - mid

    def _update_as_variance(self, data: dict, symbol: str, micro: float,
                            window: int = AS_VAR_WINDOW) -> float:
        # Rolling variance of micro-price over `window` ticks. Used in the AS reservation: r = micro - pos*gamma*var/Q.
        key = f"as_micro_{symbol}"
        buf = data.get(key, [])
        buf.append(float(micro))
        if len(buf) > window:
            buf = buf[-window:]
        data[key] = buf
        if len(buf) < 5:
            return 1.0
        arr = np.asarray(buf, dtype=np.float64)
        return float(arr.var())

    def _update_vpin(self, data: dict, market_trades_by_sym: dict,
                     mid_by_sym: dict, window: int = VPIN_WINDOW) -> float:
        # Per-tick: classify each market trade as buy- or sell-initiated by comparing fill price to current mid; record per-t...
        vb = 0
        vs = 0
        for sym, trades in market_trades_by_sym.items():
            m = mid_by_sym.get(sym)
            if m is None or not trades:
                continue
            for t in trades:
                q = abs(t.quantity)
                if t.price >= m:
                    vb += q
                else:
                    vs += q
        tot = vb + vs
        imb = abs(vb - vs) / tot if tot > 0 else 0.0
        buf = data.get("vpin_buckets", [])
        buf.append(imb)
        if len(buf) > window:
            buf = buf[-window:]
        data["vpin_buckets"] = buf
        return (sum(buf) / len(buf)) if buf else 0.0

    def _update_price_window(self, data: dict, symbol: str, price: float,
                              window: int = 50) -> list:
        key = f"price_win_{symbol}"
        buf = data.get(key, [])
        buf.append(price)
        if len(buf) > window:
            buf = buf[-window:]
        data[key] = buf
        return buf

    def _aggressive_qty(self, deviation: float, threshold: float,
                        cap: int, scale: float = 1.0) -> int:
        # Map deviation depth beyond threshold to a buy/sell quantity.
        excess = abs(deviation) - threshold
        frac = self._clamp(excess / (threshold * 2.0), 0.0, 1.0)
        return max(1, int(cap * frac * scale))

    HP_FAIR        = 10_000
    HP_LIMIT       = 200
    HP_AGG_BUY     = -50
    HP_AGG_BUY_MAX = -70
    HP_AGG_SELL    =  50
    HP_AGG_SELL_MAX = 70
    HP_PASS_BUY    = -20
    HP_PASS_SELL   =  20

    def _hydrogel(self, od: OrderDepth, pos: int, timestamp: int, data: dict) -> list[Order]:
        orders: list[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        limit      = self.HP_LIMIT
        buy_cap    = max(0, limit - pos)
        sell_cap   = max(0, limit + pos)

        fair = self._kalman_update(data, HP, mid, Q=0.5, R=16.0)

        fair = 0.85 * self.HP_FAIR + 0.15 * fair
        deviation = mid - fair

        dp = self._day_progress(timestamp)

        is_warmup    = dp < 0.03

        if is_warmup:
            aggr_scale = 0.2
        else:
            aggr_scale = 1.0

        if deviation <= self.HP_AGG_BUY and buy_cap > 0 and not is_warmup:
            ultra = deviation <= self.HP_AGG_BUY_MAX
            qty = buy_cap if ultra else self._aggressive_qty(deviation, abs(self.HP_AGG_BUY), buy_cap, aggr_scale)
            asks_asc = sorted(od.sell_orders.keys())
            for ask in asks_asc:
                price_ok = ask <= fair + 2 if not ultra else ask <= best_ask + 1
                if price_ok and qty > 0:
                    fill = min(abs(od.sell_orders[ask]), qty)
                    if fill > 0:
                        orders.append(Order(HP, ask, fill))
                        qty -= fill
                        buy_cap -= fill

        if deviation >= self.HP_AGG_SELL and sell_cap > 0 and not is_warmup:
            ultra = deviation >= self.HP_AGG_SELL_MAX
            qty = sell_cap if ultra else self._aggressive_qty(deviation, self.HP_AGG_SELL, sell_cap, aggr_scale)
            bids_desc = sorted(od.buy_orders.keys(), reverse=True)
            for bid in bids_desc:
                price_ok = bid >= fair - 2 if not ultra else bid >= best_bid - 1
                if price_ok and qty > 0:
                    fill = min(od.buy_orders[bid], qty)
                    if fill > 0:
                        orders.append(Order(HP, bid, -fill))
                        qty -= fill
                        sell_cap -= fill

        micro, delta_signal = self._micro_price(od, decay=MICRO_DECAY)
        if micro <= 0.0:
            micro = mid

        as_var = self._update_as_variance(data, HP, micro)

        dynamic_gamma = 0.05 + 0.20 * ((abs(pos) / float(LIMIT_HYDROGEL)) ** 2)

        reservation_price = micro - (pos * dynamic_gamma * as_var / float(LIMIT_HYDROGEL))

        reservation_price += MICRO_SKEW_GAIN * delta_signal

        if deviation <= self.HP_PASS_BUY:
            reservation_price += 1.0
        elif deviation >= self.HP_PASS_SELL:
            reservation_price -= 1.0

        is_toxic = data.get("is_toxic", False)
        base_spread = HP_BASE_SPREAD
        if is_toxic:

            spread = base_spread * VPIN_TOXIC_SPREAD_X
            my_bid = min(int(round(reservation_price - spread)), best_bid)
            my_ask = max(int(round(reservation_price + spread)), best_ask)
        else:

            bid_skew = -delta_signal * 0.5
            ask_skew = delta_signal * 0.5

            my_bid = int(math.floor(reservation_price - (base_spread / 2.0) + bid_skew))
            my_ask = int(math.ceil(reservation_price + (base_spread / 2.0) + ask_skew))

            my_bid = min(my_bid, best_ask - 1)
            my_ask = max(my_ask, best_bid + 1)
            if best_bid + 1 < best_ask:
                my_bid = max(my_bid, best_bid + 1)
            if best_ask - 1 > best_bid:
                my_ask = min(my_ask, best_ask - 1)

        if my_bid >= my_ask:
            my_bid = my_ask - 1

        if buy_cap > 0:
            layer1 = max(1, int(buy_cap * 0.6))
            layer2 = buy_cap - layer1
            orders.append(Order(HP, my_bid, layer1))
            if layer2 > 0 and my_bid - 1 > 0:
                orders.append(Order(HP, my_bid - 1, layer2))

        if sell_cap > 0:
            layer1 = max(1, int(sell_cap * 0.6))
            layer2 = sell_cap - layer1
            orders.append(Order(HP, my_ask, -layer1))
            if layer2 > 0:
                orders.append(Order(HP, my_ask + 1, -layer2))

        return orders

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}
        data = self._restore_state(state.traderData)

        mid_by_sym: dict = {}
        for sym, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mid_by_sym[sym] = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0
        current_vpin = self._update_vpin(data, state.market_trades, mid_by_sym)
        data["current_vpin"] = current_vpin
        is_toxic = current_vpin > VPIN_TOXIC
        data["is_toxic"] = is_toxic

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)

            if product == HP:
                result[product] = self._hydrogel(od, pos, state.timestamp, data)

        trader_data = self._serialize_state(data)
        conversions = 0
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data