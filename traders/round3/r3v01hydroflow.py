# r3v01hydroflow.py: vNN counts only inside the `hydroflow` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; flow / microprice tilt.
#
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState, Trade
except ModuleNotFoundError:
    from trader_factory.core.datamodel import Order, OrderDepth, TradingState, Trade

HYDRO = "HYDROGEL_PACK"

PRODUCT_LIMITS = {
    HYDRO: 200,
}

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0

def ema(prev: Optional[float], cur: float, alpha: float) -> float:
    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur

@dataclass
class SideQuote:
    price: int
    qty: int

class Book:
    def __init__(self, depth: Optional[OrderDepth]):
        self.valid = False
        self.buy_levels: List[Tuple[int, int]] = []
        self.sell_levels: List[Tuple[int, int]] = []
        self.best_bid = 0
        self.best_ask = 0
        self.best_bid_vol = 0
        self.best_ask_vol = 0
        self.mid = 0.0
        self.spread = 0.0
        self.micro = 0.0
        self.imbalance = 0.0
        self.top_depth = 0
        self.book_health = 0.0
        if depth is None:
            return
        self.buy_levels = sorted(
            [(int(p), int(v)) for p, v in depth.buy_orders.items() if int(v) > 0],
            key=lambda x: x[0],
            reverse=True,
        )
        self.sell_levels = sorted(
            [(int(p), abs(int(v))) for p, v in depth.sell_orders.items() if int(v) != 0],
            key=lambda x: x[0],
        )
        if not self.buy_levels or not self.sell_levels:
            return
        self.best_bid, self.best_bid_vol = self.buy_levels[0]
        self.best_ask, self.best_ask_vol = self.sell_levels[0]
        if self.best_bid >= self.best_ask:
            return
        self.mid = 0.5 * (self.best_bid + self.best_ask)
        self.spread = float(self.best_ask - self.best_bid)
        self.top_depth = self.best_bid_vol + self.best_ask_vol
        tot = max(1, self.best_bid_vol + self.best_ask_vol)
        self.micro = (self.best_ask * self.best_bid_vol + self.best_bid * self.best_ask_vol) / tot
        self.imbalance = (self.best_bid_vol - self.best_ask_vol) / tot
        health = 0.0
        health += 1.0
        if self.spread <= 12:
            health += 1.0
        elif self.spread <= 18:
            health += 0.5
        if self.top_depth >= 24:
            health += 1.0
        elif self.top_depth >= 14:
            health += 0.5
        if len(self.buy_levels) >= 2 and len(self.sell_levels) >= 2:
            health += 0.5
        self.book_health = clamp(health / 3.5, 0.0, 1.0)
        self.valid = True

    def stable_mid(self, top_n: int = 3) -> float:
        if not self.valid:
            return self.mid
        bid_levels = self.buy_levels[:top_n]
        ask_levels = self.sell_levels[:top_n]
        if not bid_levels or not ask_levels:
            return self.mid
        bid_vol = sum(v for _, v in bid_levels)
        ask_vol = sum(v for _, v in ask_levels)
        if bid_vol <= 0 or ask_vol <= 0:
            return self.mid
        popular_bid = sum(px * v for px, v in bid_levels) / bid_vol
        popular_ask = sum(px * v for px, v in ask_levels) / ask_vol

        wall_bid = max(bid_levels, key=lambda x: (x[1], x[0]))[0]
        wall_ask = min(ask_levels, key=lambda x: (-x[1], x[0]))[0]
        popular_mid = 0.5 * (popular_bid + popular_ask)
        wall_mid = 0.5 * (wall_bid + wall_ask)
        return 0.7 * popular_mid + 0.3 * wall_mid

class OrderManager:
    def __init__(self, product: str, start_pos: int, limit: int):
        self.product = product
        self.start_pos = int(start_pos)
        self.limit = int(limit)
        self.orders: List[Order] = []
        self.buy_used = 0
        self.sell_used = 0

    def projected_pos(self) -> int:
        return self.start_pos + self.buy_used - self.sell_used

    def buy_cap(self) -> int:
        return max(0, self.limit - self.projected_pos())

    def sell_cap(self) -> int:
        return max(0, self.limit + self.projected_pos())

    def buy(self, price: int, qty: int) -> None:
        qty = min(max(0, int(qty)), self.buy_cap())
        if qty > 0:
            self.orders.append(Order(self.product, int(price), qty))
            self.buy_used += qty

    def sell(self, price: int, qty: int) -> None:
        qty = min(max(0, int(qty)), self.sell_cap())
        if qty > 0:
            self.orders.append(Order(self.product, int(price), -qty))
            self.sell_used += qty

class Trader:
    def __init__(self) -> None:
        self.hydro_anchor = 10000.0

    def _load_memory(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            val = json.loads(trader_data)
            return val if isinstance(val, dict) else {}
        except Exception:
            return {}

    def _dump_memory(self, memory: dict) -> str:
        return json.dumps(memory, separators=(",", ":"))

    def _ensure_reset(self, memory: dict, timestamp: int) -> None:
        last_ts = int(memory.get("last_ts", -1))
        if last_ts >= 0 and timestamp < last_ts:
            memory.clear()
        memory["last_ts"] = int(timestamp)

    def _observe_position_change(self, memory: dict, state: TradingState) -> None:
        prev_pos = memory.get("prev_pos", {})
        new_prev: Dict[str, int] = {}
        own_flow = memory.setdefault("own_flow", {})
        ts = int(getattr(state, "timestamp", 0))
        for product, pos in state.position.items():
            pos = int(pos)
            old = int(prev_pos.get(product, pos))
            delta = pos - old
            flow = own_flow.setdefault(product, {"recent_turnover": 0.0, "last_trade_ts": -1, "last_signed_delta": 0})
            if delta != 0:
                flow["recent_turnover"] = float(flow.get("recent_turnover", 0.0)) * 0.7 + abs(delta)
                flow["last_trade_ts"] = ts
                flow["last_signed_delta"] = delta
            else:
                flow["recent_turnover"] = float(flow.get("recent_turnover", 0.0)) * 0.92
            new_prev[product] = pos
        memory["prev_pos"] = new_prev

    def _hydro_state(self, memory: dict, book: Book, position: int, timestamp: int) -> Tuple[dict, dict]:
        hs = memory.setdefault("hydro", {})
        last_mid = hs.get("last_mid")
        ret = 0.0 if last_mid is None else book.mid - float(last_mid)
        hs["ret_ema"] = ema(hs.get("ret_ema"), ret, 0.14)
        hs["vol_ema"] = ema(hs.get("vol_ema"), abs(ret), 0.10)
        hs["slow_mid"] = ema(hs.get("slow_mid"), book.mid, 0.05)
        hs["local_fair_ema"] = ema(hs.get("local_fair_ema"), 0.55 * book.stable_mid() + 0.30 * book.micro + 0.15 * self.hydro_anchor, 0.18)

        ret_ema = float(hs.get("ret_ema", 0.0))
        vol_ema = max(0.5, float(hs.get("vol_ema", 1.0)))
        slow_mid = float(hs.get("slow_mid", book.mid))
        local_fair = float(hs.get("local_fair_ema", book.mid))
        stable_mid = book.stable_mid()
        flow_score = 0.70 * ((book.micro - book.mid) / max(1.0, book.spread / 2.0)) + 0.30 * (4.0 * book.imbalance)
        trend_score = ret_ema / max(1.0, vol_ema)
        anchor_score = (self.hydro_anchor - book.mid) / 10.0
        local_score = (stable_mid - book.mid) / max(1.0, book.spread / 2.0)

        state = hs.get("state", "NEUTRAL")
        peak_trend = float(hs.get("peak_trend", 0.0))
        peak_mid = float(hs.get("peak_mid", book.mid))
        if state == "UP_TREND":
            peak_trend = max(peak_trend, trend_score)
            peak_mid = max(peak_mid, book.mid)
        elif state == "DOWN_TREND":
            peak_trend = min(peak_trend, trend_score)
            peak_mid = min(peak_mid, book.mid)
        else:
            peak_trend = trend_score
            peak_mid = book.mid

        draw_from_peak = 0.0
        if state == "UP_TREND":
            draw_from_peak = peak_mid - book.mid
        elif state == "DOWN_TREND":
            draw_from_peak = book.mid - peak_mid

        if state in ("NEUTRAL", "UP_UNWIND", "DOWN_UNWIND"):
            if trend_score > 1.10 and local_score > -0.25 and flow_score > -0.4:
                state = "UP_TREND"
                peak_trend = trend_score
                peak_mid = book.mid
            elif trend_score < -1.10 and local_score < 0.25 and flow_score < 0.4:
                state = "DOWN_TREND"
                peak_trend = trend_score
                peak_mid = book.mid
            else:
                state = "NEUTRAL"
        elif state == "UP_TREND":
            if trend_score < 0.55 or draw_from_peak > max(2.5, 2.0 * vol_ema):
                state = "UP_UNWIND"
        elif state == "DOWN_TREND":
            if trend_score > -0.55 or draw_from_peak > max(2.5, 2.0 * vol_ema):
                state = "DOWN_UNWIND"

        if state == "UP_UNWIND" and abs(position) <= 35 and trend_score < 0.9:
            state = "NEUTRAL"
        if state == "DOWN_UNWIND" and abs(position) <= 35 and trend_score > -0.9:
            state = "NEUTRAL"

        block_side = int(hs.get("block_side", 0))
        block_bars = int(hs.get("block_bars", 0))
        prev_state = hs.get("prev_state", state)
        if prev_state != state and state in ("UP_UNWIND", "DOWN_UNWIND"):
            block_side = 1 if state == "UP_UNWIND" else -1
            block_bars = 6
        else:
            block_bars = max(0, block_bars - 1)
            if block_bars == 0:
                block_side = 0

        progress = clamp(timestamp / 99900.0, 0.0, 1.0)
        if state == "UP_TREND":
            entry_cap = 160 if trend_score > 1.9 and book.book_health > 0.65 else 120
            hold_cap = 100 if abs(position) > 120 else entry_cap
            candidates = [0, 40, 80, 120, 160]
            signal = 0.85 * trend_score + 0.35 * flow_score + 0.20 * local_score
        elif state == "DOWN_TREND":
            entry_cap = 160 if trend_score < -1.9 and book.book_health > 0.65 else 120
            hold_cap = 100 if abs(position) > 120 else entry_cap
            candidates = [-160, -120, -80, -40, 0]
            signal = 0.85 * trend_score + 0.35 * flow_score + 0.20 * local_score
        elif state == "UP_UNWIND":
            entry_cap = 80
            hold_cap = 40
            candidates = [0, 20, 40, 60, 80]
            signal = 0.40 * trend_score + 0.50 * local_score + 0.20 * flow_score
        elif state == "DOWN_UNWIND":
            entry_cap = 80
            hold_cap = 40
            candidates = [-80, -60, -40, -20, 0]
            signal = 0.40 * trend_score + 0.50 * local_score + 0.20 * flow_score
        else:
            entry_cap = 60
            hold_cap = 40
            candidates = [-40, -20, 0, 20, 40]
            signal = 0.55 * local_score + 0.25 * flow_score + 0.15 * anchor_score / 4.0

        inv_ratio = abs(position) / float(PRODUCT_LIMITS[HYDRO])
        if progress > 0.80:
            hold_cap = min(hold_cap, 60)
        if progress > 0.92:
            hold_cap = min(hold_cap, 30)
        if abs(position) > 140:
            hold_cap = min(hold_cap, 25)

        best_target = 0
        best_value = -1e18
        for q in candidates:
            if abs(q) > entry_cap:
                continue

            if abs(position) > 100 and abs(q) > hold_cap and sign(q) == sign(position):
                continue

            if block_side != 0 and sign(q) == block_side and abs(q) > abs(position):
                continue
            alpha_term = signal * q
            inv_term = 0.006 * (q ** 2)
            turnover_term = 0.08 * abs(q - position)
            late_term = (0.004 + 0.012 * progress) * (abs(q) ** 1.25)
            danger_term = 0.0
            if abs(position) >= 150 and sign(q) == sign(position) and abs(q) >= abs(position):
                danger_term += 30.0
            value = alpha_term - inv_term - turnover_term - late_term - danger_term
            if value > best_value:
                best_value = value
                best_target = q

        fair = 0.45 * local_fair + 0.25 * stable_mid + 0.20 * slow_mid + 0.10 * self.hydro_anchor

        if state == "UP_TREND":
            fair += clamp(2.0 * trend_score, 0.0, 4.0)
        elif state == "DOWN_TREND":
            fair -= clamp(-2.0 * trend_score, 0.0, 4.0)

        ctx = {
            "state": state,
            "trend_score": trend_score,
            "flow_score": flow_score,
            "local_score": local_score,
            "fair": fair,
            "target": int(best_target),
            "book_health": book.book_health,
            "block_side": block_side,
            "block_bars": block_bars,
            "progress": progress,
            "vol_ema": vol_ema,
        }

        hs["last_mid"] = book.mid
        hs["peak_trend"] = peak_trend
        hs["peak_mid"] = peak_mid
        hs["state"] = state
        hs["prev_state"] = state
        hs["block_side"] = block_side
        hs["block_bars"] = block_bars
        memory["hydro"] = hs
        return memory, ctx

    def _trade_hydrogel(self, state: TradingState, memory: dict) -> List[Order]:
        book = Book(state.order_depths.get(HYDRO))
        if not book.valid:
            return []
        position = int(state.position.get(HYDRO, 0))
        ts = int(getattr(state, "timestamp", 0))
        memory, ctx = self._hydro_state(memory, book, position, ts)
        fair = float(ctx["fair"])
        target = int(ctx["target"])
        state_name = str(ctx["state"])
        block_side = int(ctx["block_side"])
        progress = float(ctx["progress"])
        mgr = OrderManager(HYDRO, position, PRODUCT_LIMITS[HYDRO])

        inv_gap = mgr.projected_pos() - target
        reservation = fair - 0.18 * inv_gap - 0.02 * inv_gap * abs(inv_gap) / 10.0

        abs_pos = abs(mgr.projected_pos())
        hard_danger = abs_pos >= 150
        max_danger = abs_pos >= 180
        unwind_mode = state_name in ("UP_UNWIND", "DOWN_UNWIND")

        buy_take_edge = 1.5
        sell_take_edge = 1.5
        if state_name == "UP_TREND":
            buy_take_edge -= 0.3
            sell_take_edge += 0.4
        elif state_name == "DOWN_TREND":
            sell_take_edge -= 0.3
            buy_take_edge += 0.4
        elif unwind_mode:
            buy_take_edge += 0.2
            sell_take_edge += 0.2

        if hard_danger:
            if position > 0:
                buy_take_edge += 1.5
                sell_take_edge -= 0.8
            elif position < 0:
                sell_take_edge += 1.5
                buy_take_edge -= 0.8

        buy_take_allowed = True
        sell_take_allowed = True
        if block_side == 1:
            buy_take_allowed = False
        elif block_side == -1:
            sell_take_allowed = False
        if position >= 150:
            buy_take_allowed = False
        if position <= -150:
            sell_take_allowed = False

        if buy_take_allowed and reservation - book.best_ask >= buy_take_edge:
            need = max(0, target - mgr.projected_pos())
            take = min(book.best_ask_vol, 18, max(6, need))
            mgr.buy(book.best_ask, take)
        if sell_take_allowed and book.best_bid - reservation >= sell_take_edge:
            need = max(0, mgr.projected_pos() - target)
            take = min(book.best_bid_vol, 18, max(6, need))
            mgr.sell(book.best_bid, take)

        rel_gap = mgr.projected_pos() - target
        clear_buy_edge = 0.8
        clear_sell_edge = 0.8
        if hard_danger:
            clear_buy_edge = 0.2
            clear_sell_edge = 0.2
        if max_danger:
            clear_buy_edge = -0.3
            clear_sell_edge = -0.3

        if rel_gap > 35 or (position > 130 and state_name != "UP_TREND"):
            if book.best_bid >= reservation - clear_sell_edge:
                size = min(book.best_bid_vol, 34 if hard_danger else 20, max(8, rel_gap))
                mgr.sell(book.best_bid, size)
        if rel_gap < -35 or (position < -130 and state_name != "DOWN_TREND"):
            if book.best_ask <= reservation + clear_buy_edge:
                size = min(book.best_ask_vol, 34 if hard_danger else 20, max(8, -rel_gap))
                mgr.buy(book.best_ask, size)

        buy_qe = 2.6
        sell_qe = 2.6
        if state_name == "UP_TREND":
            buy_qe -= 0.5
            sell_qe += 0.2
        elif state_name == "DOWN_TREND":
            sell_qe -= 0.5
            buy_qe += 0.2
        elif unwind_mode:
            buy_qe += 0.4
            sell_qe += 0.4
        if hard_danger:
            if position > 0:
                buy_qe += 1.5
                sell_qe -= 0.5
            elif position < 0:
                sell_qe += 1.5
                buy_qe -= 0.5
        if progress > 0.85:
            buy_qe += 0.4
            sell_qe += 0.4

        front_buy = int(math.floor(reservation - buy_qe))
        front_sell = int(math.ceil(reservation + sell_qe))
        front_buy = min(front_buy, book.best_ask - 1)
        front_sell = max(front_sell, book.best_bid + 1)
        back_buy = min(front_buy - 2, book.best_ask - 1)
        back_sell = max(front_sell + 2, book.best_bid + 1)

        allow_bid = True
        allow_ask = True
        if block_side == 1:
            allow_bid = False
        if block_side == -1:
            allow_ask = False
        if position >= 150:
            allow_bid = False
        if position <= -150:
            allow_ask = False

        qsize = 18
        if unwind_mode:
            qsize = 12
        if hard_danger:
            qsize = 8
        if book.book_health < 0.5:
            qsize = max(6, qsize - 4)

        if allow_bid and front_buy > 0:
            mgr.buy(front_buy, qsize)
            if back_buy > 0 and not hard_danger:
                mgr.buy(back_buy, max(4, qsize - 6))
        if allow_ask and front_sell > 0:
            mgr.sell(front_sell, qsize)
            if back_sell > 0 and not hard_danger:
                mgr.sell(back_sell, max(4, qsize - 6))

        return mgr.orders

    def run(self, state: TradingState):
        memory = self._load_memory(getattr(state, "traderData", ""))
        timestamp = int(getattr(state, "timestamp", 0))
        self._ensure_reset(memory, timestamp)
        self._observe_position_change(memory, state)

        result: Dict[str, List[Order]] = {}

        if HYDRO in state.order_depths:
            result[HYDRO] = self._trade_hydrogel(state, memory)

        return result, 0, self._dump_memory(memory)