# r3v01velvetvol.py: vNN counts only inside the `velvetvol` family (same round can have other r3v01… files with different tags).
# Velvetfruit-focused; vol-scaled edges.
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

VELVET = "VELVETFRUIT_EXTRACT"
VOUCHER_PREFIX = "VEV_"

PRODUCT_LIMITS = {
    VELVET: 200,
}

DEFAULT_VOUCHER_LIMIT = 300

TTE_DAYS = 5.0
YEAR_DAYS = 365.0
TTE_YEARS = TTE_DAYS / YEAR_DAYS

BASE_WORKING_VOUCHER_CAP = 110
MIDDLE_STRIKE_CAP = 85
MIDDLE_STRIKE_SET = {5100, 5200, 5300, 5400}
MIDDLE_CLUSTER_CAP = 240
ADJ_SAME_SIDE_CAP = 180
STRIP_DELTA_SOFT_CAP = 140.0
STRIP_DELTA_HARD_CAP = 180.0
UNCONFIRMED_OUTRIGHT_CAP = 45

ROUND3_MANUAL_BIOPOD_INFO = (
    "Manual Bio-Pod orders are not part of the submission bot. "
    "This file only handles exchange-traded products."
)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0

def ema(prev: Optional[float], cur: float, alpha: float) -> float:
    if prev is None:
        return cur
    return (1.0 - alpha) * prev + alpha * cur

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call_price(s: float, k: float, t: float, sigma: float) -> float:
    if s <= 0.0 or k <= 0.0:
        return 0.0
    if t <= 0.0:
        return max(s - k, 0.0)
    sigma = max(1e-6, sigma)
    intrinsic = max(s - k, 0.0)
    if intrinsic <= 0.0 and s < 1e-9:
        return 0.0
    st = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / st
    d2 = d1 - st
    return s * norm_cdf(d1) - k * norm_cdf(d2)

def bs_delta(s: float, k: float, t: float, sigma: float) -> float:
    if s <= 0.0 or k <= 0.0:
        return 0.0
    if t <= 0.0:
        return 1.0 if s > k else 0.0
    sigma = max(1e-6, sigma)
    st = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / st
    return norm_cdf(d1)

def bs_vega(s: float, k: float, t: float, sigma: float) -> float:
    if s <= 0.0 or k <= 0.0 or t <= 0.0:
        return 0.0
    sigma = max(1e-6, sigma)
    st = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / st
    return s * norm_pdf(d1) * math.sqrt(t)

def bs_vanna(s: float, k: float, t: float, sigma: float) -> float:
    if s <= 0.0 or k <= 0.0 or t <= 0.0:
        return 0.0
    sigma = max(1e-6, sigma)
    st = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + 0.5 * sigma * sigma * t) / st
    d2 = d1 - st

    return -norm_pdf(d1) * d2 / sigma

def implied_vol_call(price: float, s: float, k: float, t: float) -> Optional[float]:
    if s <= 0.0 or k <= 0.0:
        return None
    intrinsic = max(s - k, 0.0)
    if price < intrinsic - 1e-6:
        return None
    price = max(price, intrinsic)
    upper = s
    if price > upper + 1e-6:
        return None
    if t <= 0.0:
        return 1e-4

    lo, hi = 1e-4, 4.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        val = bs_call_price(s, k, t, mid)
        if val > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def solve_3x3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:

    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-10:
            return None
        if piv != i:
            m[i], m[piv] = m[piv], m[i]
        div = m[i][i]
        for j in range(i, n + 1):
            m[i][j] /= div
        for r in range(n):
            if r == i:
                continue
            factor = m[r][i]
            if abs(factor) < 1e-12:
                continue
            for j in range(i, n + 1):
                m[r][j] -= factor * m[i][j]
    return [m[i][n] for i in range(n)]

def weighted_quadratic_fit(xs: List[float], ys: List[float], ws: List[float]) -> Optional[Tuple[float, float, float, float]]:
    if len(xs) < 3:
        return None
    s00 = s01 = s02 = s03 = s04 = 0.0
    t0 = t1 = t2 = 0.0
    for x, y, w in zip(xs, ys, ws):
        w = max(1e-9, w)
        x2 = x * x
        s00 += w
        s01 += w * x
        s02 += w * x2
        s03 += w * x2 * x
        s04 += w * x2 * x2
        t0 += w * y
        t1 += w * x * y
        t2 += w * x2 * y
    mat = [
        [s00, s01, s02],
        [s01, s02, s03],
        [s02, s03, s04],
    ]
    rhs = [t0, t1, t2]
    sol = solve_3x3(mat, rhs)
    if sol is None:
        return None
    a, b, c = sol

    err_num = 0.0
    err_den = 0.0
    for x, y, w in zip(xs, ys, ws):
        yhat = a + b * x + c * x * x
        err_num += w * (y - yhat) ** 2
        err_den += w
    rmse = math.sqrt(err_num / max(err_den, 1e-9))
    return a, b, c, rmse

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
        self.velvet_anchor = 5250.0

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

    def _velvet_context(self, memory: dict, book: Book, timestamp: int) -> Tuple[dict, dict]:
        vs = memory.setdefault("velvet", {})
        last_mid = vs.get("last_mid")
        ret = 0.0 if last_mid is None else book.mid - float(last_mid)
        vs["ret_ema"] = ema(vs.get("ret_ema"), ret, 0.12)
        vs["vol_ema"] = ema(vs.get("vol_ema"), abs(ret), 0.08)
        fair_local = 0.45 * self.velvet_anchor + 0.35 * book.stable_mid() + 0.20 * book.micro
        vs["fair_ema"] = ema(vs.get("fair_ema"), fair_local, 0.16)
        vs["last_mid"] = book.mid
        memory["velvet"] = vs
        ctx = {
            "fair": float(vs.get("fair_ema", fair_local)),
            "ret_ema": float(vs.get("ret_ema", 0.0)),
            "vol_ema": max(0.5, float(vs.get("vol_ema", 1.0))),
            "book_health": book.book_health,
        }
        return memory, ctx

    def _voucher_universe(self, state: TradingState) -> List[Tuple[str, int]]:
        out = []
        for product in state.order_depths.keys():
            if product.startswith(VOUCHER_PREFIX):
                try:
                    strike = int(product.split("_")[1])
                    out.append((product, strike))
                except Exception:
                    continue
        out.sort(key=lambda x: x[1])
        return out

    def _extract_option_quotes(
        self,
        state: TradingState,
        vouchers: List[Tuple[str, int]],
        spot_fair: float,
    ) -> Dict[str, dict]:
        info: Dict[str, dict] = {}
        for product, strike in vouchers:
            book = Book(state.order_depths.get(product))
            if not book.valid:
                info[product] = {
                    "strike": strike,
                    "book": book,
                    "bid_px": None,
                    "ask_px": None,
                    "mid_px": None,
                    "bid_iv": None,
                    "ask_iv": None,
                    "mid_iv": None,
                    "liq": 0.0,
                }
                continue
            intrinsic = max(spot_fair - strike, 0.0)
            bid_px = max(float(book.best_bid), intrinsic)
            ask_px = max(float(book.best_ask), intrinsic)
            mid_px = 0.5 * (bid_px + ask_px)
            bid_iv = implied_vol_call(bid_px, spot_fair, strike, TTE_YEARS)
            ask_iv = implied_vol_call(ask_px, spot_fair, strike, TTE_YEARS)
            mid_iv = implied_vol_call(mid_px, spot_fair, strike, TTE_YEARS)
            info[product] = {
                "strike": strike,
                "book": book,
                "bid_px": bid_px,
                "ask_px": ask_px,
                "mid_px": mid_px,
                "bid_iv": bid_iv,
                "ask_iv": ask_iv,
                "mid_iv": mid_iv,
                "liq": float(book.top_depth),
            }
        return info

    def _repair_call_slice(self, strikes: List[int], prices: List[Optional[float]], spot_fair: float) -> List[Optional[float]]:
        repaired = [None if p is None else float(p) for p in prices]

        for i, k in enumerate(strikes):
            if repaired[i] is None:
                continue
            repaired[i] = clamp(repaired[i], max(spot_fair - k, 0.0), spot_fair)

        prev = None
        for i in range(len(strikes)):
            if repaired[i] is None:
                continue
            if prev is not None:
                repaired[i] = min(repaired[i], prev)
            prev = repaired[i]

        for _ in range(2):
            for i in range(1, len(strikes) - 1):
                if repaired[i - 1] is None or repaired[i] is None or repaired[i + 1] is None:
                    continue
                k1, k2, k3 = strikes[i - 1], strikes[i], strikes[i + 1]
                w = (k2 - k1) / max(1e-9, (k3 - k1))
                chord = repaired[i - 1] + w * (repaired[i + 1] - repaired[i - 1])
                repaired[i] = min(repaired[i], chord)
                repaired[i] = clamp(repaired[i], max(spot_fair - k2, 0.0), spot_fair)
            prev = None
            for i in range(len(strikes)):
                if repaired[i] is None:
                    continue
                if prev is not None:
                    repaired[i] = min(repaired[i], prev)
                prev = repaired[i]
        return repaired

    def _fit_surface(self, strikes: List[int], ivs: List[Optional[float]], liqs: List[float], spot_fair: float) -> dict:
        xs: List[float] = []
        ys: List[float] = []
        ws: List[float] = []
        med_vals: List[float] = []
        for k, iv, liq in zip(strikes, ivs, liqs):
            if iv is None or not (0.01 <= iv <= 4.0):
                continue
            x = math.log(max(1e-6, k / max(1e-6, spot_fair)))
            xs.append(x)
            ys.append(iv)
            ws.append(max(1.0, liq) / (1.0 + 6.0 * abs(x)))
            med_vals.append(iv)
        if len(ys) < 3:
            med = sum(med_vals) / max(1, len(med_vals)) if med_vals else 0.35
            return {
                "kind": "flat",
                "a": med,
                "b": 0.0,
                "c": 0.0,
                "rmse": 1.0,
                "confidence": 0.0,
            }
        fit = weighted_quadratic_fit(xs, ys, ws)
        if fit is None:
            med = sum(med_vals) / len(med_vals)
            return {
                "kind": "flat",
                "a": med,
                "b": 0.0,
                "c": 0.0,
                "rmse": 1.0,
                "confidence": 0.0,
            }
        a, b, c, rmse = fit
        confidence = clamp((len(ys) / 7.0) * (0.25 / max(0.25, rmse)), 0.0, 1.0)
        return {
            "kind": "quad",
            "a": a,
            "b": b,
            "c": c,
            "rmse": rmse,
            "confidence": confidence,
        }

    def _surface_eval(self, surface: dict, strike: int, spot_fair: float) -> float:
        x = math.log(max(1e-6, strike / max(1e-6, spot_fair)))
        if surface.get("kind") == "quad":
            iv = surface["a"] + surface["b"] * x + surface["c"] * x * x
        else:
            iv = surface["a"]
        return clamp(iv, 0.05, 3.0)

    def _build_strip_context(self, state: TradingState, memory: dict, velvet_fair: float) -> Tuple[dict, dict]:
        vouchers = self._voucher_universe(state)
        strikes = [k for _, k in vouchers]
        quote_info = self._extract_option_quotes(state, vouchers, velvet_fair)
        bid_prices = [quote_info[p]["bid_px"] for p, _ in vouchers]
        ask_prices = [quote_info[p]["ask_px"] for p, _ in vouchers]
        mid_prices = [quote_info[p]["mid_px"] for p, _ in vouchers]
        liqs = [quote_info[p]["liq"] for p, _ in vouchers]

        bid_prices = self._repair_call_slice(strikes, bid_prices, velvet_fair)
        ask_prices = self._repair_call_slice(strikes, ask_prices, velvet_fair)
        mid_prices = self._repair_call_slice(strikes, mid_prices, velvet_fair)

        bid_ivs = [implied_vol_call(px, velvet_fair, k, TTE_YEARS) if px is not None else None for px, k in zip(bid_prices, strikes)]
        ask_ivs = [implied_vol_call(px, velvet_fair, k, TTE_YEARS) if px is not None else None for px, k in zip(ask_prices, strikes)]
        mid_ivs = [implied_vol_call(px, velvet_fair, k, TTE_YEARS) if px is not None else None for px, k in zip(mid_prices, strikes)]

        surf_bid = self._fit_surface(strikes, bid_ivs, liqs, velvet_fair)
        surf_ask = self._fit_surface(strikes, ask_ivs, liqs, velvet_fair)
        surf_mid = self._fit_surface(strikes, mid_ivs, liqs, velvet_fair)

        local_iv_state = memory.setdefault("voucher_local_iv", {})
        data_by_product: Dict[str, dict] = {}
        residuals: Dict[str, float] = {}
        fair_price_mid: Dict[str, float] = {}
        fair_price_bid: Dict[str, float] = {}
        fair_price_ask: Dict[str, float] = {}

        usable_resids: List[float] = []
        pair_agreement_count = 0
        prev_resid: Optional[float] = None

        for (product, strike), mid_iv, bid_iv, ask_iv, liq in zip(vouchers, mid_ivs, bid_ivs, ask_ivs, liqs):
            mid_struct = self._surface_eval(surf_mid, strike, velvet_fair)
            bid_struct = self._surface_eval(surf_bid, strike, velvet_fair)
            ask_struct = self._surface_eval(surf_ask, strike, velvet_fair)

            local_prev = local_iv_state.get(product)
            if mid_iv is not None:
                local_now = ema(local_prev, mid_iv, 0.18)
            else:
                local_now = local_prev if local_prev is not None else mid_struct
            local_iv_state[product] = float(local_now)

            low_conf = surf_mid["confidence"] < 0.45
            blend_local = 0.45 if low_conf else 0.25
            hybrid_iv = (1.0 - blend_local) * mid_struct + blend_local * float(local_now)
            hybrid_iv = clamp(hybrid_iv, 0.05, 3.0)

            fair_mid_px = bs_call_price(velvet_fair, strike, TTE_YEARS, hybrid_iv)
            fair_bid_px = bs_call_price(velvet_fair, strike, TTE_YEARS, max(0.05, bid_struct))
            fair_ask_px = bs_call_price(velvet_fair, strike, TTE_YEARS, max(0.05, ask_struct))
            fair_price_mid[product] = fair_mid_px
            fair_price_bid[product] = fair_bid_px
            fair_price_ask[product] = fair_ask_px

            resid = 0.0
            if mid_iv is not None:
                resid = mid_iv - hybrid_iv
                residuals[product] = resid
                usable_resids.append(abs(resid))
                if prev_resid is not None and resid * prev_resid > 0 and abs(resid) > 0.04 and abs(prev_resid) > 0.04:
                    pair_agreement_count += 1
                prev_resid = resid

            data_by_product[product] = {
                "strike": strike,
                "bid_iv": bid_iv,
                "ask_iv": ask_iv,
                "mid_iv": mid_iv,
                "hybrid_iv": hybrid_iv,
                "fair_bid_px": fair_bid_px,
                "fair_ask_px": fair_ask_px,
                "fair_mid_px": fair_mid_px,
                "liq": liq,
                "book": quote_info[product]["book"],
            }

        avg_abs_resid = sum(usable_resids) / max(1, len(usable_resids)) if usable_resids else 0.0
        strip_state = memory.setdefault("strip_state", {})
        prev_avg = float(strip_state.get("avg_abs_resid", avg_abs_resid))
        resid_compression = prev_avg - avg_abs_resid
        strip_state["avg_abs_resid"] = avg_abs_resid
        memory["strip_state"] = strip_state
        broad_dislocation = avg_abs_resid > 0.085 and pair_agreement_count >= 2

        pair_support: Dict[str, float] = {p: 0.0 for p, _ in vouchers}
        for idx in range(len(vouchers) - 1):
            p1, _ = vouchers[idx]
            p2, _ = vouchers[idx + 1]
            r1 = residuals.get(p1)
            r2 = residuals.get(p2)
            if r1 is None or r2 is None:
                continue
            if r1 < -0.04 and r2 > 0.04:
                pair_support[p1] += min(1.0, abs(r1 - r2) / 0.12)
                pair_support[p2] -= min(1.0, abs(r1 - r2) / 0.12)
            elif r1 > 0.04 and r2 < -0.04:
                pair_support[p1] -= min(1.0, abs(r1 - r2) / 0.12)
                pair_support[p2] += min(1.0, abs(r1 - r2) / 0.12)

        strip_delta = 0.0
        strip_vega = 0.0
        strip_vanna = 0.0
        positions = state.position
        for product, strike in vouchers:
            pos = int(positions.get(product, 0))
            hybrid_iv = data_by_product[product]["hybrid_iv"]
            delta = bs_delta(velvet_fair, strike, TTE_YEARS, hybrid_iv)
            vega = bs_vega(velvet_fair, strike, TTE_YEARS, hybrid_iv)
            vanna = bs_vanna(velvet_fair, strike, TTE_YEARS, hybrid_iv)
            strip_delta += pos * delta
            strip_vega += pos * vega
            strip_vanna += pos * vanna
            data_by_product[product]["delta"] = delta
            data_by_product[product]["vega"] = vega
            data_by_product[product]["vanna"] = vanna
            data_by_product[product]["residual"] = residuals.get(product, 0.0)
            data_by_product[product]["pair_support"] = pair_support.get(product, 0.0)

        middle_gross = sum(abs(int(positions.get(p, 0))) for p, k in vouchers if k in MIDDLE_STRIKE_SET)
        abs_strip_delta = abs(strip_delta)
        if abs_strip_delta < 20:
            hedge_ratio = 0.0
        elif broad_dislocation and pair_agreement_count >= 3:
            hedge_ratio = 0.75
        elif avg_abs_resid > 0.055:
            hedge_ratio = 0.50
        else:
            hedge_ratio = 0.25
        if resid_compression > 0.015:
            hedge_ratio *= 0.7
        hedge_target = clamp(-hedge_ratio * strip_delta, -100, 100)

        context = {
            "vouchers": vouchers,
            "data": data_by_product,
            "avg_abs_resid": avg_abs_resid,
            "pair_agreement_count": pair_agreement_count,
            "broad_dislocation": broad_dislocation,
            "resid_compression": resid_compression,
            "strip_delta": strip_delta,
            "strip_vega": strip_vega,
            "strip_vanna": strip_vanna,
            "middle_gross": middle_gross,
            "hedge_ratio": hedge_ratio,
            "hedge_target": hedge_target,
        }
        return memory, context

    def _trade_velvet(self, state: TradingState, vev_ctx: dict, strip_ctx: dict) -> List[Order]:
        book = Book(state.order_depths.get(VELVET))
        if not book.valid:
            return []
        position = int(state.position.get(VELVET, 0))
        fair = float(vev_ctx["fair"])
        strip_delta = float(strip_ctx.get("strip_delta", 0.0))
        broad_dislocation = bool(strip_ctx.get("broad_dislocation", False))
        resid_compression = float(strip_ctx.get("resid_compression", 0.0))
        hedge_target = float(strip_ctx.get("hedge_target", 0.0))

        alpha_signal = (fair - book.mid) / max(1.0, book.spread / 2.0)
        alpha_target = 0.0
        if abs(strip_delta) < 60 and vev_ctx["book_health"] > 0.45:
            alpha_target = clamp(30.0 * math.tanh(0.7 * alpha_signal), -40.0, 40.0)
        final_target = int(round(clamp(hedge_target + alpha_target, -120, 120)))

        mgr = OrderManager(VELVET, position, PRODUCT_LIMITS[VELVET])
        inv_gap = mgr.projected_pos() - final_target
        reservation = fair - 0.25 * inv_gap

        buy_take_edge = 1.3
        sell_take_edge = 1.3
        if broad_dislocation:
            buy_take_edge -= 0.15
            sell_take_edge -= 0.15
        if resid_compression > 0.015:

            buy_take_edge += 0.1
            sell_take_edge += 0.1

        if reservation - book.best_ask >= buy_take_edge:
            need = max(0, final_target - mgr.projected_pos())
            if need > 0:
                mgr.buy(book.best_ask, min(book.best_ask_vol, 18, max(6, need)))
        if book.best_bid - reservation >= sell_take_edge:
            need = max(0, mgr.projected_pos() - final_target)
            if need > 0:
                mgr.sell(book.best_bid, min(book.best_bid_vol, 18, max(6, need)))

        rel_gap = mgr.projected_pos() - final_target
        if rel_gap > 25 and book.best_bid >= reservation - 0.7:
            mgr.sell(book.best_bid, min(book.best_bid_vol, 16, max(6, rel_gap)))
        if rel_gap < -25 and book.best_ask <= reservation + 0.7:
            mgr.buy(book.best_ask, min(book.best_ask_vol, 16, max(6, -rel_gap)))

        buy_qe = 2.2 + 0.002 * max(0, mgr.projected_pos() - final_target)
        sell_qe = 2.2 + 0.002 * max(0, final_target - mgr.projected_pos())
        if broad_dislocation:
            buy_qe -= 0.1
            sell_qe -= 0.1
        front_buy = int(math.floor(reservation - buy_qe))
        front_sell = int(math.ceil(reservation + sell_qe))
        front_buy = min(front_buy, book.best_ask - 1)
        front_sell = max(front_sell, book.best_bid + 1)
        if mgr.projected_pos() < final_target + 60:
            mgr.buy(front_buy, 12)
        if mgr.projected_pos() > final_target - 60:
            mgr.sell(front_sell, 12)
        return mgr.orders

    def _voucher_cap(self, strike: int) -> int:
        return MIDDLE_STRIKE_CAP if strike in MIDDLE_STRIKE_SET else BASE_WORKING_VOUCHER_CAP

    def _trade_vouchers(self, state: TradingState, memory: dict, strip_ctx: dict) -> Dict[str, List[Order]]:
        orders: Dict[str, List[Order]] = {}
        vouchers: List[Tuple[str, int]] = strip_ctx["vouchers"]
        data: Dict[str, dict] = strip_ctx["data"]
        broad_dislocation = bool(strip_ctx["broad_dislocation"])
        pair_agree = int(strip_ctx["pair_agreement_count"])
        strip_delta = float(strip_ctx["strip_delta"])
        middle_gross = int(strip_ctx["middle_gross"])

        positions = {p: int(state.position.get(p, 0)) for p, _ in vouchers}

        for idx, (product, strike) in enumerate(vouchers):
            book = data[product]["book"]
            if not book.valid:
                orders[product] = []
                continue
            position = positions[product]
            mgr = OrderManager(product, position, DEFAULT_VOUCHER_LIMIT)

            resid = float(data[product].get("residual", 0.0))
            pair_support = float(data[product].get("pair_support", 0.0))
            liq = float(data[product].get("liq", 0.0))
            bid_iv = data[product].get("bid_iv")
            ask_iv = data[product].get("ask_iv")
            fair_bid_px = float(data[product]["fair_bid_px"])
            fair_ask_px = float(data[product]["fair_ask_px"])
            fair_mid_px = float(data[product]["fair_mid_px"])
            delta = float(data[product]["delta"])

            score = -resid + 0.6 * pair_support
            if broad_dislocation:
                score += 0.15 * sign(score)

            base_cap = self._voucher_cap(strike)
            if strike in MIDDLE_STRIKE_SET and abs(pair_support) < 0.4:
                base_cap = min(base_cap, 60)

            if middle_gross >= MIDDLE_CLUSTER_CAP and strike in MIDDLE_STRIKE_SET and sign(score) == sign(position):
                base_cap = min(base_cap, abs(position))
            if abs(strip_delta) > STRIP_DELTA_HARD_CAP and sign(score) == sign(strip_delta):
                base_cap = min(base_cap, abs(position))

            target = int(round(clamp(base_cap * math.tanh(1.6 * score), -base_cap, base_cap)))
            if abs(pair_support) < 0.25 and abs(target) > UNCONFIRMED_OUTRIGHT_CAP:
                target = int(clamp(target, -UNCONFIRMED_OUTRIGHT_CAP, UNCONFIRMED_OUTRIGHT_CAP))

            if idx > 0:
                prev_p, _ = vouchers[idx - 1]
                if sign(target) == sign(positions.get(prev_p, 0)) != 0:
                    if abs(target) + abs(positions.get(prev_p, 0)) > ADJ_SAME_SIDE_CAP:
                        target = int(sign(target) * max(0, ADJ_SAME_SIDE_CAP - abs(positions.get(prev_p, 0))))
            if idx + 1 < len(vouchers):
                next_p, _ = vouchers[idx + 1]
                if sign(target) == sign(positions.get(next_p, 0)) != 0:
                    if abs(target) + abs(positions.get(next_p, 0)) > ADJ_SAME_SIDE_CAP:
                        target = int(sign(target) * max(0, ADJ_SAME_SIDE_CAP - abs(positions.get(next_p, 0))))

            take_iv_edge = 0.040
            if broad_dislocation and pair_agree >= 2:
                take_iv_edge = 0.028

            buy_confirmed = ask_iv is not None and bid_iv is not None and ask_iv < (data[product]["hybrid_iv"] - take_iv_edge)
            sell_confirmed = bid_iv is not None and ask_iv is not None and bid_iv > (data[product]["hybrid_iv"] + take_iv_edge)
            pair_confirmed = abs(pair_support) > 0.35

            buy_allowed = buy_confirmed and (pair_confirmed or abs(score) > 1.2 or broad_dislocation)
            sell_allowed = sell_confirmed and (pair_confirmed or abs(score) > 1.2 or broad_dislocation)

            if buy_allowed and mgr.projected_pos() < target and book.best_ask <= fair_bid_px:
                need = max(0, target - mgr.projected_pos())
                take_sz = 12 if pair_confirmed else 8
                if broad_dislocation:
                    take_sz += 4
                mgr.buy(book.best_ask, min(book.best_ask_vol, take_sz, need))

            if sell_allowed and mgr.projected_pos() > target and book.best_bid >= fair_ask_px:
                need = max(0, mgr.projected_pos() - target)
                take_sz = 12 if pair_confirmed else 8
                if broad_dislocation:
                    take_sz += 4
                mgr.sell(book.best_bid, min(book.best_bid_vol, take_sz, need))

            gap = mgr.projected_pos() - target
            if gap > 18 and book.best_bid >= fair_mid_px - 0.6:
                mgr.sell(book.best_bid, min(book.best_bid_vol, 10, max(6, gap)))
            if gap < -18 and book.best_ask <= fair_mid_px + 0.6:
                mgr.buy(book.best_ask, min(book.best_ask_vol, 10, max(6, -gap)))

            buy_qe = 1.0
            sell_qe = 1.0
            if broad_dislocation and pair_confirmed:
                buy_qe -= 0.15
                sell_qe -= 0.15
            buy_px = min(int(math.floor(fair_bid_px - buy_qe)), book.best_ask - 1)
            sell_px = max(int(math.ceil(fair_ask_px + sell_qe)), book.best_bid + 1)
            if mgr.projected_pos() < target and buy_px > 0:
                mgr.buy(buy_px, 6 if pair_confirmed else 4)
            if mgr.projected_pos() > target and sell_px > 0:
                mgr.sell(sell_px, 6 if pair_confirmed else 4)

            orders[product] = mgr.orders
        return orders

    def run(self, state: TradingState):
        memory = self._load_memory(getattr(state, "traderData", ""))
        timestamp = int(getattr(state, "timestamp", 0))
        self._ensure_reset(memory, timestamp)
        self._observe_position_change(memory, state)

        result: Dict[str, List[Order]] = {}

        velvet_book = Book(state.order_depths.get(VELVET))
        if velvet_book.valid:
            memory, vev_ctx = self._velvet_context(memory, velvet_book, timestamp)
            velvet_fair = float(vev_ctx["fair"])
            memory, strip_ctx = self._build_strip_context(state, memory, velvet_fair)
            result[VELVET] = self._trade_velvet(state, vev_ctx, strip_ctx)
            voucher_orders = self._trade_vouchers(state, memory, strip_ctx)
            result.update(voucher_orders)
        else:
            if VELVET in state.order_depths:
                result[VELVET] = []
            for p in state.order_depths.keys():
                if p.startswith(VOUCHER_PREFIX):
                    result[p] = []

        return result, 0, self._dump_memory(memory)