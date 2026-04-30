# r3v01hvoptpair.py: vNN counts only inside the `hvoptpair` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV; paired or residual structure across books.
#
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState
except ModuleNotFoundError:
    from Bots.datamodel import Order, OrderDepth, TradingState

HYDRO = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
VOUCHERS: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
LIMITS: Dict[str, int] = {HYDRO: 200, VELVET: 200, **{p: 300 for p in VOUCHERS}}
MIDDLE_STRIKES = {5000, 5100, 5200, 5300, 5400}
BASE_STRIKE_CAPS: Dict[int, int] = {
    4000: 110,
    4500: 120,
    5000: 110,
    5100: 85,
    5200: 72,
    5300: 60,
    5400: 54,
    5500: 46,
    6000: 34,
    6500: 24,
}
TTE_YEARS = 5.0 / 365.0
MIDDLE_CLUSTER_CAP = 180
ADJ_SAME_SIDE_CAP = 180
UNPAIRED_MIDDLE_CAP = 45
STRIP_DELTA_SOFT_CAP = 145.0
STRIP_DELTA_HARD_CAP = 185.0
PAIR_DISTANCE_LIMIT = 2
PAIR_LIMIT_NORMAL = 5
PAIR_LIMIT_SHOCK = 7
DEEP_PAIR_PRODUCTS = {"VEV_4000", "VEV_4500"}
DEEP_PAIR_LATE_LATCH_TS = 80000
DEEP_PAIR_HARD_STOP_TS = 93000
DEEP_PAIR_COOLDOWN = 7000
MIN_IV = 0.02
MAX_IV = 2.80
ROOT_2PI = math.sqrt(2.0 * math.pi)

UNDERLYING_CFG = {
    HYDRO: {
        "anchor": 10000.0,
        "anchor_w": 0.54,
        "stable_w": 0.29,
        "micro_w": 0.17,
        "imbalance_w": 1.05,
        "take_edge": 2.2,
        "quote_edge": 3.0,
        "clear_edge": 1.0,
        "soft_limit": 100,
        "take_max": 22,
        "clear_max": 36,
        "quote_size": 24,
        "inv_skew": 8.0,
    }
}

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sign(x: float) -> int:
    if x > 1e-9:
        return 1
    if x < -1e-9:
        return -1
    return 0

def ema(prev: Optional[float], value: float, alpha: float) -> float:
    if prev is None:
        return float(value)
    return (1.0 - alpha) * float(prev) + alpha * float(value)

def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / ROOT_2PI

def load_memory(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        obj = json.loads(trader_data)
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

def dump_memory(memory: dict) -> str:
    return json.dumps(memory, separators=(",", ":"))

def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders) if od.buy_orders else None

def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders) if od.sell_orders else None

def raw_mid(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None or bb >= ba:
        return None
    return 0.5 * (bb + ba)

def bid_volume(od: OrderDepth, price: Optional[int]) -> int:
    if price is None:
        return 0
    return max(0, int(od.buy_orders.get(price, 0)))

def ask_volume(od: OrderDepth, price: Optional[int]) -> int:
    if price is None:
        return 0
    return abs(int(od.sell_orders.get(price, 0)))

def stable_mid(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is not None and ba is not None:
        return 0.5 * (bb + ba)
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None

def microprice(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return stable_mid(od)
    bv = bid_volume(od, bb)
    av = ask_volume(od, ba)
    if bv + av <= 0:
        return 0.5 * (bb + ba)
    return (ba * bv + bb * av) / (bv + av)

def book_imbalance(od: OrderDepth, levels: int = 2) -> float:
    bids = sorted(od.buy_orders.items(), reverse=True)[:levels]
    asks = sorted(od.sell_orders.items())[:levels]
    buy_depth = sum(max(0, int(volume)) for _, volume in bids)
    sell_depth = sum(abs(int(volume)) for _, volume in asks)
    total = buy_depth + sell_depth
    if total <= 0:
        return 0.0
    return (buy_depth - sell_depth) / total

def book_liquidity(od: OrderDepth) -> float:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None and ba is None:
        return 0.0
    total = 0.0
    for price, volume in od.buy_orders.items():
        total += max(0, int(volume))
    for price, volume in od.sell_orders.items():
        total += abs(int(volume))
    spread = 1.0
    if bb is not None and ba is not None:
        spread = max(1.0, float(ba - bb))
    return total / spread

def bs_call_price(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    intrinsic = max(spot - strike, 0.0)
    if tte <= 0.0 or sigma <= 1e-8:
        return intrinsic
    sqrt_t = math.sqrt(tte)
    vol_t = sigma * sqrt_t
    if vol_t <= 1e-12:
        return intrinsic
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / vol_t
    d2 = d1 - vol_t
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)

def bs_delta(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0.0 or strike <= 0.0:
        return 0.0
    if tte <= 0.0 or sigma <= 1e-8:
        return 1.0 if spot > strike else 0.0
    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / (sigma * sqrt_t)
    return norm_cdf(d1)

def bs_vega(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0.0 or strike <= 0.0 or tte <= 0.0 or sigma <= 1e-8:
        return 0.0
    sqrt_t = math.sqrt(tte)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / (sigma * sqrt_t)
    return spot * sqrt_t * norm_pdf(d1)

def implied_vol_call(spot: float, strike: float, tte: float, price: float) -> Optional[float]:
    if spot <= 0.0 or strike <= 0.0 or tte <= 0.0:
        return None
    intrinsic = max(spot - strike, 0.0)
    if price <= intrinsic + 1e-6:
        return MIN_IV
    upper_bound = spot
    if price >= upper_bound:
        return None
    lo, hi = MIN_IV, MAX_IV
    plo = bs_call_price(spot, strike, tte, lo)
    phi = bs_call_price(spot, strike, tte, hi)
    if price < plo - 1e-6:
        return lo
    if price > phi + 1e-6:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pm = bs_call_price(spot, strike, tte, mid)
        if abs(pm - price) <= 1e-6:
            return mid
        if pm > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def solve_3x3(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = 3
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, n + 1):
                m[r][j] -= factor * m[col][j]
    return [m[i][n] for i in range(n)]

def fit_weighted_quadratic(xs: Sequence[float], ys: Sequence[float], ws: Sequence[float]) -> Tuple[Tuple[float, float, float], float]:
    if not xs:
        return (0.0, 0.0, 0.0), 0.0
    if len(xs) == 1:
        return (ys[0], 0.0, 0.0), 0.0
    s0 = sum(ws)
    s1 = sum(w * x for x, w in zip(xs, ws))
    s2 = sum(w * x * x for x, w in zip(xs, ws))
    s3 = sum(w * x * x * x for x, w in zip(xs, ws))
    s4 = sum(w * x * x * x * x for x, w in zip(xs, ws))
    t0 = sum(w * y for y, w in zip(ys, ws))
    t1 = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    t2 = sum(w * x * x * y for x, y, w in zip(xs, ys, ws))
    coeffs = solve_3x3([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]], [t0, t1, t2])
    if coeffs is None:
        mean_y = sum(y * w for y, w in zip(ys, ws)) / max(1e-9, sum(ws))
        return (mean_y, 0.0, 0.0), 0.0
    a0, a1, a2 = coeffs
    err = 0.0
    weight = 0.0
    for x, y, w in zip(xs, ys, ws):
        fit = a0 + a1 * x + a2 * x * x
        err += w * (fit - y) * (fit - y)
        weight += w
    rmse = math.sqrt(err / max(weight, 1e-9))
    return (a0, a1, a2), rmse

def quad_eval(coeffs: Tuple[float, float, float], x: float) -> float:
    return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x

def rank_to_size(score: float, hard_cap: int) -> int:
    if score < 0.85:
        base = 8
    elif score < 1.30:
        base = 16
    elif score < 1.85:
        base = 26
    else:
        base = 38
    return min(hard_cap, base)

class OrderAccumulator:
    def __init__(self, product: str, position: int, limit: int):
        self.product = product
        self.position = int(position)
        self.limit = int(limit)
        self.buy_left = max(0, limit - position)
        self.sell_left = max(0, limit + position)
        self.orders: List[Order] = []

    def projected(self) -> int:
        return self.position + sum(order.quantity for order in self.orders)

    def buy(self, price: int, qty: int) -> None:
        size = min(max(0, int(qty)), self.buy_left)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), size))
        self.buy_left -= size

    def sell(self, price: int, qty: int) -> None:
        size = min(max(0, int(qty)), self.sell_left)
        if size <= 0:
            return
        self.orders.append(Order(self.product, int(price), -size))
        self.sell_left -= size

    def flush(self) -> List[Order]:
        return self.orders

class Trader:
    def _reset_if_new_day(self, memory: dict, timestamp: int) -> None:
        last_ts = int(memory.get("_last_timestamp", -1))
        if timestamp < last_ts:
            memory.clear()
        memory["_last_timestamp"] = timestamp
        memory.setdefault(HYDRO, {"ema_fair": None, "open_mid": None})
        memory.setdefault(VELVET, {"ema_fair": None, "open_mid": None})
        memory.setdefault(
            "voucher",
            {
                "local_iv_ema": {},
                "avg_abs_resid_ema": None,
                "peak_avg_abs_resid": 0.0,
                "mode": "PAIR",
                "active_deep_pair": None,
                "deep_pair_cooldowns": {},
            },
        )
        memory.setdefault(
            "hydro_state",
            {
                "ema_fast": None,
                "ema_slow": None,
                "ret_ema": 0.0,
                "last_mid": None,
                "long_peak_score": 0.0,
                "short_peak_score": 0.0,
                "long_peak_mid": None,
                "short_peak_mid": None,
                "long_peak_trend": 0.0,
                "short_peak_trend": 0.0,
                "exit_mode": "",
                "exit_age": 0,
                "extreme_side": 0,
                "extreme_hold_bars": 0,
            },
        )

    def _book_snapshot(self, od: OrderDepth) -> dict:
        bb = best_bid(od)
        ba = best_ask(od)
        mid = stable_mid(od)
        micro = microprice(od)
        spread = 999.0
        if bb is not None and ba is not None:
            spread = float(max(1, ba - bb))
        return {
            "best_bid": bb,
            "best_ask": ba,
            "mid": mid,
            "micro": micro if micro is not None else mid,
            "spread": spread,
            "liq": book_liquidity(od),
            "bid_size": bid_volume(od, bb),
            "ask_size": ask_volume(od, ba),
        }

    def _robust_underlying_fair(self, product: str, od: OrderDepth, memory: dict) -> float:
        snap = self._book_snapshot(od)
        state = memory[product]
        fallback = 10000.0 if product == HYDRO else 5250.0
        raw_mid = snap["mid"] if snap["mid"] is not None else snap["micro"]
        mid = float(raw_mid if raw_mid is not None else fallback)
        micro = float(snap["micro"] if snap["micro"] is not None else mid)
        if state["open_mid"] is None:
            state["open_mid"] = mid
        state["ema_fair"] = ema(state["ema_fair"], mid, 0.16 if product == HYDRO else 0.18)
        open_mid = float(state["open_mid"])
        if product == HYDRO:
            return 0.48 * mid + 0.28 * micro + 0.14 * float(state["ema_fair"]) + 0.10 * open_mid
        return 0.42 * mid + 0.30 * micro + 0.18 * float(state["ema_fair"]) + 0.10 * open_mid

    def _hydro_underlying_fair(self, od: OrderDepth) -> float:
        cfg = UNDERLYING_CFG[HYDRO]
        bb = best_bid(od)
        ba = best_ask(od)
        mid = raw_mid(od)
        if mid is None:
            mid = stable_mid(od)
        if mid is None:
            mid = cfg["anchor"]
        stable = stable_mid(od)
        if stable is None:
            stable = mid
        micro = microprice(od)
        if micro is None:
            micro = mid
        spread = float((ba - bb) if bb is not None and ba is not None else 8.0)
        fair = (
            cfg["anchor_w"] * cfg["anchor"]
            + cfg["stable_w"] * float(stable)
            + cfg["micro_w"] * float(micro)
        )
        fair += cfg["imbalance_w"] * book_imbalance(od, levels=2) * max(1.0, 0.5 * spread)
        return fair

    def _repair_call_slice(self, strikes: Sequence[int], prices: Sequence[Optional[float]], spot: float) -> List[Optional[float]]:
        repaired: List[Optional[float]] = []
        for strike, price in zip(strikes, prices):
            if price is None:
                repaired.append(None)
                continue
            intrinsic = max(spot - strike, 0.0) + 1e-4
            repaired.append(max(float(price), intrinsic))
        prev = None
        for idx, price in enumerate(repaired):
            if price is None:
                continue
            if prev is not None and price > prev:
                repaired[idx] = prev
            prev = repaired[idx]
        for _ in range(2):
            for idx in range(1, len(repaired) - 1):
                left = repaired[idx - 1]
                mid = repaired[idx]
                right = repaired[idx + 1]
                if left is None or mid is None or right is None:
                    continue
                repaired[idx] = min(mid, 0.5 * (left + right))
            prev = None
            for idx, price in enumerate(repaired):
                if price is None:
                    continue
                if prev is not None and price > prev:
                    repaired[idx] = prev
                prev = repaired[idx]
        return repaired

    def _build_voucher_surface(self, state: TradingState, spot_fair: float, memory: dict) -> Tuple[Dict[str, dict], dict]:
        voucher_memory = memory["voucher"]
        products = sorted(VOUCHERS, key=VOUCHERS.get)
        strikes = [VOUCHERS[p] for p in products]
        raw_prices: List[Optional[float]] = []
        raw_bid_prices: List[Optional[float]] = []
        raw_ask_prices: List[Optional[float]] = []
        xs: List[float] = []
        ys: List[float] = []
        ws: List[float] = []
        surface: Dict[str, dict] = {}
        suspicious_quotes = 0

        for product, strike in zip(products, strikes):
            od = state.order_depths.get(product)
            if od is None:
                raw_prices.append(None)
                raw_bid_prices.append(None)
                raw_ask_prices.append(None)
                continue
            snap = self._book_snapshot(od)
            bb = snap["best_bid"]
            ba = snap["best_ask"]
            mid = snap["mid"]
            if mid is None or bb is None or ba is None:
                raw_prices.append(None)
                raw_bid_prices.append(float(bb) if bb is not None else None)
                raw_ask_prices.append(float(ba) if ba is not None else None)
                suspicious_quotes += 1
                continue
            raw_prices.append(float(mid))
            raw_bid_prices.append(float(bb))
            raw_ask_prices.append(float(ba))
            m = math.log(strike / spot_fair)
            market_iv = implied_vol_call(spot_fair, strike, TTE_YEARS, float(mid))
            bid_iv = implied_vol_call(spot_fair, strike, TTE_YEARS, float(bb))
            ask_iv = implied_vol_call(spot_fair, strike, TTE_YEARS, float(ba))
            liquidity_quality = clamp(float(snap["liq"]) / 45.0, 0.0, 1.0)
            spread_quality = clamp(6.0 / max(1.0, float(snap["spread"])), 0.0, 1.0)
            weight = 0.40 + 0.35 * liquidity_quality + 0.25 * spread_quality
            surface[product] = {
                "strike": strike,
                "moneyness": m,
                "bid": int(bb),
                "ask": int(ba),
                "mid": float(mid),
                "spread": float(snap["spread"]),
                "liq": float(snap["liq"]),
                "market_iv": market_iv,
                "bid_iv": bid_iv,
                "ask_iv": ask_iv,
                "weight": weight,
            }
            if market_iv is not None:
                xs.append(m)
                ys.append(float(market_iv))
                ws.append(weight)

        repaired_prices = self._repair_call_slice(strikes, raw_prices, spot_fair)
        repaired_bid_prices = self._repair_call_slice(strikes, raw_bid_prices, spot_fair)
        repaired_ask_prices = self._repair_call_slice(strikes, raw_ask_prices, spot_fair)

        def build_fit(prices: Sequence[Optional[float]], fallback_coeffs: Optional[Tuple[float, float, float]] = None):
            fit_xs: List[float] = []
            fit_ys: List[float] = []
            fit_ws: List[float] = []
            for product, strike, rep_price in zip(products, strikes, prices):
                if rep_price is None:
                    continue
                rep_iv = implied_vol_call(spot_fair, strike, TTE_YEARS, rep_price)
                if rep_iv is None:
                    continue
                fit_xs.append(math.log(strike / spot_fair))
                fit_ys.append(rep_iv)
                fit_ws.append(float(surface.get(product, {}).get("weight", 0.8)))
            if not fit_ys and ys:
                fit_xs[:], fit_ys[:], fit_ws[:] = xs[:], ys[:], ws[:]
            if fit_ys:
                coeffs, rmse = fit_weighted_quadratic(fit_xs, fit_ys, fit_ws)
                return coeffs, rmse, len(fit_ys)
            if fallback_coeffs is not None:
                return fallback_coeffs, 0.0, 0
            return (0.0, 0.0, 0.0), 0.0, 0

        coeffs_mid, rmse, usable = build_fit(repaired_prices)
        coeffs_bid, _, usable_bid = build_fit(repaired_bid_prices, coeffs_mid)
        coeffs_ask, _, usable_ask = build_fit(repaired_ask_prices, coeffs_mid)
        stability = clamp(0.18 * usable + 0.55 - 7.5 * rmse - 0.04 * suspicious_quotes, 0.0, 1.0)

        struct_resids: List[float] = []
        for product, strike in VOUCHERS.items():
            if product not in surface:
                continue
            row = surface[product]
            struct_iv = clamp(quad_eval(coeffs_mid, row["moneyness"]), MIN_IV, MAX_IV)
            struct_bid_iv = clamp(quad_eval(coeffs_bid, row["moneyness"]), MIN_IV, MAX_IV)
            struct_ask_iv = clamp(quad_eval(coeffs_ask, row["moneyness"]), MIN_IV, MAX_IV)
            local_prev = voucher_memory["local_iv_ema"].get(product)
            market_iv = row["market_iv"]
            if market_iv is not None:
                local_now = ema(local_prev, float(market_iv), 0.18)
                voucher_memory["local_iv_ema"][product] = local_now
            else:
                local_now = local_prev if local_prev is not None else struct_iv
            row["struct_iv"] = struct_iv
            row["struct_bid_iv"] = struct_bid_iv
            row["struct_ask_iv"] = max(struct_ask_iv, struct_bid_iv)
            row["local_iv"] = float(local_now)
            row["hybrid_iv"] = 0.75 * struct_iv + 0.25 * float(local_now)
            row["fair_price"] = bs_call_price(spot_fair, strike, TTE_YEARS, row["hybrid_iv"])
            row["delta"] = bs_delta(spot_fair, strike, TTE_YEARS, row["hybrid_iv"])
            row["vega"] = bs_vega(spot_fair, strike, TTE_YEARS, row["hybrid_iv"])
            row["iv_residual"] = 0.0 if market_iv is None else float(market_iv) - struct_iv
            if market_iv is not None:
                struct_resids.append(abs(row["iv_residual"]))

        avg_struct_abs = safe_mean(struct_resids)
        avg_abs_ema = voucher_memory.get("avg_abs_resid_ema")
        voucher_memory["avg_abs_resid_ema"] = ema(avg_abs_ema, avg_struct_abs, 0.12)
        broad_hint = avg_struct_abs > max(0.0135, 1.06 * float(voucher_memory["avg_abs_resid_ema"] or avg_struct_abs))

        final_resids: List[float] = []
        pair_agreement_count = 0
        ordered = sorted(surface, key=lambda p: VOUCHERS[p])
        for idx, product in enumerate(ordered):
            row = surface[product]
            local_weight = 0.42 if (stability < 0.60 or broad_hint) else 0.22
            row["hybrid_iv"] = (1.0 - local_weight) * row["struct_iv"] + local_weight * row["local_iv"]
            row["fair_price"] = bs_call_price(spot_fair, row["strike"], TTE_YEARS, row["hybrid_iv"])
            row["delta"] = bs_delta(spot_fair, row["strike"], TTE_YEARS, row["hybrid_iv"])
            row["vega"] = bs_vega(spot_fair, row["strike"], TTE_YEARS, row["hybrid_iv"])
            side_width = max(0.003, 0.5 * (row["struct_ask_iv"] - row["struct_bid_iv"]))
            row["fair_bid_iv"] = clamp(row["hybrid_iv"] - side_width, MIN_IV, MAX_IV)
            row["fair_ask_iv"] = clamp(row["hybrid_iv"] + side_width, MIN_IV, MAX_IV)
            row["fair_bid_price"] = bs_call_price(spot_fair, row["strike"], TTE_YEARS, row["fair_bid_iv"])
            row["fair_ask_price"] = bs_call_price(spot_fair, row["strike"], TTE_YEARS, row["fair_ask_iv"])
            row["iv_residual"] = 0.0 if row["market_iv"] is None else float(row["market_iv"]) - row["hybrid_iv"]
            final_resids.append(abs(row["iv_residual"]))
            neighbor_signs = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(ordered):
                    other = surface[ordered[j]]
                    if abs(other["iv_residual"]) >= 0.010:
                        neighbor_signs.append(sign(other["iv_residual"]))
            own_sign = sign(row["iv_residual"])
            row["neighbor_confirmation"] = sum(1 for s in neighbor_signs if s == own_sign and s != 0) / 2.0

        for left, right in zip(ordered, ordered[1:]):
            lres = float(surface[left]["iv_residual"])
            rres = float(surface[right]["iv_residual"])
            if abs(lres) >= 0.012 and abs(rres) >= 0.012 and sign(lres) == sign(rres) and sign(lres) != 0:
                pair_agreement_count += 1

        avg_abs = safe_mean(final_resids)
        peak_abs = float(voucher_memory.get("peak_avg_abs_resid", 0.0))
        peak_abs = max(avg_abs, 0.995 * peak_abs)
        voucher_memory["peak_avg_abs_resid"] = peak_abs
        resid_compression = 0.0 if peak_abs <= 1e-9 else clamp(1.0 - avg_abs / peak_abs, 0.0, 1.0)
        broad_dislocation = (
            avg_abs > max(0.0135, 1.03 * float(voucher_memory["avg_abs_resid_ema"] or avg_abs))
            and pair_agreement_count >= 2
        )
        cheapest = sorted(ordered, key=lambda p: surface[p]["iv_residual"])[:3]
        richest = sorted(ordered, key=lambda p: surface[p]["iv_residual"], reverse=True)[:3]
        meta = {
            "stability": stability,
            "usable": usable,
            "usable_bid": usable_bid,
            "usable_ask": usable_ask,
            "rmse": rmse,
            "avg_abs_iv_residual": avg_abs,
            "pair_agreement_count": pair_agreement_count,
            "broad_dislocation": broad_dislocation,
            "resid_compression": resid_compression,
            "cheapest": cheapest,
            "richest": richest,
        }
        voucher_memory["mode"] = "SHOCK" if broad_dislocation else "PAIR"
        return surface, meta

    def _build_strip_context(self, state: TradingState, surface: Dict[str, dict], meta: dict) -> dict:
        current_delta = 0.0
        vega_proxy = 0.0
        middle_gross = 0
        adjacent_same_side_max = 0
        ordered = sorted(surface, key=lambda p: VOUCHERS[p])
        positions = {p: int(state.position.get(p, 0)) for p in ordered}
        for product in ordered:
            row = surface[product]
            pos = positions[product]
            current_delta += pos * float(row["delta"])
            vega_proxy += abs(pos) * float(row["vega"])
            if row["strike"] in MIDDLE_STRIKES:
                middle_gross += abs(pos)
        for left, right in zip(ordered, ordered[1:]):
            p1 = positions[left]
            p2 = positions[right]
            if sign(p1) != 0 and sign(p1) == sign(p2):
                adjacent_same_side_max = max(adjacent_same_side_max, abs(p1) + abs(p2))

        hedge_ratio = 0.75 if meta["broad_dislocation"] else (0.50 if meta["avg_abs_iv_residual"] > 0.0115 else 0.25)
        if meta["resid_compression"] > 0.22:
            hedge_ratio *= 0.45
        if meta["resid_compression"] > 0.42 or (state.timestamp > 85000 and meta["resid_compression"] > 0.28):
            hedge_ratio = 0.0
        if abs(current_delta) < 20.0:
            hedge_ratio = 0.0
        target_velvet = int(round(clamp(-hedge_ratio * current_delta, -120.0, 120.0)))
        velvet_pos = int(state.position.get(VELVET, 0))
        hedge_gap = abs(target_velvet - velvet_pos)
        hedge_feasible = clamp(
            1.0
            - 0.38 * abs(current_delta) / STRIP_DELTA_SOFT_CAP
            - 0.28 * middle_gross / MIDDLE_CLUSTER_CAP
            - 0.20 * hedge_gap / 120.0
            - 0.12 * adjacent_same_side_max / ADJ_SAME_SIDE_CAP,
            0.20,
            1.0,
        )
        return {
            "positions": positions,
            "strip_delta": current_delta,
            "strip_vega_proxy": vega_proxy,
            "middle_gross": middle_gross,
            "adjacent_same_side_max": adjacent_same_side_max,
            "hedge_ratio": hedge_ratio,
            "target_velvet": target_velvet,
            "hedge_feasible": hedge_feasible,
            "shock_mode": bool(meta["broad_dislocation"]),
        }

    def _build_hydro_context(self, state: TradingState, fair: float, memory: dict) -> dict:
        od = state.order_depths[HYDRO]
        bb = best_bid(od)
        ba = best_ask(od)
        mid = raw_mid(od)
        if mid is None:
            mid = stable_mid(od)
        if mid is None:
            mid = fair
        stable = stable_mid(od)
        if stable is None:
            stable = mid
        micro = microprice(od)
        if micro is None:
            micro = mid
        spread = float((ba - bb) if bb is not None and ba is not None else 8.0)
        top_depth = float(bid_volume(od, bb) + ask_volume(od, ba))
        hydro_state = memory["hydro_state"]

        prev_fast = hydro_state.get("ema_fast")
        prev_slow = hydro_state.get("ema_slow")
        prev_mid = hydro_state.get("last_mid")
        prev_ret_ema = float(hydro_state.get("ret_ema", 0.0))
        ema_fast = float(mid) if prev_fast is None else 0.16 * float(mid) + 0.84 * float(prev_fast)
        ema_slow = float(mid) if prev_slow is None else 0.035 * float(mid) + 0.965 * float(prev_slow)
        ret = 0.0 if prev_mid is None else float(mid) - float(prev_mid)
        ret_ema = 0.12 * ret + 0.88 * prev_ret_ema
        hydro_state["ema_fast"] = ema_fast
        hydro_state["ema_slow"] = ema_slow
        hydro_state["ret_ema"] = ret_ema
        hydro_state["last_mid"] = float(mid)

        good_book = (
            bb is not None
            and ba is not None
            and bb < ba
            and spread <= 18.0
            and top_depth >= 20.0
            and abs(float(stable) - float(mid)) <= 1.1
        )
        anchor_gap = float(mid) - UNDERLYING_CFG[HYDRO]["anchor"]
        trend_gap = ema_fast - ema_slow
        micro_gap = float(micro) - float(mid)
        imbalance = book_imbalance(od, levels=2)
        anchor_score = clamp(anchor_gap / 35.0, -3.0, 3.0)
        trend_score = clamp(trend_gap / 7.5, -3.0, 3.0)
        micro_score = clamp(micro_gap / max(1.0, 0.5 * spread), -1.5, 1.5)
        flow_score = clamp(micro_score + 0.75 * imbalance + 0.35 * clamp(ret_ema / 3.0, -2.0, 2.0), -2.0, 2.0)
        regime_score = 0.60 * trend_score + 0.30 * anchor_score + 0.10 * flow_score
        strength = abs(regime_score)

        if not good_book:
            entry_cap = 60
            confidence = "guarded"
        elif strength < 0.75:
            entry_cap = 0
            confidence = "neutral"
        elif strength < 1.30:
            entry_cap = 70
            confidence = "weak"
        elif strength < 2.00:
            entry_cap = 120
            confidence = "medium"
        else:
            entry_cap = 150
            confidence = "strong"

        progress = clamp(state.timestamp / 100000.0, 0.0, 1.0)
        pos = int(state.position.get(HYDRO, 0))
        if progress > 0.78:
            entry_cap = int(round(entry_cap * 0.84))
        if progress > 0.90:
            entry_cap = int(round(entry_cap * 0.58))
        if progress > 0.97:
            entry_cap = min(entry_cap, 50)

        hold_cap = entry_cap
        if abs(pos) > 120 and abs(trend_score) < 1.8:
            hold_cap = min(hold_cap, 100)
        if abs(pos) > 140 and abs(regime_score) < 1.1:
            hold_cap = min(hold_cap, 45)
        if progress > 0.94:
            hold_cap = min(hold_cap, 70)
        if progress > 0.98:
            hold_cap = min(hold_cap, 40)

        entry_target = int(round(clamp(200.0 * math.tanh(0.95 * regime_score), -entry_cap, entry_cap)))
        hold_target = int(round(clamp(200.0 * math.tanh(0.88 * (0.88 * regime_score + 0.12 * trend_score)), -hold_cap, hold_cap)))
        target = entry_target if abs(pos) < 110 else hold_target

        if pos > 80:
            hydro_state["long_peak_score"] = max(float(hydro_state.get("long_peak_score", 0.0)), float(regime_score))
            prev_peak_mid = hydro_state.get("long_peak_mid")
            hydro_state["long_peak_mid"] = float(mid) if prev_peak_mid is None else max(float(prev_peak_mid), float(mid))
            hydro_state["long_peak_trend"] = max(float(hydro_state.get("long_peak_trend", 0.0)), float(trend_score))
            hydro_state["short_peak_score"] = 0.0
            hydro_state["short_peak_mid"] = None
            hydro_state["short_peak_trend"] = 0.0
        elif pos < -80:
            hydro_state["short_peak_score"] = min(float(hydro_state.get("short_peak_score", 0.0)), float(regime_score))
            prev_peak_mid = hydro_state.get("short_peak_mid")
            hydro_state["short_peak_mid"] = float(mid) if prev_peak_mid is None else min(float(prev_peak_mid), float(mid))
            hydro_state["short_peak_trend"] = min(float(hydro_state.get("short_peak_trend", 0.0)), float(trend_score))
            hydro_state["long_peak_score"] = 0.0
            hydro_state["long_peak_mid"] = None
            hydro_state["long_peak_trend"] = 0.0
        elif abs(pos) < 40:
            hydro_state["long_peak_score"] = 0.0
            hydro_state["short_peak_score"] = 0.0
            hydro_state["long_peak_mid"] = None
            hydro_state["short_peak_mid"] = None
            hydro_state["long_peak_trend"] = 0.0
            hydro_state["short_peak_trend"] = 0.0
            hydro_state["exit_mode"] = ""
            hydro_state["exit_age"] = 0

        side = 1 if pos > 120 else -1 if pos < -120 else 0
        if side == 0:
            hydro_state["extreme_side"] = 0
            hydro_state["extreme_hold_bars"] = 0
        elif side == int(hydro_state.get("extreme_side", 0)):
            hydro_state["extreme_hold_bars"] = int(hydro_state.get("extreme_hold_bars", 0)) + 1
        else:
            hydro_state["extreme_side"] = side
            hydro_state["extreme_hold_bars"] = 1
        extreme_hold_bars = int(hydro_state.get("extreme_hold_bars", 0))
        if extreme_hold_bars > 24 and abs(regime_score) < 1.4:
            target = int(round(target * 0.70))

        if pos * target < 0 and abs(pos) > 60:
            target = 0

        long_peak_score = float(hydro_state.get("long_peak_score", 0.0))
        short_peak_score = float(hydro_state.get("short_peak_score", 0.0))
        long_peak_mid = hydro_state.get("long_peak_mid")
        short_peak_mid = hydro_state.get("short_peak_mid")
        long_peak_trend = float(hydro_state.get("long_peak_trend", 0.0))
        short_peak_trend = float(hydro_state.get("short_peak_trend", 0.0))
        long_drawdown = 0.0 if long_peak_mid is None else float(long_peak_mid) - float(mid)
        short_drawup = 0.0 if short_peak_mid is None else float(mid) - float(short_peak_mid)
        signal_fade = abs(fair - float(mid)) < max(4.0, 0.55 * spread)
        trend_fade_long = long_peak_trend > 0.0 and trend_score < 0.70 * long_peak_trend
        trend_fade_short = short_peak_trend < 0.0 and trend_score > 0.70 * short_peak_trend

        unwind_long = pos > 150 and (
            (long_peak_score >= 1.35 and long_drawdown >= 12.0 and regime_score < max(0.70, 0.65 * long_peak_score))
            or (trend_fade_long and long_drawdown >= 8.0)
            or (ret_ema < 0.0 and long_drawdown >= 8.0)
            or (signal_fade and long_drawdown >= 10.0 and pos > 140)
        )
        unwind_short = pos < -150 and (
            (short_peak_score <= -1.35 and short_drawup >= 12.0 and regime_score > min(-0.70, 0.65 * short_peak_score))
            or (trend_fade_short and short_drawup >= 8.0)
            or (ret_ema > 0.0 and short_drawup >= 8.0)
            or (signal_fade and short_drawup >= 10.0 and pos < -140)
        )
        prev_exit_mode = str(hydro_state.get("exit_mode", ""))
        exit_mode = "long" if unwind_long else "short" if unwind_short else ""
        hydro_state["exit_age"] = int(hydro_state.get("exit_age", 0)) + 1 if exit_mode and exit_mode == prev_exit_mode else (1 if exit_mode else 0)
        hydro_state["exit_mode"] = exit_mode
        exit_age = int(hydro_state["exit_age"])
        if exit_mode == "long":
            target = min(target, 80 if exit_age <= 3 else 40 if exit_age <= 8 else 0)
        elif exit_mode == "short":
            target = max(target, -80 if exit_age <= 3 else -40 if exit_age <= 8 else 0)

        absolute_danger_long = pos >= 150
        absolute_danger_short = pos <= -150
        quote_bias = 0.0 if abs(regime_score) < 0.75 else -0.06 * ((fair - float(mid)) / max(1.0, 0.5 * spread))
        fair_shift = clamp(12.0 * regime_score, -48.0, 48.0)
        size_mult = 0.75 if confidence == "guarded" else 1.0
        if confidence == "neutral":
            size_mult *= 0.45
        if exit_mode or absolute_danger_long or absolute_danger_short:
            size_mult *= 0.82

        return {
            "target": int(clamp(float(target), -200.0, 200.0)),
            "quote_bias": quote_bias,
            "fair_shift": fair_shift,
            "size_mult": size_mult,
            "exit_mode": exit_mode,
            "same_side_bid_block": exit_mode == "long" or absolute_danger_long,
            "same_side_ask_block": exit_mode == "short" or absolute_danger_short,
            "absolute_danger_long": absolute_danger_long,
            "absolute_danger_short": absolute_danger_short,
            "buy_take_extra": 2.5 if exit_mode == "long" else 2.0 if absolute_danger_long else 0.0,
            "sell_take_extra": 2.5 if exit_mode == "short" else 2.0 if absolute_danger_short else 0.0,
            "regime_score": regime_score,
            "confidence": confidence,
        }

    def _pair_execution_edge(self, cheap_row: dict, rich_row: dict) -> Tuple[float, float, float]:
        ask_iv = cheap_row.get("ask_iv")
        bid_iv = rich_row.get("bid_iv")
        if ask_iv is None or bid_iv is None:
            return -1.0, -1.0, -1.0
        buy_edge = float(cheap_row.get("fair_bid_iv", cheap_row["hybrid_iv"])) - float(ask_iv)
        sell_edge = float(bid_iv) - float(rich_row.get("fair_ask_iv", rich_row["hybrid_iv"]))
        return buy_edge + sell_edge, buy_edge, sell_edge

    def _select_pairs_and_targets(
        self, state: TradingState, surface: Dict[str, dict], meta: dict, strip: dict, memory: dict
    ) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
        ordered = sorted(surface, key=lambda p: VOUCHERS[p])
        entry_threshold = max(0.0098, 0.80 * meta["avg_abs_iv_residual"])
        shock_entry_threshold = max(0.0078, 0.72 * entry_threshold)
        outright_threshold = max(0.038, 2.40 * entry_threshold)
        active_entry = shock_entry_threshold if strip["shock_mode"] else entry_threshold
        max_vega = max((float(surface[p]["vega"]) for p in ordered), default=1.0)
        candidates: List[dict] = []
        for i, cheap in enumerate(ordered):
            row_c = surface[cheap]
            if row_c["market_iv"] is None or row_c["iv_residual"] >= -active_entry:
                continue
            for j in range(i + 1, min(len(ordered), i + PAIR_DISTANCE_LIMIT + 2)):
                rich = ordered[j]
                row_r = surface[rich]
                if row_r["market_iv"] is None or row_r["iv_residual"] <= active_entry:
                    continue
                spread = float(row_r["iv_residual"] - row_c["iv_residual"])
                core_pair = 5000 <= VOUCHERS[cheap] <= 5400 and 5000 <= VOUCHERS[rich] <= 5500
                min_spread = 0.68 * active_entry if core_pair else 0.82 * active_entry
                if spread <= min_spread:
                    continue
                liq = min(row_c["liq"], row_r["liq"])
                liq_score = clamp(liq / 55.0, 0.0, 1.0)
                vega_score = clamp(min(float(row_c["vega"]), float(row_r["vega"])) / max(max_vega, 1e-6), 0.25, 1.0)
                neigh = 0.5 * (row_c["neighbor_confirmation"] + row_r["neighbor_confirmation"])
                pair_delta = float(row_c["delta"] - row_r["delta"])
                delta_penalty = abs(strip["strip_delta"] + pair_delta) - abs(strip["strip_delta"])
                delta_score = 1.0 - clamp(max(0.0, delta_penalty) / 0.25, 0.0, 1.0)
                fit_score = meta["stability"]
                residual_strength = spread / max(active_entry, 1e-6)
                confidence = (
                    0.36 * residual_strength
                    + 0.22 * fit_score
                    + 0.18 * neigh
                    + 0.16 * liq_score
                    + 0.08 * strip["hedge_feasible"]
                ) * delta_score * vega_score
                score = confidence - 0.10 * max(0, VOUCHERS[rich] - 5300) / 400.0
                candidates.append(
                    {
                        "cheap": cheap,
                        "rich": rich,
                        "spread": spread,
                        "score": score,
                        "confidence": confidence,
                        "pair_delta": pair_delta,
                    }
                )

        candidates.sort(key=lambda row: (row["score"], row["spread"]), reverse=True)
        pair_limit = PAIR_LIMIT_SHOCK if strip["shock_mode"] else PAIR_LIMIT_NORMAL
        selected: List[dict] = []
        used = set()
        for row in candidates:
            if row["cheap"] in used or row["rich"] in used:
                continue
            selected.append(row)
            used.add(row["cheap"])
            used.add(row["rich"])
            if len(selected) >= pair_limit:
                break

        voucher_memory = memory["voucher"]
        cooldowns = {
            key: int(value)
            for key, value in voucher_memory.get("deep_pair_cooldowns", {}).items()
            if int(value) > state.timestamp
        }
        voucher_memory["deep_pair_cooldowns"] = cooldowns
        active_deep_pair = voucher_memory.get("active_deep_pair")
        late_latch = state.timestamp >= DEEP_PAIR_LATE_LATCH_TS and not strip["shock_mode"]
        target_bias = {p: 0 for p in ordered}
        source = {p: "flat" for p in ordered}
        selected_pairs: List[str] = []

        for row in selected:
            cheap = row["cheap"]
            rich = row["rich"]
            if cheap in DEEP_PAIR_PRODUCTS or rich in DEEP_PAIR_PRODUCTS:
                continue
            selected_pairs.append(f"{cheap}->{rich}")
            cheap_strike = surface[cheap]["strike"]
            rich_strike = surface[rich]["strike"]
            cheap_cap = BASE_STRIKE_CAPS[cheap_strike]
            rich_cap = BASE_STRIKE_CAPS[rich_strike]
            if strip["shock_mode"]:
                cheap_cap = int(round(1.15 * cheap_cap))
                rich_cap = int(round(1.15 * rich_cap))
            size_cap = min(cheap_cap, rich_cap, 58)
            size = rank_to_size(row["confidence"] * strip["hedge_feasible"], size_cap)
            target_bias[cheap] += size
            target_bias[rich] -= size
            source[cheap] = "pair"
            source[rich] = "pair"

        deep_candidates = [
            row
            for row in candidates
            if row["cheap"] in DEEP_PAIR_PRODUCTS and row["rich"] in DEEP_PAIR_PRODUCTS
        ]
        best_deep = deep_candidates[0] if deep_candidates else None
        if active_deep_pair:
            cheap = str(active_deep_pair.get("cheap"))
            rich = str(active_deep_pair.get("rich"))
            pair_key = f"{cheap}|{rich}"
            if cheap not in surface or rich not in surface:
                voucher_memory["active_deep_pair"] = None
                cooldowns[pair_key] = state.timestamp + DEEP_PAIR_COOLDOWN
                active_deep_pair = None
            else:
                row_c = surface[cheap]
                row_r = surface[rich]
                exec_edge, _, _ = self._pair_execution_edge(row_c, row_r)
                cur_spread = float(row_r["iv_residual"] - row_c["iv_residual"])
                exit_edge = max(0.0045, 0.45 * float(active_deep_pair.get("entry_edge", 0.010)))
                exit_spread = max(0.0060, 0.45 * float(active_deep_pair.get("entry_spread", 0.014)))
                pair_age = state.timestamp - int(active_deep_pair.get("entry_ts", state.timestamp))
                should_close = (
                    exec_edge <= exit_edge
                    or cur_spread <= exit_spread
                    or meta["resid_compression"] > 0.45
                    or state.timestamp >= DEEP_PAIR_HARD_STOP_TS
                    or (late_latch and pair_age >= 4000)
                )
                if should_close:
                    voucher_memory["active_deep_pair"] = None
                    cooldowns[pair_key] = state.timestamp + DEEP_PAIR_COOLDOWN
                    active_deep_pair = None
                else:
                    size = int(active_deep_pair.get("size", 12))
                    target_bias[cheap] = max(target_bias[cheap], size)
                    target_bias[rich] = min(target_bias[rich], -size)
                    source[cheap] = "pair_hold"
                    source[rich] = "pair_hold"
                    selected_pairs.append(f"{cheap}->{rich}*")

        if active_deep_pair is None and best_deep is not None and not late_latch:
            cheap = best_deep["cheap"]
            rich = best_deep["rich"]
            pair_key = f"{cheap}|{rich}"
            if cooldowns.get(pair_key, -1) <= state.timestamp:
                exec_edge, buy_edge, sell_edge = self._pair_execution_edge(surface[cheap], surface[rich])
                cost_gate = max(0.0085, 0.62 * active_entry)
                if (
                    exec_edge > cost_gate
                    and buy_edge > 0.003
                    and sell_edge > 0.003
                    and best_deep["confidence"] > 0.80
                ):
                    cap = min(BASE_STRIKE_CAPS[VOUCHERS[cheap]], BASE_STRIKE_CAPS[VOUCHERS[rich]], 18)
                    if strip["shock_mode"]:
                        cap = min(22, int(round(1.15 * cap)))
                    size = rank_to_size(best_deep["confidence"] * strip["hedge_feasible"], cap)
                    voucher_memory["active_deep_pair"] = {
                        "cheap": cheap,
                        "rich": rich,
                        "size": size,
                        "entry_ts": int(state.timestamp),
                        "entry_edge": float(exec_edge),
                        "entry_spread": float(best_deep["spread"]),
                    }
                    target_bias[cheap] = max(target_bias[cheap], size)
                    target_bias[rich] = min(target_bias[rich], -size)
                    source[cheap] = "pair_hold"
                    source[rich] = "pair_hold"
                    selected_pairs.append(f"{cheap}->{rich}+")

        outright_candidates: List[dict] = []
        for product in ordered:
            if source[product] != "flat":
                continue
            row = surface[product]
            resid = float(row["iv_residual"])
            if abs(resid) < outright_threshold:
                continue
            if row["neighbor_confirmation"] < 0.5:
                continue
            strike = row["strike"]
            if strike <= 4500:
                continue
            if strike >= 5500:
                continue
            if strike >= 5400:
                cap = 10
            elif strike in MIDDLE_STRIKES:
                cap = min(BASE_STRIKE_CAPS[strike], UNPAIRED_MIDDLE_CAP)
            else:
                cap = max(14, int(round(0.35 * BASE_STRIKE_CAPS[strike])))
            liq_score = clamp(row["liq"] / 55.0, 0.0, 1.0)
            vega_score = clamp(float(row["vega"]) / max(max_vega, 1e-6), 0.18, 1.0)
            score = (
                0.42 * abs(resid) / max(outright_threshold, 1e-6)
                + 0.26 * meta["stability"]
                + 0.16 * row["neighbor_confirmation"]
                + 0.16 * liq_score
            ) * strip["hedge_feasible"] * vega_score
            outright_candidates.append(
                {
                    "product": product,
                    "score": score,
                    "cap": cap,
                    "sign": -1 if resid > 0.0 else 1,
                }
            )

        outright_candidates.sort(key=lambda row: row["score"], reverse=True)
        outright_used_signs = set()
        for row in outright_candidates:
            if row["score"] < 1.08:
                continue
            if row["sign"] in outright_used_signs and not strip["shock_mode"]:
                continue
            product = row["product"]
            size = rank_to_size(row["score"], row["cap"])
            target_bias[product] += row["sign"] * size
            source[product] = "outright"
            outright_used_signs.add(row["sign"])
            if len(outright_used_signs) >= 2:
                break

        middle_target_gross = sum(abs(target_bias[p]) for p in ordered if surface[p]["strike"] in MIDDLE_STRIKES)
        if middle_target_gross > MIDDLE_CLUSTER_CAP:
            scale = MIDDLE_CLUSTER_CAP / middle_target_gross
            for product in ordered:
                if surface[product]["strike"] in MIDDLE_STRIKES:
                    target_bias[product] = int(round(target_bias[product] * scale))

        for left, right in zip(ordered, ordered[1:]):
            l = target_bias[left]
            r = target_bias[right]
            if sign(l) != 0 and sign(l) == sign(r):
                gross = abs(l) + abs(r)
                if gross > ADJ_SAME_SIDE_CAP:
                    scale = ADJ_SAME_SIDE_CAP / gross
                    target_bias[left] = int(round(l * scale))
                    target_bias[right] = int(round(r * scale))

        target_delta = strip["strip_delta"]
        for product in ordered:
            current_pos = int(state.position.get(product, 0))
            target_delta += (target_bias[product] - current_pos) * float(surface[product]["delta"])
        if abs(target_delta) > STRIP_DELTA_HARD_CAP:
            scale = STRIP_DELTA_HARD_CAP / max(abs(target_delta), 1e-6)
            for product in ordered:
                target_bias[product] = int(round(target_bias[product] * scale))

        if late_latch and voucher_memory.get("active_deep_pair") is None:
            for product in DEEP_PAIR_PRODUCTS:
                if product in target_bias:
                    target_bias[product] = 0
                    if source[product] == "flat":
                        source[product] = "latched"

        return target_bias, source, selected_pairs

    def _build_velvet_plan(self, state: TradingState, spot_fair: float, strip: dict, meta: dict) -> dict:
        od = state.order_depths[VELVET]
        snap = self._book_snapshot(od)
        mid = float(snap["mid"] if snap["mid"] is not None else spot_fair)
        spread = float(snap["spread"] if snap["spread"] < 900 else 6.0)
        fair_gap = spot_fair - mid
        alpha_signal = fair_gap / max(1.5, 0.5 * spread)
        alpha_target = 18.0 * math.tanh(0.70 * alpha_signal)
        if abs(strip["strip_delta"]) > 65.0:
            alpha_target *= 0.25
        if meta["resid_compression"] > 0.35:
            alpha_target *= 0.55
        if meta["resid_compression"] > 0.60:
            alpha_target = 0.0
        hedge_target = float(strip["target_velvet"])
        final_target = int(round(clamp(hedge_target + alpha_target, -120.0, 120.0)))
        quote_bias = 0.0
        if hedge_target > state.position.get(VELVET, 0):
            quote_bias -= 0.4
        elif hedge_target < state.position.get(VELVET, 0):
            quote_bias += 0.4
        if meta["resid_compression"] > 0.35:
            quote_bias *= 0.6
        return {
            "hedge_ratio": strip["hedge_ratio"],
            "vev_hedge_target": int(round(hedge_target)),
            "vev_alpha_target": int(round(alpha_target)),
            "vev_final_target": final_target,
            "quote_bias": quote_bias,
            "mid": mid,
        }

    def _trade_hydro(self, state: TradingState, fair: float, hydro_ctx: dict) -> List[Order]:
        od = state.order_depths.get(HYDRO)
        if od is None:
            return []
        mgr = OrderAccumulator(HYDRO, int(state.position.get(HYDRO, 0)), LIMITS[HYDRO])
        cfg = UNDERLYING_CFG[HYDRO]
        bb = best_bid(od)
        ba = best_ask(od)
        spread = float((ba - bb) if bb is not None and ba is not None else 2.0)
        fair = fair + float(hydro_ctx.get("fair_shift", 0.0))
        target = float(hydro_ctx["target"])
        current_pos = int(state.position.get(HYDRO, 0))
        unwind_long = str(hydro_ctx.get("exit_mode", "")).startswith("long")
        unwind_short = str(hydro_ctx.get("exit_mode", "")).startswith("short")
        same_side_bid_block = bool(hydro_ctx.get("same_side_bid_block", False))
        same_side_ask_block = bool(hydro_ctx.get("same_side_ask_block", False))
        absolute_danger_long = bool(hydro_ctx.get("absolute_danger_long", False))
        absolute_danger_short = bool(hydro_ctx.get("absolute_danger_short", False))
        buy_take_allowed = not (same_side_bid_block or unwind_long or absolute_danger_long or current_pos >= 140)
        sell_take_allowed = not (same_side_ask_block or unwind_short or absolute_danger_short or current_pos <= -140)

        buy_take_edge = cfg["take_edge"] + float(hydro_ctx["buy_take_extra"])
        sell_take_edge = cfg["take_edge"] + float(hydro_ctx["sell_take_extra"])
        for ask, vol in sorted(od.sell_orders.items()):
            edge = fair - ask
            if not buy_take_allowed:
                break
            if edge >= buy_take_edge or edge >= cfg["take_edge"] + 2.5:
                mgr.buy(ask, min(-int(vol), cfg["take_max"]))
            else:
                break
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            edge = bid - fair
            if not sell_take_allowed:
                break
            if edge >= sell_take_edge or edge >= cfg["take_edge"] + 2.5:
                mgr.sell(bid, min(int(vol), cfg["take_max"]))
            else:
                break

        pos = mgr.projected()
        relative_pos = pos - target
        soft_limit = 80
        hard_zone = 130
        clear_edge = cfg["clear_edge"]
        hard_clear_edge = clear_edge + 1.5
        soft_clear_max = cfg["clear_max"]
        hard_clear_max = max(cfg["clear_max"], 56)
        if unwind_long or unwind_short:
            soft_limit = 25
            hard_zone = 45
            clear_edge += 2.0
            hard_clear_edge += 3.0
            soft_clear_max = max(cfg["clear_max"], 72)
            hard_clear_max = max(cfg["clear_max"], 96)
        elif absolute_danger_long or absolute_danger_short:
            soft_limit = 30
            hard_zone = 55
            clear_edge += 1.6
            hard_clear_edge += 2.6
            soft_clear_max = max(cfg["clear_max"], 64)
            hard_clear_max = max(cfg["clear_max"], 84)

        if relative_pos > hard_zone and bb is not None and (bb >= fair - hard_clear_edge or unwind_long or absolute_danger_long):
            mgr.sell(bb, min(int(math.ceil(relative_pos - soft_limit)), hard_clear_max))
        elif relative_pos > soft_limit and bb is not None and (bb >= fair - clear_edge or unwind_long or absolute_danger_long):
            mgr.sell(bb, min(int(math.ceil(relative_pos - soft_limit)), soft_clear_max))
        pos = mgr.projected()
        relative_pos = pos - target
        if relative_pos < -hard_zone and ba is not None and (ba <= fair + hard_clear_edge or unwind_short or absolute_danger_short):
            mgr.buy(ba, min(int(math.ceil((-soft_limit) - relative_pos)), hard_clear_max))
        elif relative_pos < -soft_limit and ba is not None and (ba <= fair + clear_edge or unwind_short or absolute_danger_short):
            mgr.buy(ba, min(int(math.ceil((-soft_limit) - relative_pos)), soft_clear_max))

        pos = mgr.projected()
        relative_pos = pos - target
        inv_ratio = relative_pos / LIMITS[HYDRO]
        reservation = fair + float(hydro_ctx["quote_bias"]) - (cfg["inv_skew"] + 4.0) * inv_ratio
        quote_edge = cfg["quote_edge"] + max(0.0, 0.12 * (spread - 4.0))
        buy_px = int(math.floor(reservation - quote_edge))
        sell_px = int(math.ceil(reservation + quote_edge))
        if bb is not None and ba is not None and bb < ba:
            if spread >= 3:
                buy_px = max(buy_px, bb + 1)
                sell_px = min(sell_px, ba - 1)
            else:
                buy_px = min(buy_px, bb)
                sell_px = max(sell_px, ba)
            buy_px = min(buy_px, ba - 1)
            sell_px = max(sell_px, bb + 1)
        buy_px = max(0, buy_px)
        sell_px = max(buy_px + 1, sell_px)
        size_scale = max(0.20, 1.0 - abs(inv_ratio)) * max(0.35, float(hydro_ctx["size_mult"]))
        if same_side_bid_block or same_side_ask_block:
            size_scale *= 0.85
        quote_size = max(6, int(round(cfg["quote_size"] * size_scale)))
        can_bid = mgr.buy_left > 0 and not (same_side_bid_block or unwind_long or absolute_danger_long)
        can_ask = mgr.sell_left > 0 and not (same_side_ask_block or unwind_short or absolute_danger_short)
        if can_bid:
            mgr.buy(buy_px, quote_size)
        if can_ask:
            mgr.sell(sell_px, quote_size)
        return mgr.flush()

    def _trade_velvet(self, state: TradingState, fair: float, plan: dict) -> List[Order]:
        od = state.order_depths.get(VELVET)
        if od is None:
            return []
        mgr = OrderAccumulator(VELVET, int(state.position.get(VELVET, 0)), LIMITS[VELVET])
        snap = self._book_snapshot(od)
        bb = snap["best_bid"]
        ba = snap["best_ask"]
        spread = float(snap["spread"] if snap["spread"] < 900 else 6.0)
        target = int(plan["vev_final_target"])
        take_edge = 1.1
        quote_edge = max(1.5, 0.35 * spread + 0.8)
        clear_edge = 0.7
        pos = mgr.projected()

        if target > pos and ba is not None:
            for ask, vol in sorted(od.sell_orders.items()):
                if ask <= fair - take_edge or ask <= fair + 0.1 and pos < target:
                    qty = min(target - pos, -int(vol), 26)
                    mgr.buy(ask, qty)
                    pos = mgr.projected()
                else:
                    break
        if target < pos and bb is not None:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                if bid >= fair + take_edge or bid >= fair - 0.1 and pos > target:
                    qty = min(pos - target, int(vol), 26)
                    mgr.sell(bid, qty)
                    pos = mgr.projected()
                else:
                    break

        pos = mgr.projected()
        if pos > target and bb is not None and bb >= fair - clear_edge:
            mgr.sell(bb, min(pos - target, 24))
        if pos < target and ba is not None and ba <= fair + clear_edge:
            mgr.buy(ba, min(target - pos, 24))

        pos = mgr.projected()
        inv_ratio = (pos - target) / LIMITS[VELVET]
        reservation = fair + float(plan["quote_bias"]) - 6.0 * inv_ratio
        buy_px = int(math.floor(reservation - quote_edge))
        sell_px = int(math.ceil(reservation + quote_edge))
        if bb is not None and ba is not None and bb < ba:
            buy_px = min(max(buy_px, bb + 1), ba - 1)
            sell_px = max(min(sell_px, ba - 1), bb + 1)
        buy_px = max(0, buy_px)
        sell_px = max(buy_px + 1, sell_px)
        quote_size = max(6, int(round(22 * clamp(1.0 - 0.60 * abs(inv_ratio), 0.25, 1.0))))
        if pos < target:
            mgr.buy(buy_px, quote_size)
        if pos > target:
            mgr.sell(sell_px, quote_size)
        return mgr.flush()

    def _trade_voucher(self, product: str, state: TradingState, row: dict, target: int, source: str, meta: dict, strip: dict) -> List[Order]:
        od = state.order_depths.get(product)
        if od is None:
            return []
        mgr = OrderAccumulator(product, int(state.position.get(product, 0)), LIMITS[product])
        pos = mgr.projected()
        if source == "latched" and pos == 0:
            return []
        fair_price = float(row["fair_price"])
        fair_iv = float(row["hybrid_iv"])
        fair_bid_iv = float(row.get("fair_bid_iv", fair_iv))
        fair_ask_iv = float(row.get("fair_ask_iv", fair_iv))
        fair_bid_price = float(row.get("fair_bid_price", fair_price))
        fair_ask_price = float(row.get("fair_ask_price", fair_price))
        market_iv = row["market_iv"]
        if market_iv is None:
            return []
        bb = row["bid"]
        ba = row["ask"]
        spread = float(row["spread"])
        strike = row["strike"]
        mid_band = max(0.75, 0.22 * spread)
        iv_band = max(0.010, 0.55 * max(0.012, 0.95 * meta["avg_abs_iv_residual"]))
        if strip["shock_mode"] and source == "pair":
            mid_band *= 0.75
            if 5000 <= strike <= 5400:
                iv_band *= 0.74
            else:
                iv_band *= 0.84
        elif source == "pair" and 5000 <= strike <= 5400:
            iv_band *= 0.88
        if source == "outright":
            mid_band *= 1.20
            iv_band *= 1.20
        if source == "pair_hold":
            mid_band *= 1.15
            iv_band *= 1.10
        price_band = max(0.25, 0.10 * spread)

        if target > pos:
            for ask, vol in sorted(od.sell_orders.items()):
                ask_iv = row.get("ask_iv")
                iv_ok = ask_iv is not None and float(ask_iv) <= fair_bid_iv - iv_band
                px_ok = ask <= fair_bid_price - price_band
                if ask <= fair_price - mid_band or iv_ok or px_ok:
                    mgr.buy(ask, min(target - mgr.projected(), -int(vol), 28))
                else:
                    break
        if target < pos:
            for bid, vol in sorted(od.buy_orders.items(), reverse=True):
                bid_iv = row.get("bid_iv")
                iv_ok = bid_iv is not None and float(bid_iv) >= fair_ask_iv + iv_band
                px_ok = bid >= fair_ask_price + price_band
                if bid >= fair_price + mid_band or iv_ok or px_ok:
                    mgr.sell(bid, min(mgr.projected() - target, int(vol), 28))
                else:
                    break

        pos = mgr.projected()
        clear_band = max(0.40, 0.15 * spread)
        if source == "pair_hold":
            clear_band *= 0.75
        if pos > target and bb is not None and bb >= fair_price - clear_band:
            mgr.sell(bb, min(pos - target, 22))
        if pos < target and ba is not None and ba <= fair_price + clear_band:
            mgr.buy(ba, min(target - pos, 22))

        pos = mgr.projected()
        inv_ratio = (pos - target) / LIMITS[product]
        reservation = fair_price - 1.8 * inv_ratio
        quote_edge = max(1.0, 0.38 * spread + 0.3)
        if source == "pair_hold":
            quote_edge *= 1.12
        if source == "latched":
            quote_edge *= 1.30
        bid_quote = int(math.floor(reservation - quote_edge))
        ask_quote = int(math.ceil(reservation + quote_edge))
        if bb is not None and ba is not None and bb < ba:
            bid_quote = min(max(bid_quote, bb + 1), ba - 1)
            ask_quote = max(min(ask_quote, ba - 1), bb + 1)
        bid_quote = max(0, bid_quote)
        ask_quote = max(bid_quote + 1, ask_quote)
        base_quote = 10 if strike <= 5000 else 7
        quote_size = max(4, int(round(base_quote * clamp(1.0 - 0.70 * abs(inv_ratio), 0.25, 1.0))))
        if source == "pair_hold":
            quote_size = max(3, int(round(0.75 * quote_size)))
        if source == "latched":
            quote_size = 0
        if pos < target:
            mgr.buy(bid_quote, quote_size)
        if pos > target:
            mgr.sell(ask_quote, quote_size)
        return mgr.flush()

    def run(self, state: TradingState):
        memory = load_memory(state.traderData)
        self._reset_if_new_day(memory, state.timestamp)
        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}

        if HYDRO in state.order_depths:
            hydro_fair = self._hydro_underlying_fair(state.order_depths[HYDRO])
            hydro_ctx = self._build_hydro_context(state, hydro_fair, memory)
            result[HYDRO] = self._trade_hydro(state, hydro_fair, hydro_ctx)
        else:
            hydro_fair = None
            hydro_ctx = {}

        strip_monitor = {}
        if VELVET in state.order_depths:
            spot_fair = self._robust_underlying_fair(VELVET, state.order_depths[VELVET], memory)
            surface, meta = self._build_voucher_surface(state, spot_fair, memory)
            strip = self._build_strip_context(state, surface, meta)
            target_bias, source_map, selected_pairs = self._select_pairs_and_targets(state, surface, meta, strip, memory)
            velvet_plan = self._build_velvet_plan(state, spot_fair, strip, meta)
            result[VELVET] = self._trade_velvet(state, spot_fair, velvet_plan)
            for product, row in surface.items():
                target = int(target_bias.get(product, 0))
                result[product] = self._trade_voucher(product, state, row, target, source_map.get(product, "flat"), meta, strip)
            strip_monitor = {
                "spot_fair": round(float(spot_fair), 3),
                "mode": "BROAD_SHOCK" if strip["shock_mode"] else "STEADY_PAIR_SCALP",
                "stability": round(float(meta["stability"]), 4),
                "avg_abs_iv_residual": round(float(meta["avg_abs_iv_residual"]), 4),
                "pair_agreement_count": int(meta["pair_agreement_count"]),
                "resid_compression": round(float(meta["resid_compression"]), 4),
                "strip_delta": round(float(strip["strip_delta"]), 3),
                "strip_vega_proxy": round(float(strip["strip_vega_proxy"]), 3),
                "hedge_ratio": round(float(velvet_plan["hedge_ratio"]), 3),
                "hedge_feasible": round(float(strip["hedge_feasible"]), 3),
                "vev_hedge_target": int(velvet_plan["vev_hedge_target"]),
                "vev_alpha_target": int(velvet_plan["vev_alpha_target"]),
                "vev_final_target": int(velvet_plan["vev_final_target"]),
                "cheapest": list(meta["cheapest"]),
                "richest": list(meta["richest"]),
                "selected_pairs": selected_pairs,
                "pair_trade_count": sum(1 for product in source_map if source_map[product] == "pair"),
                "outright_trade_count": sum(1 for product in source_map if source_map[product] == "outright"),
                "deep_pair": memory["voucher"].get("active_deep_pair"),
                "deep_pair_cooldowns": len(memory["voucher"].get("deep_pair_cooldowns", {})),
            }

        memory["monitor"] = {
            "strip": strip_monitor,
            "hydro": {
                "fair": round(float(hydro_fair), 3) if hydro_fair is not None else None,
                "target": int(hydro_ctx.get("target", 0)) if hydro_ctx else 0,
                "regime_score": round(float(hydro_ctx.get("regime_score", 0.0)), 3) if hydro_ctx else 0.0,
                "confidence": hydro_ctx.get("confidence", "") if hydro_ctx else "",
                "exit_mode": hydro_ctx.get("exit_mode", "") if hydro_ctx else "",
            },
        }
        conversions = 0
        return result, conversions, dump_memory(memory)
