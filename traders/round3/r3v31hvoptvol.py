# r3v31hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, Trade, TradingState
except ModuleNotFoundError:
    from trader_factory.core.datamodel import Order, OrderDepth, Trade, TradingState

_N = NormalDist()

HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
VOUCHER_STRIKES: Dict[str, int] = {
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
LIMITS: Dict[str, int] = {
    HYDROGEL: 200,
    VELVET: 200,
    **{p: 300 for p in VOUCHER_STRIKES},
}
_TTE_TOTAL_DAYS = 5

MM: Dict[str, dict] = {
    HYDROGEL: {
        "anchor":     10000.0,
        "anchor_w":   0.15,
        "imb_w":      0.80,
        "take_edge":  1.5,
        "take_max":   20,
        "soft_limit": 50,
        "clear_edge": 1.0,
        "clear_max":  25,
        "quote_edge": 2.5,
        "quote_size": 18,
        "skew":       6.0,
        "n_levels":   2,
        "level_gap":  3.0,
    },
    VELVET: {
        "anchor":     5250.0,
        "anchor_w":   0.46,
        "imb_w":      0.85,
        "take_edge":  1.2,
        "take_max":   34,
        "soft_limit": 100,
        "clear_edge": 0.6,
        "clear_max":  40,
        "quote_edge": 1.5,
        "quote_size": 30,
        "skew":       5.0,
        "n_levels":   1,
        "level_gap":  0.0,
    },
}

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def load_memory(s: str) -> dict:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def dump_memory(m: dict) -> str:
    return json.dumps(m, separators=(",", ":"))

class OrderManager:
    def __init__(self, product: str, position: int, limit: int) -> None:
        self.product = product
        self.position = int(position)
        self.limit = int(limit)
        self.buy_cap = max(0, limit - position)
        self.sell_cap = max(0, limit + position)
        self._orders: List[Order] = []

    def projected(self) -> int:
        return self.position + sum(o.quantity for o in self._orders)

    def buy(self, price: int, qty: int) -> None:
        size = min(max(0, int(qty)), self.buy_cap)
        if size > 0:
            self._orders.append(Order(self.product, int(price), size))
            self.buy_cap -= size

    def sell(self, price: int, qty: int) -> None:
        size = min(max(0, int(qty)), self.sell_cap)
        if size > 0:
            self._orders.append(Order(self.product, int(price), -size))
            self.sell_cap -= size

    def flush(self) -> List[Order]:
        orders = self._orders
        self._orders = []
        return orders

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

def top_bid_levels(od: OrderDepth, n: int = 3) -> List[Tuple[int, int]]:
    return sorted(od.buy_orders.items(), reverse=True)[:n]

def top_ask_levels(od: OrderDepth, n: int = 3) -> List[Tuple[int, int]]:
    return sorted(od.sell_orders.items())[:n]

def stable_mid(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None or bb >= ba:
        return raw_mid(od)

    bid_levels = top_bid_levels(od, 3)
    ask_levels = top_ask_levels(od, 3)
    bvol = sum(v for _, v in bid_levels)
    avol = sum(abs(v) for _, v in ask_levels)
    if bvol <= 0 or avol <= 0:
        return raw_mid(od)
    bpx = sum(p * v for p, v in bid_levels) / bvol
    apx = sum(p * abs(v) for p, v in ask_levels) / avol
    if bpx >= apx:
        return raw_mid(od)
    return 0.5 * (bpx + apx)

def micro_price(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None or bb >= ba:
        return raw_mid(od)
    bv = od.buy_orders[bb]
    av = abs(od.sell_orders[ba])
    total = bv + av
    if total <= 0:
        return raw_mid(od)
    return (ba * bv + bb * av) / total

def book_imbalance(od: OrderDepth, levels: int = 2) -> float:
    if not od.buy_orders or not od.sell_orders:
        return 0.0
    bl = top_bid_levels(od, levels)
    al = top_ask_levels(od, levels)
    bv = sum(v for _, v in bl)
    av = sum(abs(v) for _, v in al)
    total = bv + av
    return (bv - av) / total if total > 0 else 0.0

def norm_cdf(x: float) -> float:
    return _N.cdf(x)

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def bs_call(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if tte <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)
    sq = math.sqrt(tte)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / (sigma * sq)
    d2 = d1 - sigma * sq
    return spot * norm_cdf(d1) - strike * norm_cdf(d2)

def bs_delta(spot: float, strike: float, tte: float, sigma: float) -> float:
    if spot <= 0 or strike <= 0:
        return 0.0
    if tte <= 0 or sigma <= 0:
        return 1.0 if spot > strike else 0.0
    sq = math.sqrt(tte)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * tte) / (sigma * sq)
    return norm_cdf(d1)

def implied_vol(price: float, spot: float, strike: float, tte: float, iters: int = 50) -> float:
    intrinsic = max(spot - strike, 0.0)
    if spot <= 0 or strike <= 0 or tte <= 0:
        return 1e-6
    if price <= intrinsic + 1e-6:
        return 1e-6
    lo, hi = 1e-6, 3.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bs_call(spot, strike, tte, mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def fit_quadratic(
    xs: List[float], ys: List[float], ws: List[float]
) -> Tuple[float, float, float]:
    if len(xs) < 3:
        median = sorted(ys)[len(ys) // 2] if ys else 0.18
        return 0.0, 0.0, float(median)
    sw = sum(ws)
    s1 = sum(w * x for x, w in zip(xs, ws))
    s2 = sum(w * x * x for x, w in zip(xs, ws))
    s3 = sum(w * x * x * x for x, w in zip(xs, ws))
    s4 = sum(w * x * x * x * x for x, w in zip(xs, ws))
    t0 = sum(w * y for y, w in zip(ys, ws))
    t1 = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    t2 = sum(w * x * x * y for x, y, w in zip(xs, ys, ws))
    mat = [[s4, s3, s2, t2], [s3, s2, s1, t1], [s2, s1, sw, t0]]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            median = sorted(ys)[len(ys) // 2]
            return 0.0, 0.0, float(median)
        mat[col], mat[pivot] = mat[pivot], mat[col]
        f = mat[col][col]
        mat[col] = [v / f for v in mat[col]]
        for r in range(3):
            if r != col:
                fac = mat[r][col]
                mat[r] = [mat[r][j] - fac * mat[col][j] for j in range(4)]
    return float(mat[0][3]), float(mat[1][3]), float(mat[2][3])

class Trader:

    def _fair(self, product: str, od: OrderDepth) -> float:
        cfg = MM[product]
        anchor = cfg["anchor"]
        stable = stable_mid(od)
        micro = micro_price(od)
        bb = best_bid(od)
        ba = best_ask(od)
        spread = float((ba - bb) if bb is not None and ba is not None else 2.0)

        stable_c = stable if stable is not None else anchor
        micro_c = micro if micro is not None else stable_c
        fair = cfg["anchor_w"] * anchor + (1.0 - cfg["anchor_w"]) * 0.5 * (stable_c + micro_c)
        fair += cfg["imb_w"] * book_imbalance(od) * max(1.0, 0.5 * spread)
        return fair

    def _trade_mm(self, product: str, state: TradingState, fair: float) -> List[Order]:
        # Take → Clear → Ladder-Make. No regime, no directional target. The anchor_w in fair() creates take
        od = state.order_depths[product]
        cfg = MM[product]
        limit = LIMITS[product]
        pos = int(state.position.get(product, 0))
        mgr = OrderManager(product, pos, limit)

        bb = best_bid(od)
        ba = best_ask(od)
        spread = float((ba - bb) if bb is not None and ba is not None else 2.0)

        for ask, vol in sorted(od.sell_orders.items()):
            if fair - ask >= cfg["take_edge"]:
                mgr.buy(ask, min(-vol, cfg["take_max"]))
            else:
                break
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid - fair >= cfg["take_edge"]:
                mgr.sell(bid, min(vol, cfg["take_max"]))
            else:
                break

        proj = mgr.projected()
        if proj > cfg["soft_limit"] and bb is not None and bb >= fair - cfg["clear_edge"]:
            mgr.sell(bb, min(proj - cfg["soft_limit"], cfg["clear_max"]))
        elif proj < -cfg["soft_limit"] and ba is not None and ba <= fair + cfg["clear_edge"]:
            mgr.buy(ba, min(-cfg["soft_limit"] - proj, cfg["clear_max"]))

        proj = mgr.projected()
        inv_ratio = proj / float(limit)
        reservation = fair - cfg["skew"] * inv_ratio
        base_edge = cfg["quote_edge"] + max(0.0, 0.1 * (spread - 4.0))

        for lvl in range(int(cfg["n_levels"])):
            offset = lvl * cfg["level_gap"]
            size_scale = 0.70 ** lvl
            buy_px = math.floor(reservation - base_edge - offset)
            sell_px = math.ceil(reservation + base_edge + offset)

            if bb is not None and ba is not None and bb < ba:
                buy_px = min(buy_px, ba - 1)
                sell_px = max(sell_px, bb + 1)

            qty = max(4, int(round(cfg["quote_size"] * size_scale)))
            if mgr.buy_cap > 0 and (ba is None or buy_px < ba):
                mgr.buy(buy_px, qty)
            if mgr.sell_cap > 0 and (bb is None or sell_px > bb):
                mgr.sell(sell_px, qty)

        return mgr.flush()

    def _build_vol_surface(
        self, state: TradingState, velvet_fair: float, tte: float
    ) -> Dict[str, float]:
        # Quadratic fit to market IVs → fitted sigma per strike.
        ms: List[float] = []
        ivs: List[float] = []
        ws: List[float] = []
        for product, strike in VOUCHER_STRIKES.items():
            od = state.order_depths.get(product)
            if od is None:
                continue
            mid = raw_mid(od)
            if mid is None or mid <= 0.0:
                continue
            guarded = max(float(mid), max(velvet_fair - strike, 0.0) + 1e-3)
            iv = implied_vol(guarded, velvet_fair, strike, tte)
            if not (1e-6 < iv < 3.0):
                continue
            m = math.log(strike / velvet_fair) / math.sqrt(tte)
            bb = best_bid(od)
            ba = best_ask(od)
            spread = float((ba - bb) if bb is not None and ba is not None else 4.0)
            liq = 0.0
            if bb is not None:
                liq += float(max(0, od.buy_orders.get(bb, 0)))
            if ba is not None:
                liq += float(abs(od.sell_orders.get(ba, 0)))
            w = clamp((math.log1p(liq) + 0.5) / max(spread, 0.5), 0.3, 4.0)
            ms.append(m)
            ivs.append(iv)
            ws.append(w)

        if len(ivs) >= 3:
            sorted_iv = sorted(ivs)
            med = sorted_iv[len(sorted_iv) // 2]
            clipped = [clamp(iv, med - 0.20, med + 0.20) for iv in ivs]
            coeffs = fit_quadratic(ms, clipped, ws)
        elif ivs:
            med = sorted(ivs)[len(ivs) // 2]
            coeffs = (0.0, 0.0, float(med))
        else:
            coeffs = (0.0, 0.0, 0.18)

        a, b, c = coeffs
        sigmas: Dict[str, float] = {}
        for product, strike in VOUCHER_STRIKES.items():
            m = math.log(strike / velvet_fair) / math.sqrt(tte)
            sigma = clamp(a * m * m + b * m + c, 0.01, 3.0)
            sigmas[product] = sigma
        return sigmas

    def _trade_voucher(
        self,
        product: str,
        state: TradingState,
        velvet_fair: float,
        sigma: float,
        strip_delta: float,
        tte: float,
        pair_edge_5100_5200: float,
        pair_edge_5200_5300: float,
    ) -> List[Order]:
        # Simple BS fair + Take/Clear/Make for one voucher strike.
        od = state.order_depths[product]
        strike = VOUCHER_STRIKES[product]
        limit = LIMITS[product]
        pos = int(state.position.get(product, 0))
        mgr = OrderManager(product, pos, limit)

        fair = bs_call(velvet_fair, strike, tte, sigma)
        if fair < 0.5:
            return mgr.flush()

        delta = bs_delta(velvet_fair, strike, tte, sigma)
        bb = best_bid(od)
        ba = best_ask(od)
        spread = float((ba - bb) if bb is not None and ba is not None else 2.0)

        take_edge = max(0.6, 0.30 * spread)
        clear_edge = max(0.3, 0.12 * spread)
        quote_edge = max(0.9, 0.40 * spread)
        if strike <= 5000:
            soft_limit = 80
            take_max = 15
        elif strike == 5100:
            soft_limit = 75
            take_max = 12
        elif strike == 5200:
            soft_limit = 48
            take_max = 10
        elif strike == 5300:
            soft_limit = 24
            take_max = 5
        elif strike == 5400:
            soft_limit = 10
            take_max = 3
        elif strike == 5500:
            soft_limit = 6
            take_max = 2
        else:
            soft_limit = 4
            take_max = 1

        mid_market = raw_mid(od)
        if mid_market is not None:
            guarded = max(float(mid_market), max(velvet_fair - strike, 0.0) + 1e-3)
            market_iv = implied_vol(guarded, velvet_fair, strike, tte)
            iv_residual = market_iv - sigma
        else:
            iv_residual = 0.0

        if delta > 0.85:
            resid_threshold = 0.014
            soft_limit = max(soft_limit, 90)
            take_max = max(take_max, 16)
        elif delta > 0.65:
            resid_threshold = 0.016
            soft_limit = max(soft_limit, 82)
            take_max = max(take_max, 14)
        elif delta > 0.35:
            resid_threshold = 0.026
            soft_limit = min(soft_limit, 35)
            take_max = min(take_max, 8)
        elif delta > 0.12:
            resid_threshold = 0.040
            soft_limit = min(soft_limit, 15)
            take_max = min(take_max, 4)
        else:
            resid_threshold = 0.060
            soft_limit = min(soft_limit, 4)
            take_max = min(take_max, 1)

        sell_threshold = resid_threshold
        if strike == 5200:

            if pair_edge_5100_5200 > 0.010:
                resid_threshold = max(0.014, resid_threshold - 0.002)
                sell_threshold = max(0.012, sell_threshold - 0.004)
                soft_limit = min(max(soft_limit, 44), 60)
                take_max = min(max(take_max, 10), 12)
            elif pair_edge_5100_5200 < -0.010:
                resid_threshold = resid_threshold + 0.004
                sell_threshold = sell_threshold + 0.006
                soft_limit = min(soft_limit, 40)
                take_max = min(take_max, 8)
            if pair_edge_5200_5300 > 0.008:
                sell_threshold = max(0.012, sell_threshold - 0.004)
                soft_limit = min(max(soft_limit, 44), 60)
                take_max = min(max(take_max, 10), 12)
            elif pair_edge_5200_5300 < -0.008:
                sell_threshold = max(sell_threshold, resid_threshold + 0.008)
                soft_limit = min(soft_limit, 40)
                take_max = min(take_max, 8)
        elif strike == 5300:
            if pair_edge_5200_5300 < -0.008:
                sell_threshold = max(0.024, resid_threshold - 0.008)
                soft_limit = 30
                take_max = 6
            elif pair_edge_5200_5300 > 0.008:
                sell_threshold = resid_threshold + 0.012
        buy_signal = iv_residual < -resid_threshold
        sell_signal = iv_residual > sell_threshold

        if strike > velvet_fair:
            buy_signal = False

        delta_pressure = clamp(strip_delta / 120.0, -1.5, 1.5)
        if delta_pressure > 0.8:
            buy_signal = False
        if delta_pressure < -0.8:
            sell_signal = False

        for ask, vol in sorted(od.sell_orders.items()):
            edge = fair - ask
            if edge >= take_edge and buy_signal and pos < soft_limit:
                mgr.buy(ask, min(-vol, take_max))
            else:
                break
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            edge = bid - fair
            if edge >= take_edge and sell_signal and pos > -soft_limit:
                mgr.sell(bid, min(vol, take_max))
            else:
                break

        proj = mgr.projected()
        if proj > soft_limit and bb is not None and bb >= fair - clear_edge:
            mgr.sell(bb, min(proj - soft_limit, 30))
        elif proj < -soft_limit and ba is not None and ba <= fair + clear_edge:
            mgr.buy(ba, min(-soft_limit - proj, 30))

        proj = mgr.projected()
        inv_ratio = proj / float(limit)
        inv_penalty = (0.015 * max(20.0, fair) + 1.5) * inv_ratio
        reservation = fair - inv_penalty

        buy_px = math.floor(reservation - quote_edge)
        sell_px = math.ceil(reservation + quote_edge)
        if bb is not None and ba is not None and bb < ba:
            if spread >= 2:
                buy_px = max(buy_px, bb + 1)
                sell_px = min(sell_px, ba - 1)
            else:
                buy_px = min(buy_px, bb)
                sell_px = max(sell_px, ba)
            buy_px = min(buy_px, ba - 1)
            sell_px = max(sell_px, bb + 1)

        qty = max(3, int(round(8 * max(0.2, 1.0 - abs(inv_ratio)))))

        can_bid = mgr.buy_cap > 0 and pos < soft_limit and (buy_signal or pos < 0)
        can_ask = mgr.sell_cap > 0 and pos > -soft_limit and (sell_signal or pos > 0)

        if can_bid and (ba is None or buy_px < ba):
            mgr.buy(buy_px, qty)
        if can_ask and (bb is None or sell_px > bb):
            mgr.sell(sell_px, qty)

        return mgr.flush()

    def _velvet_overlay(self, state: TradingState, memory: dict) -> float:
        # Detect anonymous bot (Olivia pattern) in VELVET trades → small fair shift.
        ov = memory.setdefault(
            "velvet_overlay",
            {"day_low": 1e18, "day_high": -1e18, "signal": 0.0, "age": 999},
        )
        saw = False
        for trade in sorted(state.market_trades.get(VELVET, []), key=lambda t: t.timestamp):
            qty = abs(int(trade.quantity))
            px = float(trade.price)
            if px < float(ov["day_low"]):
                ov["day_low"] = px
            if px > float(ov["day_high"]):
                ov["day_high"] = px
            if 10 <= qty <= 11 and px < float(ov["day_low"]) + 1e-6:
                ov["signal"] = min(1.5, float(ov["signal"]) + 1.0)
                ov["age"] = 0
                saw = True
        if not saw:
            ov["age"] = int(ov["age"]) + 1
            ov["signal"] = float(ov["signal"]) * 0.96
        return 0.75 * max(0.0, min(1.0, float(ov["signal"])))

    def _velvet_breakout_up(self, state: TradingState, velvet_fair: float) -> bool:
        od = state.order_depths.get(VELVET)
        if od is None:
            return False
        mid = raw_mid(od)
        micro = micro_price(od)
        bb = best_bid(od)
        ba = best_ask(od)
        if mid is None or micro is None or bb is None or ba is None or bb >= ba:
            return False
        spread = float(ba - bb)
        return (
            mid >= velvet_fair + max(2.0, 0.75 * spread)
            and micro >= mid
            and book_imbalance(od) >= -0.05
        )

    def _reset_day(self, memory: dict, timestamp: int) -> None:
        last = memory.get("last_timestamp", -1)
        if last >= 0 and timestamp < last:

            memory["day_count"] = int(memory.get("day_count", 0)) + 1
            memory.pop("velvet_overlay", None)
        memory["last_timestamp"] = timestamp

    def run(self, state: TradingState):
        memory = load_memory(state.traderData)
        self._reset_day(memory, state.timestamp)
        result: Dict[str, List[Order]] = {}

        day_count = int(memory.get("day_count", 0))
        t_rem = max(0.001, _TTE_TOTAL_DAYS - day_count - state.timestamp / 999900.0)
        tte = t_rem / 365.0

        if HYDROGEL in state.order_depths:
            try:
                hydro_fair = self._fair(HYDROGEL, state.order_depths[HYDROGEL])
                result[HYDROGEL] = self._trade_mm(HYDROGEL, state, hydro_fair)
            except Exception as exc:
                memory.setdefault("errors", {})["hydrogel"] = type(exc).__name__

        velvet_fair: Optional[float] = None
        voucher_velvet_fair: Optional[float] = None
        sigmas: Optional[Dict[str, float]] = None
        strip_delta = 0.0
        if VELVET in state.order_depths:
            try:
                overlay_bias = self._velvet_overlay(state, memory)
                voucher_velvet_fair = self._fair(VELVET, state.order_depths[VELVET]) + overlay_bias
                velvet_trade_fair = voucher_velvet_fair
                if any(product in state.order_depths for product in VOUCHER_STRIKES):
                    sigmas = self._build_vol_surface(state, voucher_velvet_fair, tte)
                    strip_delta = sum(
                        int(state.position.get(p, 0)) * bs_delta(voucher_velvet_fair, k, tte, sigmas.get(p, 0.18))
                        for p, k in VOUCHER_STRIKES.items()
                    )
                    velvet_pos = int(state.position.get(VELVET, 0))
                    hedge_target = clamp(-0.40 * strip_delta, -80.0, 80.0)
                    hedge_gap = hedge_target - velvet_pos
                    hedge_shift = clamp(hedge_gap / 28.0, -2.20, 2.20)
                    velvet_trade_fair = voucher_velvet_fair + hedge_shift
                velvet_fair = velvet_trade_fair
                result[VELVET] = self._trade_mm(VELVET, state, velvet_trade_fair)
            except Exception as exc:
                memory.setdefault("errors", {})["velvet"] = type(exc).__name__

        if voucher_velvet_fair is not None:
            try:
                if sigmas is None:
                    sigmas = self._build_vol_surface(state, voucher_velvet_fair, tte)
                    strip_delta = sum(
                        int(state.position.get(p, 0)) * bs_delta(voucher_velvet_fair, k, tte, sigmas.get(p, 0.18))
                        for p, k in VOUCHER_STRIKES.items()
                    )
                pair_edge_5100_5200 = 0.0
                if "VEV_5100" in state.order_depths and "VEV_5200" in state.order_depths:
                    residuals_5100_5200: Dict[str, float] = {}
                    for pair_product in ("VEV_5100", "VEV_5200"):
                        od = state.order_depths[pair_product]
                        pair_mid = raw_mid(od)
                        if pair_mid is None:
                            residuals_5100_5200[pair_product] = 0.0
                            continue
                        guarded = max(float(pair_mid), max(voucher_velvet_fair - VOUCHER_STRIKES[pair_product], 0.0) + 1e-3)
                        pair_market_iv = implied_vol(guarded, voucher_velvet_fair, VOUCHER_STRIKES[pair_product], tte)
                        residuals_5100_5200[pair_product] = pair_market_iv - sigmas[pair_product]
                    pair_edge_5100_5200 = residuals_5100_5200["VEV_5100"] - residuals_5100_5200["VEV_5200"]
                pair_edge_5200_5300 = 0.0
                if "VEV_5200" in state.order_depths and "VEV_5300" in state.order_depths:
                    residuals: Dict[str, float] = {}
                    for pair_product in ("VEV_5200", "VEV_5300"):
                        od = state.order_depths[pair_product]
                        pair_mid = raw_mid(od)
                        if pair_mid is None:
                            residuals[pair_product] = 0.0
                            continue
                        guarded = max(float(pair_mid), max(voucher_velvet_fair - VOUCHER_STRIKES[pair_product], 0.0) + 1e-3)
                        pair_market_iv = implied_vol(guarded, voucher_velvet_fair, VOUCHER_STRIKES[pair_product], tte)
                        residuals[pair_product] = pair_market_iv - sigmas[pair_product]
                    pair_edge_5200_5300 = residuals["VEV_5200"] - residuals["VEV_5300"]
                for product in VOUCHER_STRIKES:
                    if product in state.order_depths:
                        result[product] = self._trade_voucher(
                            product,
                            state,
                            voucher_velvet_fair,
                            sigmas[product],
                            strip_delta,
                            tte,
                            pair_edge_5100_5200,
                            pair_edge_5200_5300,
                        )
            except Exception as exc:
                memory.setdefault("errors", {})["vouchers"] = type(exc).__name__

        return result, 0, dump_memory(memory)