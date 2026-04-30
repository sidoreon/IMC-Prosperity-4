# r3v03hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datamodel import Order, OrderDepth, TradingState

DEFAULT_LIMIT = 20
_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 32,
    "HYDROGEL_PACK": 32,
    "VEV_4000": 20,
    "VEV_4500": 20,
    "VEV_5000": 20,
    "VEV_5100": 20,
    "VEV_5200": 20,
    "VEV_5300": 20,
    "VEV_5400": 20,
    "VEV_5500": 20,
    "VEV_6000": 20,
    "VEV_6500": 20,
}

UNDERLYING_SYMBOL = "VELVETFRUIT_EXTRACT"
INDEPENDENT_SPOT = "HYDROGEL_PACK"

OPTION_STRIKES: Dict[str, int] = {
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
VEV_LIST = tuple(OPTION_STRIKES.keys())

RISK_FREE_RATE = 0.0

YEAR_FRACTION_BS = 1.0

def _T_years() -> float:
    return max(1.0e-8, float(YEAR_FRACTION_BS))

ENABLE_IV_SMILE = True
ENABLE_IVR_Z = True
ENABLE_PAIR_ARB_BS = False

USE_STABLE_TRADE_SURFACE = True
ENABLE_COINT_PAIRS = True
ENABLE_UMR_UNDERLYING = True

BS_FAIR_LEVEL_BLEND = 0.0

SIGNAL_CONFLICT_DAMP = 0.4

IV_SOLVER_LO, IV_SOLVER_HI = 0.01, 2.5
IV_SOLVER_IT = 55
IV_MIN, IV_MAX = 0.02, 1.8
IV_TRAIL_MAX = 64
SMEAR_MIN_PTS = 3

PAIR_ARB_MIN_EDGE = 4.0
PAIR_ARB_QTY = 1

UMR_FAST = 0.12
UMR_SLOW = 0.004
UMR_DEADBAND_FRAC = 0.0
UMR_STRENGTH = 0.12

DELTA_HEDGE_DEADBAND = 4.0
DELTA_HEDGE_SCALE = 0.42
WIDE_HEDGE_SPREAD = 7

HORIZON = 100
ALPHA_FAIR = 2.0 / (60.0 + 1.0)
ALPHA_SLOW_FAIR = 2.0 / (10000.0 + 1.0)
ALPHA_VOL = 2.0 / (100.0 + 1.0)
WARMUP_TICKS = 20
MICRO_TILT = 0.3
DRIFT_TARGET_SCALE = 2.0
DRIFT_T_THRESHOLD = 10.0
INV_SKEW_K = 2.0
ANTI_TREND_BARRIER_MULT = 3.0
RESIDUAL_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.05,
    "HYDROGEL_PACK": 0.47,
}
SLOW_TARGET_PRODUCTS = frozenset({UNDERLYING_SYMBOL, INDEPENDENT_SPOT})
SLOW_TARGET_SCALES: Dict[str, float] = {
    "VELVETFRUIT_EXTRACT": 1.20,
    "HYDROGEL_PACK": 1.50,
}

STRIKE_VOL_MULT: Dict[int, float] = {
    4000: 1.0,
    4500: 1.0,
    5000: 1.0,
    5100: 1.0,
    5200: 1.0,
    5300: 1.01,
    5400: 0.95,
    5500: 1.03,
    6000: 1.05,
    6500: 1.05,
}
SURFACE_VOL_INIT = 0.032
SURFACE_VOL_ALPHA = 0.20
OPTION_MODEL_BLEND = 0.0

OPTION_TARGET_SCALE = 0.0
OPTION_TARGET_SCALES: Dict[str, float] = {}

COINT_TRIGGER_Z = 1.25
COINT_ENTRY_Z = 2.5
COINT_MAX_PAIR_QTY = 2
COINT_MODEL_BLEND = 0.0
COINT_TARGET_SCALE = 0.0
COINT_PAIRS: Tuple[Tuple[str, str, float, float, float], ...] = (
    ("VEV_4000", "VEV_4500", 499.906, 1.0001, 0.409),
    ("VELVETFRUIT_EXTRACT", "VEV_4500", 4501.328, 0.9982, 0.758),
    ("VEV_5000", "VEV_5100", 70.098, 1.1086, 2.663),
    ("VEV_5100", "VEV_5200", 42.692, 1.2990, 2.188),
    ("VEV_5200", "VEV_5300", 24.333, 1.5230, 1.850),
    ("VEV_5400", "VEV_5500", 3.625, 1.8560, 1.159),
)

_VEV_LADDER: Tuple[str, ...] = VEV_LIST
NEIGHBOR_RESIDUAL_SCALE = 0.10
_VEV_DELTA_W: Dict[str, float] = {
    "VEV_4000": 0.745,
    "VEV_4500": 0.662,
    "VEV_5000": 0.654,
    "VEV_5100": 0.577,
    "VEV_5200": 0.437,
    "VEV_5300": 0.273,
    "VEV_5400": 0.129,
    "VEV_5500": 0.055,
    "VEV_6000": 0.02,
    "VEV_6500": 0.02,
}
_BLOCK_RISK_DIV = 80.0
BLOCK_RISK_SKEW_K = 0.12
USE_WING_THROTTLE = False
WING_VEV: frozenset[str] = frozenset({"VEV_6000", "VEV_6500"})
WING_MAKE_FRAC = 0.55
WING_TAKE_FRAC = 0.6
WING_MAKE_EDGE_MULT = 1.05

def _load_memory(trader_data: str) -> dict:
    if not trader_data:
        return {}
    try:
        m = json.loads(trader_data)
        return m if isinstance(m, dict) else {}
    except (TypeError, ValueError):
        return {}

def _best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    if not od.buy_orders or not od.sell_orders:
        return None, None
    return max(od.buy_orders), min(od.sell_orders)

def _book_mid(od: OrderDepth) -> Optional[float]:
    b, a = _best_bid_ask(od)
    if b is None or a is None:
        return None
    return 0.5 * (b + a)

def _microprice(od: OrderDepth) -> Optional[float]:
    b, a = _best_bid_ask(od)
    if b is None and a is None:
        return None
    if b is None:
        return float(a)
    if a is None:
        return float(b)
    bs = abs(od.buy_orders[b])
    ax = abs(od.sell_orders[a])
    t = bs + ax
    if t <= 0:
        return (b + a) / 2.0
    return (ax * b + bs * a) / t

def _spread(od: OrderDepth) -> float:
    b, a = _best_bid_ask(od)
    if b is None or a is None:
        return 0.0
    return float(a - b)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_price(
    s: float,
    k: float,
    t: float,
    r: float,
    vol: float,
) -> float:
    # Black–Scholes call, continuous div/riskless r.
    if s <= 0.0 or k <= 0.0:
        return 0.0
    intrinsic = max(0.0, s - k)
    if t <= 1e-12 or vol <= 1e-12:
        return intrinsic
    vsqrt = vol * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / vsqrt
    d2 = d1 - vsqrt
    return math.exp(-r * t) * (s * _norm_cdf(d1) - k * _norm_cdf(d2))

def bs_call_delta(
    s: float,
    k: float,
    t: float,
    r: float,
    vol: float,
) -> float:
    if s <= 0.0 or k <= 0.0 or t <= 1e-12 or vol <= 1e-12:
        return 1.0 if s > k else 0.0
    vsqrt = vol * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * vol * vol) * t) / vsqrt
    return math.exp(-r * t) * _norm_cdf(d1)

def implied_vol_bisect(
    s: float,
    k: float,
    t: float,
    r: float,
    price: float,
) -> Optional[float]:
    # Return annualized vol or None if price not in (intrinsic, S).
    if s <= 0.0 or k <= 0.0 or t <= 1e-12:
        return None
    intrinsic = max(0.0, s - k) * math.exp(r * t)
    disc_s = s
    if price <= intrinsic - 0.2 or price >= disc_s + 0.2:
        return None
    lo, hi = IV_SOLVER_LO, IV_SOLVER_HI
    f_lo = bs_call_price(s, k, t, r, lo) - price
    f_hi = bs_call_price(s, k, t, r, hi) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(IV_SOLVER_IT):
        mid = 0.5 * (lo + hi)
        fm = bs_call_price(s, k, t, r, mid) - price
        if abs(fm) < 0.01:
            return mid
        if f_lo * fm <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return 0.5 * (lo + hi)

def _solve_3x3(
    a11: float,
    a12: float,
    a13: float,
    a21: float,
    a22: float,
    a23: float,
    a31: float,
    a32: float,
    a33: float,
    b1: float,
    b2: float,
    b3: float,
) -> Optional[Tuple[float, float, float]]:
    # Cramer's rule for 3x3 (columns: a,b,c for a*m^2 + b*m + c).
    def det3(
        x11: float,
        x12: float,
        x13: float,
        x21: float,
        x22: float,
        x23: float,
        x31: float,
        x32: float,
        x33: float,
    ) -> float:
        return (
            x11 * (x22 * x33 - x23 * x32)
            - x12 * (x21 * x33 - x23 * x31)
            + x13 * (x21 * x32 - x22 * x31)
        )

    d = det3(a11, a12, a13, a21, a22, a23, a31, a32, a33)
    if abs(d) < 1e-12:
        return None
    d1 = det3(b1, a12, a13, b2, a22, a23, b3, a32, a33)
    d2 = det3(a11, b1, a13, a21, b2, a23, a31, b3, a33)
    d3 = det3(a11, a12, b1, a21, a22, b2, a31, a32, b3)
    return d1 / d, d2 / d, d3 / d

def fit_quadratic_smile(
    m_iv: Sequence[Tuple[float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    # Least squares: IV ≈ a*m^2 + b*m + c. Returns (a,b,c, mse) or None.
    if len(m_iv) < SMEAR_MIN_PTS:
        return None
    s11 = s12 = s13 = s22 = s23 = s33 = 0.0
    t1 = t2 = t3 = 0.0
    for m, iv in m_iv:
        m2, m1 = m * m, m
        s11 += m2 * m2
        s12 += m2 * m1
        s13 += m2
        s22 += m1 * m1
        s23 += m1
        s33 += 1.0
        t1 += m2 * iv
        t2 += m1 * iv
        t3 += iv
    sol = _solve_3x3(s11, s12, s13, s12, s22, s23, s13, s23, s33, t1, t2, t3)
    if sol is None:
        return None
    a, b, c = sol

    se = 0.0
    for m, iv in m_iv:
        p = a * m * m + b * m + c
        d = p - iv
        se += d * d
    n = max(1, len(m_iv))
    return a, b, c, se / n

def _iv_trail_update(mem: dict, product: str, iv: float) -> None:
    tr = mem.setdefault("_iv_trails", {})
    lst = tr.get(product) or []
    lst.append(float(iv))
    if len(lst) > IV_TRAIL_MAX:
        lst = lst[-IV_TRAIL_MAX:]
    tr[product] = lst
    w = mem.setdefault("_iv_welford", {})
    s = w.get(product) or [0, 0.0, 0.0]
    n = int(s[0]) + 1
    mean = float(s[1])
    m2 = float(s[2])
    delta = iv - mean
    mean += delta / n
    m2 += delta * (iv - mean)
    w[product] = [n, mean, m2]

def _iv_z(mem: dict, product: str, current_iv: float) -> float:
    w = (mem.get("_iv_welford") or {}).get(product)
    if w is None or int(w[0]) < 3:
        return 0.0
    n, mean, m2 = int(w[0]), float(w[1]), float(w[2])
    var = m2 / max(1, n - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std < 1e-6:
        return 0.0
    return (current_iv - mean) / std

def _surface_from_median(
    s: float,
    order_depths: Dict[str, OrderDepth],
    mem: dict,
) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    st = mem.get("_surface") or {}
    implied_bases: List[float] = []
    t = _T_years()
    for prod, k in OPTION_STRIKES.items():
        if 5000 <= k <= 5500:
            od = order_depths.get(prod)
            if od is None:
                continue
            m = _book_mid(od)
            if m is None:
                continue
            iv0 = implied_vol_bisect(s, float(k), t, RISK_FREE_RATE, m)
            if iv0 is None:
                continue
            mult = STRIKE_VOL_MULT.get(k, 1.0)
            base = iv0 / max(0.5, mult)
            if 0.015 < base < 0.7:
                implied_bases.append(base)
    prev = float(st.get("base_vol", SURFACE_VOL_INIT))
    if not implied_bases:
        bvol = prev
    else:
        implied_bases.sort()
        med = implied_bases[len(implied_bases) // 2]
        bvol = SURFACE_VOL_ALPHA * med + (1.0 - SURFACE_VOL_ALPHA) * prev
        bvol = _clamp(bvol, 0.016, 0.6)
    st["base_vol"] = bvol
    fairs, dels = {}, {}
    for pr, k in OPTION_STRIKES.items():
        v = bvol * STRIKE_VOL_MULT.get(k, 1.0)
        fairs[pr] = bs_call_price(s, float(k), t, RISK_FREE_RATE, v)
        dels[pr] = bs_call_delta(s, float(k), t, RISK_FREE_RATE, v)
    return fairs, dels, st

@dataclass
class EdgeConfig:
    k_take: float
    k_make: float
    min_take: int
    min_make: int
    take_frac: float
    make_frac: float
    vol_floor: float

def _edge_config(product: str, mid: float) -> EdgeConfig:
    c = EdgeConfig(2.0, 0.5, 2, 1, 0.25, 0.125, 0.0)
    if product == INDEPENDENT_SPOT:
        c.min_take, c.min_make = max(c.min_take, 8), max(c.min_make, 4)
        c.k_make, c.take_frac, c.make_frac, c.vol_floor = 0.42, 0.18, 0.19, 1.0
    elif product == UNDERLYING_SYMBOL:
        c.k_take = 2.3
        c.min_take, c.min_make = max(4, c.min_take), max(2, c.min_make)
        c.k_make, c.take_frac, c.make_frac, c.vol_floor = 0.45, 0.14, 0.1, 0.5
    elif product.startswith("VEV_"):
        if product in {"VEV_5000", "VEV_5100"}:
            c.k_take, c.k_make = 2.6, 0.55
            c.min_take, c.min_make = 3, 2
            c.take_frac, c.make_frac = 0.20, 0.10
            c.vol_floor = 0.5
            return c
        if mid < 2.0:
            c.k_take, c.k_make, c.min_take, c.min_make = 1.2, 0.4, 1, 1
            c.take_frac, c.make_frac, c.vol_floor = 0.2, 0.15, 0.35
        elif mid < 30.0:
            c.k_take, c.k_make = 2.4, 0.5
            c.min_take, c.min_make = 1, 1
            c.vol_floor = 0.5
        elif mid < 2000.0:
            c.min_take, c.min_make = max(2, c.min_take), max(2, c.min_make)
        else:
            c.min_take, c.min_make = max(4, c.min_take), max(3, c.min_make)
            c.k_take, c.k_make, c.take_frac, c.make_frac = 2.0, 0.45, 0.2, 0.1
    return c

def _update_online_state(pstate: dict, mid: float, vol_floor: float) -> dict:
    prev_fair = pstate.get("fair")
    prev_vol = float(pstate.get("vol", 1.0))
    if prev_fair is None:
        pstate["fair"] = mid
    else:
        pstate["fair"] = ALPHA_FAIR * mid + (1.0 - ALPHA_FAIR) * float(prev_fair)
    p_slow = pstate.get("slow_fair")
    if p_slow is None:
        pstate["slow_fair"] = mid
    else:
        pstate["slow_fair"] = ALPHA_SLOW_FAIR * mid + (1.0 - ALPHA_SLOW_FAIR) * float(p_slow)
    p_mid0 = pstate.get("prev_mid")
    pstate["prev_mid"] = mid
    if p_mid0 is None:
        pstate.setdefault("ret_n", 0)
        pstate.setdefault("ret_mean", 0.0)
        pstate.setdefault("ret_M2", 0.0)
        pstate["vol"] = max(prev_vol, vol_floor)
        return pstate
    ret = float(mid) - float(p_mid0)
    n = int(pstate.get("ret_n", 0)) + 1
    mean = float(pstate.get("ret_mean", 0.0))
    m2 = float(pstate.get("ret_M2", 0.0))
    delta = ret - mean
    mean += delta / n
    m2 += delta * (ret - mean)
    pstate["ret_n"] = n
    pstate["ret_mean"] = mean
    pstate["ret_M2"] = m2
    pstate["vol"] = max(ALPHA_VOL * abs(ret - mean) + (1.0 - ALPHA_VOL) * prev_vol, vol_floor)
    return pstate

def _umr_state(mem: dict, mid: float) -> Tuple[float, float, float]:
    g = mem.setdefault("_umr", {"fast": None, "slow": None})
    fe = g.get("fast")
    sl = g.get("slow")
    if fe is None:
        g["fast"] = g["slow"] = mid
        return 0.0, float(mid), float(mid)
    g["fast"] = (1.0 - UMR_FAST) * fe + UMR_FAST * mid
    g["slow"] = (1.0 - UMR_SLOW) * (sl or mid) + UMR_SLOW * mid
    fe2, sl2 = float(g["fast"]), float(g["slow"])
    return (fe2 - sl2) / max(1.0, abs(sl2) * UMR_DEADBAND_FRAC + 1.0), fe2, sl2

def _drift_stats(pstate: dict) -> Tuple[float, float]:
    n = int(pstate.get("ret_n", 0))
    if n < 2:
        return 0.0, 0.0
    mean, m2 = float(pstate.get("ret_mean", 0.0)), float(pstate.get("ret_M2", 0.0))
    var = m2 / (n - 1)
    if var <= 0.0:
        return mean, 0.0
    sm = (var / n) ** 0.5
    return mean, mean / sm if sm > 0 else 0.0

def _vev_neighbor_predicted_mid(product: str, ods: Dict[str, OrderDepth]) -> Optional[float]:
    if product not in _VEV_LADDER:
        return None
    i = _VEV_LADDER.index(product)
    acc: List[float] = []
    for j in (i - 1, i + 1):
        if 0 <= j < len(_VEV_LADDER):
            od2 = ods.get(_VEV_LADDER[j])
            if od2 and _book_mid(od2) is not None:
                acc.append(_book_mid(od2) or 0.0)
    if not acc:
        return None
    return float(sum(acc) / len(acc))

def _vev_block_risk(pos: Dict[str, int]) -> float:
    s = 0.0
    for p, w in _VEV_DELTA_W.items():
        s += int(pos.get(p, 0)) * w
    return _clamp(s / _BLOCK_RISK_DIV, -1.0, 1.0)

def _build_bsmile(
    s: float,
    order_depths: Dict[str, OrderDepth],
    mem: dict,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], dict]:
    # Returns: fairs, deltas, obs_iv, iv_residual vs smile, updated memory.
    t = _T_years()
    m_iv: List[Tuple[float, float]] = []
    obs_iv: Dict[str, float] = {}

    for prod, k in OPTION_STRIKES.items():
        od = order_depths.get(prod)
        if od is None:
            continue
        mpx = _book_mid(od)
        if mpx is None or s <= 0:
            continue
        iv0 = implied_vol_bisect(s, float(k), t, RISK_FREE_RATE, mpx)
        if iv0 is None or not (IV_MIN <= iv0 <= IV_MAX):
            continue
        mm = math.log(float(k) / s)
        m_iv.append((mm, float(iv0)))
        obs_iv[prod] = float(iv0)
        _iv_trail_update(mem, prod, float(iv0))

    fairs, dels, iv_res = {}, {}, {}
    fq: Optional[Tuple[float, float, float, float]] = (
        fit_quadratic_smile(m_iv) if (ENABLE_IV_SMILE and len(m_iv) >= SMEAR_MIN_PTS) else None
    )
    if fq is not None:
        a, b, c, _mse = fq
        mem["_smile_abc"] = [a, b, c, len(m_iv)]
        for prod, k in OPTION_STRIKES.items():
            mone = math.log(float(k) / s)
            iv_fit = _clamp(a * mone * mone + b * mone + c, IV_MIN, IV_MAX)
            obs = obs_iv.get(prod)
            if obs is not None:
                iv_res[prod] = float(obs) - float(iv_fit)
            fairs[prod] = bs_call_price(s, float(k), t, RISK_FREE_RATE, iv_fit)
            dels[prod] = bs_call_delta(s, float(k), t, RISK_FREE_RATE, iv_fit)
    else:

        ff, dd, st = _surface_from_median(s, order_depths, mem)
        mem["_surface"] = st
        fairs, dels = ff, dd
        mem["_smile_abc"] = [0, 0, 0, 0]

    return fairs, dels, obs_iv, iv_res, mem

def _planned_positions(
    pos: Dict[str, int], result: Dict[str, List[Order]]
) -> Dict[str, int]:
    out = {p: int(q) for p, q in pos.items()}
    for p, olist in result.items():
        out.setdefault(p, int(pos.get(p, 0)))
        for o in olist:
            out[p] = out.get(p, 0) + int(o.quantity)
    return out

def _append_if_room(
    result: Dict[str, List[Order]],
    plan: Dict[str, int],
    product: str,
    price: int,
    qty: int,
) -> bool:
    if qty == 0:
        return False
    lim = int(_LIMITS.get(product, DEFAULT_LIMIT))
    cur = int(plan.get(product, 0))
    if qty > 0:
        qty = min(qty, lim - cur)
    else:
        qty = -min(-qty, lim + cur)
    if qty == 0:
        return False
    result.setdefault(product, []).append(Order(product, int(price), int(qty)))
    plan[product] = cur + qty
    return True

def _add_cointegration_pair_orders(
    result: Dict[str, List[Order]],
    ods: Dict[str, OrderDepth],
    pos: Dict[str, int],
) -> None:
    if not ENABLE_COINT_PAIRS:
        return
    plan = _planned_positions(pos, result)
    for y, x, al, be, sg in COINT_PAIRS:
        yod, xod = ods.get(y), ods.get(x)
        if yod is None or xod is None or be <= 0.0:
            continue
        yb, ya = _best_bid_ask(yod)
        xb, xa = _best_bid_ask(xod)
        if yb is None or ya is None or xb is None or xa is None:
            continue
        ent = max(1.0, COINT_ENTRY_Z * sg)
        re = yb - al - be * xa
        ch = al + be * xb - ya
        if re > ent:
            yr = min(abs(yod.buy_orders[yb]), _LIMITS.get(y, DEFAULT_LIMIT) + plan.get(y, 0))
            xr = min(abs(xod.sell_orders[xa]), _LIMITS.get(x, DEFAULT_LIMIT) - plan.get(x, 0))
            yq = min(COINT_MAX_PAIR_QTY, yr, int(xr / max(1.0, be)))
            if yq > 0:
                xq = min(xr, max(1, int(round(be * yq))))
                if _append_if_room(result, plan, y, yb, -yq):
                    _append_if_room(result, plan, x, xa, xq)
        elif ch > ent:
            yr = min(abs(yod.sell_orders[ya]), _LIMITS.get(y, DEFAULT_LIMIT) - plan.get(y, 0))
            xr = min(abs(xod.buy_orders[xb]), _LIMITS.get(x, DEFAULT_LIMIT) + plan.get(x, 0))
            yq = min(COINT_MAX_PAIR_QTY, yr, int(xr / max(1.0, be)))
            if yq > 0:
                xq = min(xr, max(1, int(round(be * yq))))
                if _append_if_room(result, plan, y, ya, yq):
                    _append_if_room(result, plan, x, xb, -xq)

def _add_bs_pair_arb(
    result: Dict[str, List[Order]],
    ods: Dict[str, OrderDepth],
    pos: Dict[str, int],
    fairs: Dict[str, float],
) -> None:
    if not ENABLE_PAIR_ARB_BS or not fairs:
        return
    plan = _planned_positions(pos, result)
    for j in range(len(_VEV_LADDER) - 1):
        p0, p1 = _VEV_LADDER[j], _VEV_LADDER[j + 1]
        f0, f1 = fairs.get(p0), fairs.get(p1)
        o0, o1 = ods.get(p0), ods.get(p1)
        if f0 is None or f1 is None or o0 is None or o1 is None:
            continue
        m0, m1 = _book_mid(o0), _book_mid(o1)
        if m0 is None or m1 is None:
            continue
        theo = f1 - f0
        mkt = m1 - m0
        edge = theo - mkt
        if abs(edge) < PAIR_ARB_MIN_EDGE:
            continue
        b0, a0 = _best_bid_ask(o0)
        b1, a1 = _best_bid_ask(o1)
        if b0 is None or a0 is None or b1 is None or a1 is None:
            continue
        if edge > 0.0:
            if _append_if_room(result, plan, p0, a0, PAIR_ARB_QTY) and _append_if_room(
                result, plan, p1, b1, -PAIR_ARB_QTY
            ):
                pass
        else:
            if _append_if_room(result, plan, p0, b0, -PAIR_ARB_QTY) and _append_if_room(
                result, plan, p1, a1, PAIR_ARB_QTY
            ):
                pass

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        mem = _load_memory(state.traderData)
        mem = dict(mem)

        hydrogel_ema: float = float(mem.get("_hydrogel_ema", 10000.0))
        hydrogel_mids: List[float] = list(mem.get("_hydrogel_mids", []))

        res: Dict[str, List[Order]] = {}
        od = state.order_depths.get(INDEPENDENT_SPOT)
        if od is not None:
            pos = int(state.position.get(INDEPENDENT_SPOT, 0))
            orders, hydrogel_ema, hydrogel_mids = self._hydrogel(
                od, pos, hydrogel_ema, hydrogel_mids
            )
            if orders:
                res[INDEPENDENT_SPOT] = orders

        mem["_hydrogel_ema"] = hydrogel_ema
        mem["_hydrogel_mids"] = hydrogel_mids
        return res, 0, json.dumps(mem, separators=(",", ":"))

    FVG_TICK_THRESH = 1.5
    FVG_SIZE        = 15
    MM_OFFSET       = 1
    MM_SIZE         = 50
    INV_SKEW        = 0.05

    def _hydrogel(
        self,
        od: OrderDepth,
        pos: int,
        ema: float,
        mids: List[float],
    ) -> Tuple[List[Order], float, List[float]]:
        # Fair-value-gap hydrogel. Fair value = microprice (imbalance-weighted bid/ask).
        if not od.buy_orders or not od.sell_orders:
            return [], ema, mids

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        ema = 0.01 * mid + 0.99 * ema

        bid_vol = sum(od.buy_orders.values())
        ask_vol = sum(abs(v) for v in od.sell_orders.values())
        total = bid_vol + ask_vol
        micro = (ask_vol * best_bid + bid_vol * best_ask) / total if total > 0 else mid

        limit = 200
        long_cap = limit - pos
        short_cap = limit + pos
        orders: List[Order] = []

        if mids:
            last_return = mid - mids[-1]

            if last_return >= self.FVG_TICK_THRESH and short_cap > 0:
                qty = min(self.FVG_SIZE, short_cap, abs(od.buy_orders[best_bid]))
                if qty > 0:
                    orders.append(Order(INDEPENDENT_SPOT, best_bid, -qty))

            elif last_return <= -self.FVG_TICK_THRESH and long_cap > 0:
                qty = min(self.FVG_SIZE, long_cap, abs(od.sell_orders[best_ask]))
                if qty > 0:
                    orders.append(Order(INDEPENDENT_SPOT, best_ask, qty))

        skew = pos * self.INV_SKEW
        bid_px = round(micro - self.MM_OFFSET - skew)
        ask_px = round(micro + self.MM_OFFSET - skew)

        bid_px = min(bid_px, best_bid + self.MM_OFFSET)
        ask_px = max(ask_px, best_ask - self.MM_OFFSET)
        mm_size = max(0, int(self.MM_SIZE * (1.0 - abs(pos) / limit)))

        if bid_px < ask_px and mm_size > 0:
            if long_cap > 0:
                orders.append(Order(INDEPENDENT_SPOT, bid_px, min(mm_size, long_cap)))
            if short_cap > 0:
                orders.append(Order(INDEPENDENT_SPOT, ask_px, -min(mm_size, short_cap)))

        mids = (mids + [mid])[-3:]
        return orders, ema, mids

    def _adaptive(
        self,
        product: str,
        od: OrderDepth,
        position: int,
        pstate: dict,
        option_fair: Optional[float],
        option_delta: Optional[float],
        coint_fair: Optional[float],
        mem: dict,
        *,
        block_risk: float = 0.0,
        neighbor_pred: Optional[float] = None,
        option_delta_exposure: float = 0.0,
        iv_z: float = 0.0,
        iv_res: float = 0.0,
        hedge_scale: float = 1.0,
    ) -> Tuple[List[Order], dict]:
        _ = option_delta
        out: List[Order] = []
        bb, ba = _best_bid_ask(od)
        if bb is None or ba is None:
            return out, pstate
        mid = (bb + ba) / 2.0
        ec = _edge_config(product, mid)
        pstate = _update_online_state(pstate, float(mid), ec.vol_floor)
        if int(pstate.get("ret_n", 0)) < WARMUP_TICKS:
            return out, pstate

        fair = float(pstate["fair"])
        vol = float(pstate["vol"])
        dpt, tstat = _drift_stats(pstate)
        micro = _microprice(od) or mid
        if abs(tstat) >= DRIFT_T_THRESHOLD:
            ed = dpt * HORIZON
            target_frac = _clamp(ed / DRIFT_TARGET_SCALE, -1.0, 1.0)
        else:
            ed, target_frac = 0.0, 0.0
        bexp = (fair + ed) * (1.0 - MICRO_TILT) + micro * MICRO_TILT
        if option_fair is not None and BS_FAIR_LEVEL_BLEND > 0.0:
            bexp = BS_FAIR_LEVEL_BLEND * option_fair + (1.0 - BS_FAIR_LEVEL_BLEND) * bexp
        if option_fair is not None:
            ex = OPTION_MODEL_BLEND * option_fair + (1.0 - OPTION_MODEL_BLEND) * bexp
        else:
            ex = bexp
        if coint_fair is not None:
            ex = COINT_MODEL_BLEND * coint_fair + (1.0 - COINT_MODEL_BLEND) * ex
        bte = max(float(ec.min_take), ec.k_take * vol)
        rscale = float(RESIDUAL_TARGET_SCALES.get(product, 0.0))
        if product in SLOW_TARGET_PRODUCTS:
            sf = float(pstate.get("slow_fair", fair))
            ss = float(SLOW_TARGET_SCALES.get(product, rscale))
            rtarget = _clamp((sf - mid) / max(1.0, bte), -1.0, 1.0) * ss
        elif coint_fair is not None:
            rtarget = _clamp((coint_fair - mid) / max(1.0, bte), -1.0, 1.0) * COINT_TARGET_SCALE
        elif option_fair is not None:
            ots = float(OPTION_TARGET_SCALES.get(product, OPTION_TARGET_SCALE))
            rtarget = _clamp((option_fair - mid) / max(1.0, bte), -1.0, 1.0) * ots
        else:
            rtarget = _clamp((fair - mid) / max(1.0, bte), -1.0, 1.0) * rscale

        if product.startswith("VEV_") and option_fair is not None and ENABLE_IVR_Z:
            sm_edge = (option_fair - mid) / max(1.0, bte)
            if sm_edge * iv_z < -0.5:
                rtarget *= 1.0 - SIGNAL_CONFLICT_DAMP
        if OPTION_TARGET_SCALE > 0.0 and abs(iv_res) > 0.0 and product.startswith("VEV_") and option_fair is not None:
            rtarget = _clamp(
                rtarget + 0.05 * (iv_res / max(0.1, bte * 0.1)), -1.0, 1.0
            )
        if ENABLE_UMR_UNDERLYING and product == UNDERLYING_SYMBOL:
            umr, _, _ = _umr_state(mem, float(mid))
            rtarget = _clamp(rtarget - UMR_STRENGTH * _clamp(umr / 50.0, -1.0, 1.0), -1.0, 1.0)
        target_frac = _clamp(target_frac + rtarget, -1.0, 1.0)
        if product.startswith("VEV_") and neighbor_pred is not None:
            nrr = _clamp(
                (neighbor_pred - mid) / max(1.0, bte), -1.0, 1.0
            ) * NEIGHBOR_RESIDUAL_SCALE
            target_frac = _clamp(target_frac + nrr, -1.0, 1.0)
        me = max(float(ec.min_make), ec.k_make * vol)
        if USE_WING_THROTTLE and product in WING_VEV:
            me *= WING_MAKE_EDGE_MULT
        if ed > 0.0:
            b_edge = max(float(ec.min_take), bte - max(0.0, ed))
            s_edge = max(float(ec.min_take), bte + max(0.0, ed))
        else:
            b_edge, s_edge = max(float(ec.min_take), bte - ed), max(float(ec.min_take), bte + ed)
        lim = int(_LIMITS.get(product, DEFAULT_LIMIT))
        if product == UNDERLYING_SYMBOL:
            nde = option_delta_exposure + position
            if abs(nde) > DELTA_HEDGE_DEADBAND:
                hs = _clamp(
                    -DELTA_HEDGE_SCALE
                    * hedge_scale
                    * nde
                    / max(1.0, float(lim)),
                    -0.75,
                    0.75,
                )
                target_frac = _clamp(target_frac + hs, -1.0, 1.0)
        tpos = int(round(target_frac * lim))
        c_buy, c_sell = lim - position, lim + position
        tc, mc = max(1, int(lim * ec.take_frac)), max(1, int(lim * ec.make_frac))
        if USE_WING_THROTTLE and product in WING_VEV:
            tc, mc = max(1, int(tc * WING_TAKE_FRAC)), max(1, int(mc * WING_MAKE_FRAC))
        r = tc
        for ap in sorted(od.sell_orders):
            if c_buy <= 0 or r <= 0:
                break
            bt = ap <= ex - b_edge
            mt = option_fair is not None and ap <= option_fair - bte
            ct = coint_fair is not None and ap <= coint_fair - bte
            if not (bt or mt or ct):
                break
            sz = abs(od.sell_orders[ap])
            q = min(sz, c_buy, r)
            if q:
                out.append(Order(product, int(ap), int(q)))
                c_buy, r = c_buy - q, r - q
        r = tc
        for bp in sorted(od.buy_orders, reverse=True):
            if c_sell <= 0 or r <= 0:
                break
            bt2 = bp >= ex + s_edge
            mt2 = option_fair is not None and bp >= option_fair + bte
            ct2 = coint_fair is not None and bp >= coint_fair + bte
            if not (bt2 or mt2 or ct2):
                break
            sz = abs(od.buy_orders[bp])
            q = min(sz, c_sell, r)
            if q:
                out.append(Order(product, int(bp), -int(q)))
                c_sell, r = c_sell - q, r - q
        inv_e = position - tpos
        sk = -INV_SKEW_K * (inv_e / max(1.0, float(lim))) * me
        if product.startswith("VEV_") and BLOCK_RISK_SKEW_K > 0.0:
            sk -= BLOCK_RISK_SKEW_K * block_risk * me
        sk = _clamp(sk, -me, me)
        sb, bb_ = max(0.0, ed) * ANTI_TREND_BARRIER_MULT, max(0.0, -ed) * ANTI_TREND_BARRIER_MULT
        mb, ma = int(round(ex + sk - me - bb_)), int(round(ex + sk + me + sb))
        if bb_ == 0:
            mb = min(mb, bb)
        if sb == 0:
            ma = max(ma, ba)
        if mb >= ba:
            mb = ba - 1
        if ma <= bb:
            ma = bb + 1
        if mb >= ma:
            return out, pstate
        bq, aq = min(mc, c_buy), min(mc, c_sell)
        if bq > 0:
            out.append(Order(product, mb, int(bq)))
        if aq > 0:
            out.append(Order(product, ma, -int(aq)))
        return out, pstate