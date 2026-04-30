# r4v02hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r4v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
import json
import math
from datamodel import Order, OrderDepth, Symbol, TradingState

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T_days: float, sigma: float = 0.013) -> float:
    if T_days <= 0.0 or sigma <= 0.0:
        return max(0.0, S - K)
    sv = sigma * math.sqrt(T_days)
    d1 = (math.log(S / K) + 0.5 * sigma ** 2 * T_days) / sv
    return S * _norm_cdf(d1) - K * _norm_cdf(d1 - sv)

SIGMA           = 0.013
TTE_START_DAYS  = 4.0
TICKS_PER_DAY   = 1_000_000

HYDROGEL_FV     = 9991
HYDROGEL_HALF   = 7
HYDROGEL_LIMIT  = 200
HYDROGEL_PASS_SZ = 30

VEV_MM_HALF     = 2
VEV_LIMIT       = 200
VEV_PASS_SZ     = 10

SHORT_STRIKES   = {5300: 300, 5400: 300, 5500: 300}
SHORT_THRESH    = 50

BUY_DIP_THRESH  = 50
BUY_RECOVER     = 25
BUY_STRIKES     = {5000: 300, 5100: 300, 5200: 300}
BUY_THRESH      = 50

VOUCHER_LIMIT   = 300

def mid(od: OrderDepth) -> float | None:
    b, a = od.buy_orders, od.sell_orders
    if b and a:
        return (max(b) + min(a)) / 2.0
    return float(max(b)) if b else (float(min(a)) if a else None)

def get_T(timestamp: int) -> float:
    return max(TTE_START_DAYS - timestamp / TICKS_PER_DAY, 1e-4)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def mark38_signal(market_trades: dict) -> int:
    # +1=lean short (Mark38 bought), -1=lean long (Mark38 sold), 0=none
    trades = market_trades.get("HYDROGEL_PACK", [])
    if not trades:
        return 0
    recent = trades[-10:]
    bought = any(getattr(t, "buyer", "") == "Mark 38" for t in recent)
    sold   = any(getattr(t, "seller", "") == "Mark 38" for t in recent)
    if bought and not sold:
        return 1
    if sold and not bought:
        return -1
    return 0

def trade_hydrogel(od: OrderDepth, position: int,
                   orders: list, signal: int = 0) -> None:
    fv   = HYDROGEL_FV
    half = HYDROGEL_HALF
    pos  = position
    skew = clamp(-pos * 0.03, -half, half)

    sell_thresh = fv + half - (2 if signal > 0 else 0)
    buy_thresh  = fv - half + (2 if signal < 0 else 0)

    for px in sorted(od.buy_orders, reverse=True):
        if px <= sell_thresh or pos <= -HYDROGEL_LIMIT:
            break
        qty = min(od.buy_orders[px], pos + HYDROGEL_LIMIT)
        if qty > 0:
            orders.append(Order("HYDROGEL_PACK", px, -qty))
            pos -= qty

    for px in sorted(od.sell_orders):
        if px >= buy_thresh or pos >= HYDROGEL_LIMIT:
            break
        qty = min(-od.sell_orders[px], HYDROGEL_LIMIT - pos)
        if qty > 0:
            orders.append(Order("HYDROGEL_PACK", px, qty))
            pos += qty

    bq = min(HYDROGEL_LIMIT - pos, HYDROGEL_PASS_SZ)
    sq = min(pos + HYDROGEL_LIMIT, HYDROGEL_PASS_SZ)
    if bq > 0:
        orders.append(Order("HYDROGEL_PACK", int(fv + skew - half), bq))
    if sq > 0:
        orders.append(Order("HYDROGEL_PACK", int(fv + skew + half) + 1, -sq))

def trade_vev(od: OrderDepth, position: int, orders: list) -> None:
    # MM at mid±2 vs Mark 55 (noise trader crossing ±2.5). NO delta hedge.
    S = mid(od)
    if S is None:
        return
    pos  = position
    half = VEV_MM_HALF
    skew = clamp(-pos * 0.05, -half, half)

    bid_px = int(math.floor(S + skew - half))
    ask_px = int(math.ceil(S + skew + half))
    if ask_px <= bid_px:
        ask_px = bid_px + 1

    for px in sorted(od.sell_orders):
        if px >= S - half or pos >= VEV_LIMIT:
            break
        qty = min(-od.sell_orders[px], VEV_LIMIT - pos)
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", px, qty))
            pos += qty

    for px in sorted(od.buy_orders, reverse=True):
        if px <= S + half or pos <= -VEV_LIMIT:
            break
        qty = min(od.buy_orders[px], pos + VEV_LIMIT)
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", px, -qty))
            pos -= qty

    bq = min(VEV_LIMIT - pos, VEV_PASS_SZ)
    sq = min(pos + VEV_LIMIT, VEV_PASS_SZ)
    if bq > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, bq))
    if sq > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, -sq))

def trade_options(state: TradingState, S: float, T_days: float,
                  vev_start: float, orders: dict) -> None:
    # Phase 1 (default): short 5300/5400/5500. Phase 2 (if VEV dips 50pts from start): buy 5000/5100/5200 instead.
    in_dip_mode = (S < vev_start - BUY_DIP_THRESH)

    if in_dip_mode:

        for strike, cap in BUY_STRIKES.items():
            sym = f"VEV_{strike}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            pos = state.position.get(sym, 0)
            fv  = bs_call(S, strike, T_days)
            o   = []
            for ask_px in sorted(od.sell_orders):
                if ask_px > fv + BUY_THRESH or pos >= cap:
                    break
                qty = min(-od.sell_orders[ask_px], cap - pos)
                if qty > 0:
                    o.append(Order(sym, ask_px, qty))
                    pos += qty
            if o:
                orders[sym] = o
    else:

        for strike, cap in SHORT_STRIKES.items():
            sym = f"VEV_{strike}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            pos = state.position.get(sym, 0)
            fv  = bs_call(S, strike, T_days)
            o   = []
            for bid_px in sorted(od.buy_orders, reverse=True):
                if fv - bid_px > SHORT_THRESH or pos <= -cap:
                    break
                qty = min(od.buy_orders[bid_px], pos + cap)
                if qty > 0:
                    o.append(Order(sym, bid_px, -qty))
                    pos -= qty
            if o:
                orders[sym] = o

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        T_days = get_T(state.timestamp)

        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = mid(vev_od) if vev_od else td.get("last_S", 5232.0)
        if S is None:
            S = td.get("last_S", 5232.0)
        td["last_S"] = S

        if "vev_start" not in td:
            td["vev_start"] = S
        vev_start = td["vev_start"]

        if "HYDROGEL_PACK" in state.order_depths:
            signal = mark38_signal(state.market_trades)
            o: list[Order] = []
            trade_hydrogel(state.order_depths["HYDROGEL_PACK"],
                           state.position.get("HYDROGEL_PACK", 0), o, signal)
            if o:
                result["HYDROGEL_PACK"] = o

        if vev_od:
            o = []
            trade_vev(vev_od, state.position.get("VELVETFRUIT_EXTRACT", 0), o)
            if o:
                result["VELVETFRUIT_EXTRACT"] = o

        trade_options(state, S, T_days, vev_start, result)

        return result, 0, json.dumps(td)