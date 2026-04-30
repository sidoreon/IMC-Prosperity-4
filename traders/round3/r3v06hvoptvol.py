# r3v06hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
from datamodel import OrderDepth, UserId, TradingState, Order
import json
import math
from collections import defaultdict

class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str):
        print(json.dumps([
            [
                state.timestamp,
                state.traderData,
                [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
                {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for trades in state.own_trades.values() for t in trades],
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for trades in state.market_trades.values() for t in trades],
                dict(state.position),
                [state.observations.plainValueObservations, {
                    p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                    for p, o in state.observations.conversionObservations.items()
                }],
            ],
            [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
            conversions,
            trader_data,
            self.logs,
        ], separators=(",", ":")))
        self.logs = ""

logger = Logger()

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    # European call price (r=0). T in years, sigma annualised.
    if T <= 1e-9:
        return max(S - K, 0.0)
    if sigma <= 1e-9:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def _implied_vol(S: float, K: float, T: float, price: float):
    # Bisection IV search; returns None if ill-conditioned.
    if T <= 1e-9:
        return None
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9:
        return None
    lo, hi = 1e-6, 5.0
    if _bs_call(S, K, T, hi) < price:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _bs_call(S, K, T, mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    return (lo + hi) / 2.0

def _solve3x3(A: list, b: list):
    # Gaussian elimination for 3×3 system Ax=b. Returns x or None.
    import copy
    M = [A[i][:] + [b[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(M[r][col]))
        M[col], M[pivot] = M[pivot], M[col]
        if abs(M[col][col]) < 1e-15:
            return None
        for row in range(col + 1, 3):
            f = M[row][col] / M[col][col]
            for j in range(col, 4):
                M[row][j] -= f * M[col][j]
    x = [0.0] * 3
    for i in range(2, -1, -1):
        x[i] = M[i][3]
        for j in range(i + 1, 3):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x

def _fit_quadratic(xs: list, ys: list):
    # Least-squares quadratic fit y = a*x^2 + b*x + c. Returns (a, b, c) or None if not enough data.
    n = len(xs)
    if n < 3:
        return None
    s4 = s3 = s2 = s1 = s0 = sy2 = sy1 = sy0 = 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        s4 += x2 * x2
        s3 += x2 * x
        s2 += x2
        s1 += x
        s0 += 1.0
        sy2 += x2 * y
        sy1 += x * y
        sy0 += y
    A = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    b = [sy2, sy1, sy0]
    return _solve3x3(A, b)

VOUCHER_STRIKES = {
    'VEV_4000': 4000, 'VEV_4500': 4500,
    'VEV_5000': 5000, 'VEV_5100': 5100,
    'VEV_5200': 5200, 'VEV_5300': 5300,
    'VEV_5400': 5400, 'VEV_5500': 5500,
    'VEV_6000': 6000, 'VEV_6500': 6500,
}
DEEP_OTM = {'VEV_6000', 'VEV_6500'}

VOUCHER_MM = {
    'VEV_4000': (300, 5), 'VEV_4500': (300, 5),
    'VEV_5000': (200, 3), 'VEV_5100': (200, 3),
    'VEV_5200': (100, 2), 'VEV_5300': (100, 2),
    'VEV_5400': (50,  1), 'VEV_5500': (50,  1),
}
MID_HIST_WINDOW = 11

class Trader:
    def run(self, state: TradingState):
        result = defaultdict(list)
        conversions = 0
        shared = {}
        if state.traderData:
            try:
                shared = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        result['HYDROGEL_PACK'], hp_data = self.hydrogel(state, shared)
        result['VELVETFRUIT_EXTRACT'], vev_data = self.vev_spot(state, shared)
        opt_orders, opt_data = self.vev_options(state, shared)
        for sym, orders in opt_orders.items():
            result[sym] = orders

        traderData = json.dumps({**hp_data, **vev_data, **opt_data})
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def hydrogel(self, state: TradingState, shared: dict):
        product = 'HYDROGEL_PACK'
        result = []
        pos_lim = 200
        quote_lim = 15
        fair = 10000

        if product not in state.order_depths:
            return result, {}

        od = state.order_depths[product]
        pos = state.position.get(product, 0)
        bids = sorted(od.buy_orders, reverse=True)
        asks = sorted(od.sell_orders)

        buy_cap = pos_lim - pos
        sell_cap = pos_lim + pos

        for ask in asks:
            if ask < fair and buy_cap > 0:
                qty = min(buy_cap, abs(od.sell_orders[ask]))
                result.append(Order(product, ask, qty))
                buy_cap -= qty
        for bid in bids:
            if bid > fair and sell_cap > 0:
                qty = min(sell_cap, od.buy_orders[bid])
                result.append(Order(product, bid, -qty))
                sell_cap -= qty

        if buy_cap > 0:
            result.append(Order(product, fair - 1, min(quote_lim, buy_cap)))
        if sell_cap > 0:
            result.append(Order(product, fair + 1, -min(quote_lim, sell_cap)))

        return result, {}

    def vev_spot(self, state: TradingState, shared: dict):
        product = 'VELVETFRUIT_EXTRACT'
        result = []
        pos_lim = 200
        quote_lim = 10
        sma_window = 5

        bid_hist = shared.get("vev_bid_hist", [])
        ask_hist = shared.get("vev_ask_hist", [])

        if product not in state.order_depths:
            return result, {"vev_bid_hist": bid_hist, "vev_ask_hist": ask_hist}

        od = state.order_depths[product]
        bids = sorted(od.buy_orders, reverse=True)
        asks = sorted(od.sell_orders)
        best_bid = bids[0] if bids else (bid_hist[-1] if bid_hist else None)
        best_ask = asks[0] if asks else (ask_hist[-1] if ask_hist else None)

        if best_bid is not None:
            bid_hist.append(best_bid)
        if best_ask is not None:
            ask_hist.append(best_ask)
        if len(bid_hist) > sma_window:
            bid_hist.pop(0)
        if len(ask_hist) > sma_window:
            ask_hist.pop(0)

        if len(bid_hist) < sma_window or len(ask_hist) < sma_window:
            return result, {"vev_bid_hist": bid_hist, "vev_ask_hist": ask_hist}

        fair = (sum(bid_hist) + sum(ask_hist)) / (2.0 * sma_window)
        pos = state.position.get(product, 0)
        buy_cap = pos_lim - pos
        sell_cap = pos_lim + pos

        for ask in asks:
            if ask <= fair - 1 and buy_cap > 0:
                qty = min(buy_cap, abs(od.sell_orders[ask]))
                result.append(Order(product, ask, qty))
                buy_cap -= qty
                break
        for bid in bids:
            if bid >= fair + 1 and sell_cap > 0:
                qty = min(sell_cap, od.buy_orders[bid])
                result.append(Order(product, bid, -qty))
                sell_cap -= qty
                break

        pb = next((p for p in bids if p + 1 <= fair), None)
        pa = next((p for p in asks if p - 1 >= fair), None)
        if pb is None and best_bid is not None and best_bid + 1 <= fair:
            pb = best_bid
        if pa is None and best_ask is not None and best_ask - 1 >= fair:
            pa = best_ask
        if pb is not None and buy_cap > 0:
            result.append(Order(product, pb + 1, min(quote_lim, buy_cap)))
        if pa is not None and sell_cap > 0:
            result.append(Order(product, pa - 1, -min(quote_lim, sell_cap)))

        return result, {"vev_bid_hist": bid_hist, "vev_ask_hist": ask_hist}

    def vev_options(self, state: TradingState, shared: dict):
        result = defaultdict(list)
        mid_hist = shared.get("vev_opt_mid_hist", {})

        for sym, (pos_lim, quote_qty) in VOUCHER_MM.items():
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders:
                continue
            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid = (best_bid + best_ask) / 2.0

            hist = mid_hist.get(sym, [])
            hist.append(mid)
            if len(hist) > MID_HIST_WINDOW:
                hist.pop(0)
            mid_hist[sym] = hist
            fair = sum(hist) / len(hist)
            fair_int = round(fair)

            pos = state.position.get(sym, 0)
            buy_cap = pos_lim - pos
            sell_cap = pos_lim + pos

            if buy_cap > 0:
                result[sym].append(Order(sym, fair_int - 1, min(quote_qty, buy_cap)))
            if sell_cap > 0:
                result[sym].append(Order(sym, fair_int + 1, -min(quote_qty, sell_cap)))

        return result, {"vev_opt_mid_hist": mid_hist}