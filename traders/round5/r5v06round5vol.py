# r5v06round5vol.py: vNN counts only inside the `round5vol` family (same round can have other r5v01… files with different tags).
# Round 5 multi-product book: RV / microprice stack with per-order clip and hard per-product caps.
#
import json
import math
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple

POS_LIMIT = 10

EXCLUDED = {
    "SLEEP_POD_LAMB_WOOL",
    "PANEL_1X2",
    "PEBBLES_M",
}

PRODUCT_CFG: Dict[str, Dict] = {

    "TRANSLATOR_ECLIPSE_CHARCOAL": {"size": 6, "min_half": 2, "inv_skew": 3},
    "TRANSLATOR_ASTRO_BLACK":      {"size": 6, "min_half": 2, "inv_skew": 3},
    "ROBOT_LAUNDRY":               {"size": 6, "min_half": 2, "inv_skew": 3},
    "SNACKPACK_PISTACHIO":         {"size": 8, "min_half": 3, "inv_skew": 4},
    "SNACKPACK_RASPBERRY":         {"size": 8, "min_half": 3, "inv_skew": 4},

    "ROBOT_MOPPING":               {"size": 6, "min_half": 1, "inv_skew": 5},
    "ROBOT_VACUUMING":             {"size": 6, "min_half": 1, "inv_skew": 5},
    "UV_VISOR_MAGENTA":            {"size": 6, "min_half": 1, "inv_skew": 4},
    "PANEL_4X4":                   {"size": 6, "min_half": 1, "inv_skew": 4},
    "GALAXY_SOUNDS_SOLAR_FLAMES":  {"size": 6, "min_half": 1, "inv_skew": 5},
    "TRANSLATOR_SPACE_GRAY":       {"size": 5, "min_half": 1, "inv_skew": 6},
    "TRANSLATOR_GRAPHITE_MIST":    {"size": 6, "min_half": 1, "inv_skew": 4},
    "PEBBLES_XS":                  {"size": 6, "min_half": 1, "inv_skew": 4},
    "OXYGEN_SHAKE_MINT":           {"size": 6, "min_half": 1, "inv_skew": 4},
}
DEFAULT_CFG = {"size": 8, "min_half": 1, "inv_skew": 2}

SPREAD_CONFIGS: List[Tuple] = [
    ("PANEL_2X2",                   "PANEL_4X4",                  -0.807164,  17890.61179,  260.9, 0.70),
    ("UV_VISOR_AMBER",              "UV_VISOR_MAGENTA",            -1.240911,  21922.884435, 371.3, 0.55),
    ("OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_GARLIC",         -0.406479,  13843.508074, 358.6, 0.50),
    ("UV_VISOR_YELLOW",             "UV_VISOR_MAGENTA",             0.744220,   3006.040784, 406.2, 0.45),
    ("GALAXY_SOUNDS_DARK_MATTER",   "GALAXY_SOUNDS_SOLAR_FLAMES",   0.063507,   9559.949840, 326.5, 0.45),
    ("MICROCHIP_OVAL",              "MICROCHIP_TRIANGLE",           0.292957,   6153.576042, 782.5, 0.45),
    ("ROBOT_LAUNDRY",               "ROBOT_IRONING",                0.118598,   9134.334229, 273.8, 0.45),
    ("SLEEP_POD_NYLON",             "SLEEP_POD_COTTON",            -0.144960,  10970.990143, 319.4, 0.40),
    ("SNACKPACK_CHOCOLATE",         "SNACKPACK_VANILLA",           -0.869762,  18667.165060,  58.6, 0.50),
    ("SNACKPACK_VANILLA",           "SNACKPACK_RASPBERRY",          0.012307,   9921.009379, 157.2, 0.35),
]

SPREAD_ALPHA = 0.025

def microprice(od: OrderDepth) -> float:
    # Volume-weighted mid — shifts toward the thinner side of the book.
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    bid_vol = od.buy_orders[best_bid]
    ask_vol = abs(od.sell_orders[best_ask])
    total = bid_vol + ask_vol
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_bid * ask_vol + best_ask * bid_vol) / total

def spread_biases(
    mids: Dict[str, float],
    spread_mem: Dict[str, Dict],
) -> Dict[str, float]:
    # For each cointegration pair, compute a z-score of the current spread vs its adaptive EMA. Translate into a per-prod...
    biases: Dict[str, float] = {}

    for a, b, beta, intercept, default_std, weight in SPREAD_CONFIGS:
        if a not in mids or b not in mids:
            continue

        spread = mids[a] - beta * mids[b] - intercept
        key = f"{a}|{b}"
        item = spread_mem.get(key)
        if item is None:

            item = {"ema": spread, "var": default_std * default_std}

        ema = float(item["ema"])
        var = max(25.0, float(item["var"]))

        resid = spread - ema
        ema += SPREAD_ALPHA * resid
        var = (1.0 - SPREAD_ALPHA) * var + SPREAD_ALPHA * resid * resid
        spread_mem[key] = {"ema": round(ema, 4), "var": round(var, 4)}

        z = (spread - ema) / max(default_std * 0.35, math.sqrt(var))
        z = max(-4.0, min(4.0, z))
        if abs(z) < 0.75:
            continue

        edge = weight * z
        biases[a] = biases.get(a, 0.0) - edge
        b_scale = max(0.25, min(1.25, abs(beta)))
        biases[b] = biases.get(b, 0.0) + edge * b_scale

    return biases

class Trader:
    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        spread_mem: Dict[str, Dict] = saved.get("spread", {})

        mids: Dict[str, float] = {}
        for product, od in state.order_depths.items():
            if od.buy_orders and od.sell_orders:
                mids[product] = (max(od.buy_orders) + min(od.sell_orders)) / 2.0

        biases = spread_biases(mids, spread_mem)

        orders: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product in EXCLUDED:
                continue
            if not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            book_spread = best_ask - best_bid
            if book_spread <= 0:
                continue

            fair = microprice(od) + biases.get(product, 0.0)
            position = state.position.get(product, 0)
            cfg = PRODUCT_CFG.get(product, DEFAULT_CFG)
            size = cfg["size"]
            min_half = cfg["min_half"]
            inv_skew = cfg["inv_skew"]

            skewed_fair = fair - inv_skew * (position / POS_LIMIT)
            half = max(min_half, book_spread / 4.0)

            buy_cap = max(0, POS_LIMIT - position)
            sell_cap = max(0, POS_LIMIT + position)
            buy_qty = min(size, buy_cap)
            sell_qty = min(size, sell_cap)

            ords: List[Order] = []

            if buy_qty > 0 and best_ask < skewed_fair - half:
                take = min(abs(od.sell_orders[best_ask]), buy_qty)
                if take > 0:
                    ords.append(Order(product, best_ask, take))
                    buy_qty -= take

            if sell_qty > 0 and best_bid > skewed_fair + half:
                take = min(od.buy_orders[best_bid], sell_qty)
                if take > 0:
                    ords.append(Order(product, best_bid, -take))
                    sell_qty -= take

            passive_bid = min(best_bid + 1, math.floor(skewed_fair - half))
            passive_ask = max(best_ask - 1, math.ceil(skewed_fair + half))

            if (buy_qty > 0
                    and 0 < passive_bid < best_ask
                    and skewed_fair - passive_bid >= half):
                ords.append(Order(product, passive_bid, buy_qty))

            if (sell_qty > 0
                    and passive_ask > best_bid
                    and passive_ask > passive_bid
                    and passive_ask - skewed_fair >= half):
                ords.append(Order(product, passive_ask, -sell_qty))

            if ords:
                orders[product] = ords

        trader_data = json.dumps({"spread": spread_mem}, separators=(",", ":"))
        return orders, 0, trader_data
