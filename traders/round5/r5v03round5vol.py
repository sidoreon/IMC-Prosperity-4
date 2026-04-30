# r5v03round5vol.py: vNN counts only inside the `round5vol` family (same round can have other r5v01… files with different tags).
# Round 5 multi-product book: RV / microprice stack with per-order clip and hard per-product caps.
#
import math
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional

POS_LIMIT = 10

EXCLUDED = {
    "SLEEP_POD_LAMB_WOOL",
    "PANEL_1X2",
    "PEBBLES_M",
    "ROBOT_MOPPING",
    "UV_VISOR_MAGENTA",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "PANEL_4X4",
    "TRANSLATOR_SPACE_GRAY",
    "ROBOT_VACUUMING",
    "TRANSLATOR_GRAPHITE_MIST",
    "PEBBLES_XS",
    "OXYGEN_SHAKE_MINT",
}

PRODUCT_CFG: Dict[str, Dict] = {

    "TRANSLATOR_ECLIPSE_CHARCOAL": {"size": 6, "min_half": 2, "inv_skew": 3},
    "TRANSLATOR_ASTRO_BLACK":      {"size": 6, "min_half": 2, "inv_skew": 3},
    "ROBOT_LAUNDRY":               {"size": 6, "min_half": 2, "inv_skew": 3},
    "SNACKPACK_PISTACHIO":         {"size": 8, "min_half": 3, "inv_skew": 4},
    "SNACKPACK_RASPBERRY":         {"size": 8, "min_half": 3, "inv_skew": 4},
}
DEFAULT_CFG = {"size": 8, "min_half": 1, "inv_skew": 2}

def microprice(od: OrderDepth) -> float:
    # Volume-weighted price between best bid and best ask. Shifts toward the thinner side of the book — if there's 1 unit...
    best_bid = max(od.buy_orders)
    best_ask = min(od.sell_orders)
    bid_vol = od.buy_orders[best_bid]
    ask_vol = abs(od.sell_orders[best_ask])
    total = bid_vol + ask_vol
    if total <= 0:
        return (best_bid + best_ask) / 2.0
    return (best_bid * ask_vol + best_ask * bid_vol) / total

class Trader:
    def run(self, state: TradingState):
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

            fair = microprice(od)
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

        return orders, 0, ""
