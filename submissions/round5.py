import math
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List

POS_LIMIT = 10

# bingchillin had 12 hard exclusions based on latest_trader results alone — that's
# too data-fitted to a single strategy. Cross-referencing with lawdaseries narrows
# the confident exclusions to 3: products that BOTH strategies agree are losers.
#
# Everything else is re-included. Previously-excluded products get higher inv_skew
# as a caution valve: large inventory triggers aggressive quote pullback, limiting
# damage if the product trends against us — softer than a hard ban.
EXCLUDED = {
    "SLEEP_POD_LAMB_WOOL",  # -23.9k in latest_trader, consistently negative
    "PANEL_1X2",             # -18.8k in latest_trader, consistently negative
    "PEBBLES_M",             # -19.3k in latest_trader, -6.5k in lawdaseries
}

# size:     max units quoted per side
# min_half: floor on half-spread quoted (seashells); adapts up when book spread is wide
# inv_skew: seashells of fair-value shift at full position (position / POS_LIMIT);
#           higher = quotes pull back faster as inventory builds
PRODUCT_CFG: Dict[str, Dict] = {
    # From bingbing — validated params, same as bingchillin
    "TRANSLATOR_ECLIPSE_CHARCOAL": {"size": 6, "min_half": 2, "inv_skew": 3},
    "TRANSLATOR_ASTRO_BLACK":      {"size": 6, "min_half": 2, "inv_skew": 3},
    "ROBOT_LAUNDRY":               {"size": 6, "min_half": 2, "inv_skew": 3},
    "SNACKPACK_PISTACHIO":         {"size": 8, "min_half": 3, "inv_skew": 4},
    "SNACKPACK_RASPBERRY":         {"size": 8, "min_half": 3, "inv_skew": 4},
    # Re-included vs bingchillin — lawdaseries showed these can be profitable.
    # Higher inv_skew means we pull quotes back aggressively when holding inventory.
    "ROBOT_MOPPING":               {"size": 6, "min_half": 1, "inv_skew": 5},
    "ROBOT_VACUUMING":             {"size": 6, "min_half": 1, "inv_skew": 5},
    # Re-included vs bingchillin — marginal in lawdaseries, cautious here
    "UV_VISOR_MAGENTA":            {"size": 6, "min_half": 1, "inv_skew": 4},
    "PANEL_4X4":                   {"size": 6, "min_half": 1, "inv_skew": 4},
    "GALAXY_SOUNDS_SOLAR_FLAMES":  {"size": 6, "min_half": 1, "inv_skew": 5},
    "TRANSLATOR_SPACE_GRAY":       {"size": 5, "min_half": 1, "inv_skew": 6},
    "TRANSLATOR_GRAPHITE_MIST":    {"size": 6, "min_half": 1, "inv_skew": 4},
    "PEBBLES_XS":                  {"size": 6, "min_half": 1, "inv_skew": 4},
    "OXYGEN_SHAKE_MINT":           {"size": 6, "min_half": 1, "inv_skew": 4},
}
DEFAULT_CFG = {"size": 8, "min_half": 1, "inv_skew": 2}


def microprice(od: OrderDepth) -> float:
    """Volume-weighted mid — shifts toward the thinner side of the book."""
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

            # Shift fair value against our inventory. Long → fair moves down →
            # our ask tightens and bid pulls back. Naturally flattens the book.
            skewed_fair = fair - inv_skew * (position / POS_LIMIT)
            half = max(min_half, book_spread / 4.0)

            buy_cap = max(0, POS_LIMIT - position)
            sell_cap = max(0, POS_LIMIT + position)
            buy_qty = min(size, buy_cap)
            sell_qty = min(size, sell_cap)

            ords: List[Order] = []

            # Aggressive take: eat resting orders priced through our skewed fair
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

            # Passive quotes inside the book; inventory skew pulls them back
            # naturally when position is large
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