# r0v02allin.py: vNN counts only inside the `allin` family (round 0; other families reuse v01 independently).
# Same architecture as r0v01allin with per-product PARAMS; tomato fair from EMA (not hardcoded 10000).
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# FIX 1: per-product fair values and params — replaces single hardcoded dict
# For products with no known fair value, set "fair" to None → EMA is used
PARAMS: Dict[str, dict] = {
    "EMERALDS": {
        "fair"          : 10000,   # domain knowledge; stable stablecoin
        "half_spread"   : 1,
        "take_edge"     : 1,
        "ema_alpha"     : None,    # unused when fair is hardcoded
        "inv_skew_max"  : 2,
        "position_limit": 20,
    },
    "TOMATOES": {
        "fair"          : None,    # FIX 2: was wrong hardcoded 10000 — now uses EMA
        "half_spread"   : 6,
        "take_edge"     : 12,
        "ema_alpha"     : 0.10,
        "inv_skew_max"  : 2,
        "position_limit": 20,
    },
}

DEFAULT_PARAMS = {
    "fair"          : None,
    "half_spread"   : 4,
    "take_edge"     : 8,
    "ema_alpha"     : 0.10,
    "inv_skew_max"  : 2,
    "position_limit": 20,
}


class Trader:
    """
    r0v01allin — fixes applied:
      1. Tomatoes fair value: was hardcoded 10000 (placeholder/wrong) → EMA of mid.
      2. Per-product params table: each product gets its own tuned constants.
      3. Unified _trade() preserved — clean single-path architecture kept.
    """

    def __init__(self):
        self.ema: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            p     = PARAMS.get(product, DEFAULT_PARAMS)
            limit = p["position_limit"]
            pos   = state.position.get(product, 0)

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid      = (best_bid + best_ask) / 2.0

            fair = self._fair_value(product, mid, p)

            result[product] = self._trade(
                product, od, pos, limit, fair, best_bid, best_ask, p
            )

        return result, 0, ""

    # -------------------------------------------------------------------------
    def _fair_value(self, product: str, mid: float, p: dict) -> float:
        if p["fair"] is not None:
            return float(p["fair"])

        alpha = p["ema_alpha"]
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = alpha * mid + (1 - alpha) * self.ema[product]
        return self.ema[product]

    # -------------------------------------------------------------------------
    def _trade(self, product, od, pos, limit, fair, best_bid, best_ask, p) -> List[Order]:
        orders: List[Order] = []
        running_pos = pos

        take_edge   = p["take_edge"]
        half_spread = p["half_spread"]
        inv_max     = p["inv_skew_max"]

        # Pass 1: take mispricings
        for ask_px in sorted(od.sell_orders):
            if ask_px >= fair - take_edge:
                break
            room = limit - running_pos
            if room <= 0:
                break
            qty = min(-od.sell_orders[ask_px], room)
            if qty > 0:
                orders.append(Order(product, ask_px, qty))
                running_pos += qty

        for bid_px in sorted(od.buy_orders, reverse=True):
            if bid_px <= fair + take_edge:
                break
            room = running_pos + limit
            if room <= 0:
                break
            qty = min(od.buy_orders[bid_px], room)
            if qty > 0:
                orders.append(Order(product, bid_px, -qty))
                running_pos -= qty

        # Pass 2: passive quotes with remaining capacity
        buy_cap  = limit - running_pos
        sell_cap = running_pos + limit

        inv_skew = int(-pos / limit * inv_max)

        buy_px  = round(fair) - half_spread + inv_skew
        sell_px = round(fair) + half_spread + inv_skew

        buy_px  = min(buy_px,  best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
