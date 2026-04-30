# r0v05emerald.py: vNN counts only inside the `emerald` family (round 0; other families reuse v01 independently).
# Tomato momentum on EMA fair, HALF_SPREAD 0, position limits 80; grid-friendly constants.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    r0v01emerald — fixes applied:
      1. TOMATOES params lifted to class-level constants (easy to grid-search).
      2. EMA alpha made a named constant (was buried inline).
      3. Momentum tilt added to Tomatoes fair value estimate.
    """

    EMERALDS_FAIR = 10000


    # ── Tomatoes params — grid-search these ──────────────────────────────────
    TOM_HALF_SPREAD  = 0
    TOM_TAKE_EDGE    = 12
    TOM_EMA_ALPHA    = 0.10
    TOM_INV_SKEW_MAX = 5
    TOM_MOMENTUM_W   = 0.20   # NEW: momentum tilt weight on EMA fair

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "TOMATOES": 80,
            "EMERALDS": 80,
        }
        self.ema_mid: Dict[str, float] = {}
        self.prev_ema: Dict[str, float] = {}   # NEW: for momentum tilt

    # -------------------------------------------------------------------------
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid      = (best_bid + best_ask) / 2.0
            pos      = state.position.get(product, 0)
            limit    = self.position_limit.get(product, 80)

            if product == "EMERALDS":
                result[product] = self._emeralds(order_depth, pos, limit, best_bid, best_ask)
            else:
                result[product] = self._tomatoes(product, order_depth, pos, limit, mid, best_bid, best_ask)

        return result, 0, ""

    # -------------------------------------------------------------------------
    def _emeralds(self, od, pos, limit, best_bid, best_ask) -> List[Order]:
        # Unchanged from r0v01emerald — hardcoded fair is correct domain knowledge
        fair   = self.EMERALDS_FAIR
        orders = []
        p      = pos

        for ask_px in sorted(od.sell_orders):
            if ask_px >= fair:
                break
            room = limit - p
            if room <= 0:
                break
            qty = min(-od.sell_orders[ask_px], room)
            if qty > 0:
                orders.append(Order("EMERALDS", ask_px, qty))
                p += qty

        for bid_px in sorted(od.buy_orders, reverse=True):
            if bid_px <= fair:
                break
            room = p + limit
            if room <= 0:
                break
            qty = min(od.buy_orders[bid_px], room)
            if qty > 0:
                orders.append(Order("EMERALDS", bid_px, -qty))
                p -= qty

        buy_cap  = limit - p
        sell_cap = p + limit

        buy_px  = best_bid + 1
        sell_px = best_ask - 1

        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order("EMERALDS", buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", sell_px, -sell_cap))

        return orders

    # -------------------------------------------------------------------------
    def _tomatoes(self, product, od, pos, limit, mid, best_bid, best_ask) -> List[Order]:
        # FIX 1: EMA update with named alpha constant
        if product not in self.ema_mid:
            self.ema_mid[product]  = mid
            self.prev_ema[product] = mid
        else:
            self.prev_ema[product] = self.ema_mid[product]
            self.ema_mid[product]  = (
                self.TOM_EMA_ALPHA * mid
                + (1 - self.TOM_EMA_ALPHA) * self.ema_mid[product]
            )

        # FIX 2: momentum tilt — nudges fair in direction EMA is drifting
        momentum_tilt = self.TOM_MOMENTUM_W * (self.ema_mid[product] - self.prev_ema[product])
        fair = self.ema_mid[product] + momentum_tilt

        orders = []
        p      = pos

        for ask_px in sorted(od.sell_orders):
            if ask_px >= fair - self.TOM_TAKE_EDGE:
                break
            room = limit - p
            if room <= 0:
                break
            qty = min(-od.sell_orders[ask_px], room)
            if qty > 0:
                orders.append(Order(product, ask_px, qty))
                p += qty

        for bid_px in sorted(od.buy_orders, reverse=True):
            if bid_px <= fair + self.TOM_TAKE_EDGE:
                break
            room = p + limit
            if room <= 0:
                break
            qty = min(od.buy_orders[bid_px], room)
            if qty > 0:
                orders.append(Order(product, bid_px, -qty))
                p -= qty

        inv_skew = int(-pos / limit * self.TOM_INV_SKEW_MAX)

        buy_px  = round(fair) - self.TOM_HALF_SPREAD + inv_skew
        sell_px = round(fair) + self.TOM_HALF_SPREAD + inv_skew

        buy_px  = min(buy_px,  best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        buy_cap  = limit - p
        sell_cap = p + limit

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
