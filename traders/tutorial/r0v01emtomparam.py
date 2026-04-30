# r0v01emtomparam.py: vNN counts only inside the `emtomparam` family (round 0; other families reuse v01 independently).
# Two-pass take-then-make driven by a per-product PARAMS table (momentum tilt on tomato EMA fair).
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# ══════════════════════════════════════════════════════════════════════════════
# PER-PRODUCT PARAMETERS  ← tune / grid-search these
# ══════════════════════════════════════════════════════════════════════════════
PARAMS = {
    "EMERALDS": {
        "fair"          : 10000,   # hard-coded domain knowledge; EMA not used
        "half_spread"   : 1,       # passive quote distance from fair
        "take_edge"     : 1,       # take any ask < fair-edge or bid > fair+edge
        "ema_alpha"     : None,    # not used for Emeralds
        "momentum_w"    : 0.0,     # not used for Emeralds
        "inv_skew_max"  : 2,       # max tick skew from inventory
        "position_limit": 20,
    },
    "TOMATOES": {
        "fair"          : None,    # learned via EMA
        "half_spread"   : 6,       # passive quote distance from EMA fair
        "take_edge"     : 12,      # take aggressively when mispricing > this
        "ema_alpha"     : 0.10,    # EMA smoothing — lower = slower/stabler fair
        "momentum_w"    : 0.20,    # small momentum tilt on fair value estimate
        "inv_skew_max"  : 3,       # max tick skew from inventory
        "position_limit": 20,
    },
}

DEFAULT_PARAMS = {
    "fair"          : None,
    "half_spread"   : 4,
    "take_edge"     : 8,
    "ema_alpha"     : 0.10,
    "momentum_w"    : 0.10,
    "inv_skew_max"  : 2,
    "position_limit": 20,
}


class Trader:
    """
    Best-of-breed two-pass take-then-make market maker.

    Architecture (r0v01emerald):
      Pass 1 — aggressively take any clear mispricing vs fair value.
      Pass 2 — post passive quotes with ALL remaining capacity.

    Improvements over r0v01emerald / r0v01allin:
      • Momentum tilt on EMA fair value (replaces raw mid as signal).
      • Parameterised TAKE_EDGE / HALF_SPREAD per product (grid-searchable).
      • Inventory skew applied on passive quotes (lean against open position).
      • Unified _trade() handles all products; per-product table drives behaviour.
      • Tomatoes fair = EMA(mid) + momentum_w * (EMA - prev_EMA), NOT hardcoded.
    """

    def __init__(self):
        self.ema: Dict[str, float] = {}
        self.prev_ema: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            p      = PARAMS.get(product, DEFAULT_PARAMS)
            limit  = p["position_limit"]
            pos    = state.position.get(product, 0)

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid      = (best_bid + best_ask) / 2.0

            fair = self._fair_value(product, mid, p)

            result[product] = self._trade(
                product, od, pos, limit, fair,
                best_bid, best_ask, p
            )

        return result, 0, ""

    # ──────────────────────────────────────────────────────────────────────────
    def _fair_value(self, product: str, mid: float, p: dict) -> float:
        """
        For products with a known fair value, return it directly.
        Otherwise compute EMA(mid) with a light momentum tilt:
            fair = EMA + momentum_w * (EMA - prev_EMA)
        This nudges the fair value in the direction the slow average is drifting,
        giving a tiny but principled directional edge without overfitting to noise.
        """
        if p["fair"] is not None:
            return float(p["fair"])

        alpha = p["ema_alpha"]

        if product not in self.ema:
            self.ema[product]      = mid
            self.prev_ema[product] = mid
        else:
            prev               = self.ema[product]
            self.prev_ema[product] = prev
            self.ema[product]  = alpha * mid + (1 - alpha) * prev

        momentum_tilt = p["momentum_w"] * (self.ema[product] - self.prev_ema[product])
        return self.ema[product] + momentum_tilt

    # ──────────────────────────────────────────────────────────────────────────
    def _trade(
        self,
        product : str,
        od      : OrderDepth,
        pos     : int,
        limit   : int,
        fair    : float,
        best_bid: int,
        best_ask: int,
        p       : dict,
    ) -> List[Order]:

        orders: List[Order] = []
        running_pos = pos

        take_edge   = p["take_edge"]
        half_spread = p["half_spread"]
        inv_max     = p["inv_skew_max"]

        # ── Pass 1: take mispricings ──────────────────────────────────────────
        # Buy asks priced below fair - take_edge
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

        # Sell bids priced above fair + take_edge
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

        # ── Pass 2: passive quotes with remaining capacity ────────────────────
        buy_cap  = limit - running_pos
        sell_cap = running_pos + limit

        # Inventory skew: lean against position, capped at inv_skew_max ticks
        inv_skew = int(-pos / limit * inv_max)

        buy_px  = round(fair) - half_spread + inv_skew
        sell_px = round(fair) + half_spread + inv_skew

        # Never quote worse than top-of-book (don't improve past best by more than 1)
        buy_px  = min(buy_px, best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        # Safety: don't cross our own quotes
        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
