# r0v01allin.py: vNN counts only inside the `allin` family (round 0; other families reuse v01 independently).
# Take-then-make v2: hard emerald fair, EMA tomato with take and passive legs (class-level constants).
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    v2 — take-then-make strategy

    EMERALDS:
      - Hard-coded fair value = 10000 (it's a stablecoin; mid never leaves 9996-10004)
      - Pass 1: aggressively take any ask < fair or bid > fair (free money)
      - Pass 2: passive quote just inside spread with ALL remaining capacity
      - No inventory-driven price skew — position limits alone control exposure

    TOMATOES:
      - Fair value = EMA(mid, alpha=0.1) — tracks the slow drift without overfitting noise
      - Pass 1: take any ask more than EDGE ticks below fair or bid above fair
      - Pass 2: passive quote at fair ± HALF_SPREAD with remaining capacity
      - Light inventory skew (max ±2 ticks) so we don't swing wildly
    """

    EMERALDS_FAIR = 10000

    # TOMATOES quoting params
    TOM_HALF_SPREAD  = 6    # quote at fair ± 6
    TOM_TAKE_EDGE    = 12   # take aggressively when mispricing > 12 ticks from fair
    TOM_EMA_ALPHA    = 0.10
    TOM_INV_SKEW_MAX = 2    # at most ±2 tick price skew from inventory

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "TOMATOES": 20,
            "EMERALDS": 20,
        }
        self.ema_mid: Dict[str, float] = {}

    # ------------------------------------------------------------------
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
            limit    = self.position_limit.get(product, 20)

            if product == "EMERALDS":
                result[product] = self._emeralds(order_depth, pos, limit, best_bid, best_ask)
            else:
                result[product] = self._tomatoes(product, order_depth, pos, limit, mid, best_bid, best_ask)

        return result, 0, ""

    # ------------------------------------------------------------------
    def _emeralds(
        self,
        od: OrderDepth,
        pos: int,
        limit: int,
        best_bid: int,
        best_ask: int,
    ) -> List[Order]:

        fair   = self.EMERALDS_FAIR
        orders = []
        p      = pos   # local copy we update as we place orders

        # ── Pass 1: take mispricings ──────────────────────────────────
        # Buy any ask strictly below fair value
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

        # Sell any bid strictly above fair value
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

        # ── Pass 2: passive market making with remaining capacity ─────
        buy_cap  = limit - p
        sell_cap = p + limit

        buy_px  = best_bid + 1
        sell_px = best_ask - 1

        # Protect against a crossed/1-tick spread
        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order("EMERALDS", buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order("EMERALDS", sell_px, -sell_cap))

        return orders

    # ------------------------------------------------------------------
    def _tomatoes(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        limit: int,
        mid: float,
        best_bid: int,
        best_ask: int,
    ) -> List[Order]:

        # Update EMA fair value
        if product not in self.ema_mid:
            self.ema_mid[product] = mid
        else:
            self.ema_mid[product] = (
                self.TOM_EMA_ALPHA * mid
                + (1 - self.TOM_EMA_ALPHA) * self.ema_mid[product]
            )
        fair = self.ema_mid[product]

        orders = []
        p      = pos

        # ── Pass 1: take clear mispricings vs fair ────────────────────
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

        # ── Pass 2: passive quote around fair value ───────────────────
        # Tiny inventory skew — lean against open position, max ±2 ticks
        inv_skew = int(-pos / limit * self.TOM_INV_SKEW_MAX)

        buy_px  = round(fair) - self.TOM_HALF_SPREAD + inv_skew
        sell_px = round(fair) + self.TOM_HALF_SPREAD + inv_skew

        # Never cross inside best bid/ask in a way that costs us
        buy_px  = min(buy_px,  best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        # Ensure we're not crossing
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
