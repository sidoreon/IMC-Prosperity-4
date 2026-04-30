# r0v01emtomanchor.py: vNN counts only inside the `emtomanchor` family (round 0; other families reuse v01 independently).
# Take-then-make anchored to known fair values (same logic template for every listed product).
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

# Known fair values from competition spec
FAIR_VALUE = {
    "EMERALDS": 10000,
    "TOMATOES": 10000,   # <-- update with actual fair value from the competition
}

class Trader:
    """
    Take-then-make market maker anchored to known fair values.

    Logic (identical for every product):
      1. Aggressively take any ask priced strictly below fair value.
      2. Aggressively take any bid priced strictly above fair value.
      3. Post passive quotes just inside the best bid/ask with all
         remaining position capacity.

    Inventory is controlled purely through position capacity arithmetic —
    no price skew, no tuned multipliers, nothing fitted to historical data.
    """

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "EMERALDS": 20,
            "TOMATOES": 20,
        }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            fair  = FAIR_VALUE.get(product)
            limit = self.position_limit.get(product, 20)
            pos   = state.position.get(product, 0)

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)

            # Fall back to mid-price if this product has no hardcoded fair value
            if fair is None:
                fair = (best_bid + best_ask) / 2.0

            result[product] = self._trade(
                product, od, pos, limit, fair, best_bid, best_ask
            )

        return result, 0, ""

    # ------------------------------------------------------------------
    def _trade(
        self,
        product: str,
        od: OrderDepth,
        pos: int,
        limit: int,
        fair: float,
        best_bid: int,
        best_ask: int,
    ) -> List[Order]:

        orders: List[Order] = []
        p = pos   # track running position as orders are placed

        # ── Pass 1: take mispricings ──────────────────────────────────
        # Buy every ask that is strictly below fair value
        for ask_px in sorted(od.sell_orders):
            if ask_px >= fair:
                break
            room = limit - p
            if room <= 0:
                break
            qty = min(-od.sell_orders[ask_px], room)
            if qty > 0:
                orders.append(Order(product, ask_px, qty))
                p += qty

        # Sell every bid that is strictly above fair value
        for bid_px in sorted(od.buy_orders, reverse=True):
            if bid_px <= fair:
                break
            room = p + limit
            if room <= 0:
                break
            qty = min(od.buy_orders[bid_px], room)
            if qty > 0:
                orders.append(Order(product, bid_px, -qty))
                p -= qty

        # ── Pass 2: passive quotes with remaining capacity ────────────
        buy_cap  = limit - p
        sell_cap = p + limit

        buy_px  = best_bid + 1
        sell_px = best_ask - 1

        # If the spread has collapsed to 1 tick, sit at touch
        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
