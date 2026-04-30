# r0v01emtomlinreg.py: vNN counts only inside the `emtomlinreg` family (round 0; other families reuse v01 independently).
# Rolling linear regression on tomato mids for dynamic fair; fixed fair for EMERALDS from spec.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
from collections import deque

# ── Known fair values (from competition spec) ─────────────────────────────────
# Products listed here skip regression and use the hardcoded value.
FIXED_FAIR: Dict[str, float] = {
    "EMERALDS": 10000,
}

# ── Rolling-regression window ─────────────────────────────────────────────────
# How many mid-price observations to include in the local linear fit.
# Large  → smoother but slower to react to genuine shifts.
# Small  → more responsive but more noise-sensitive.
# 30 ticks ≈ 3 000 timestamps, a reasonable middle ground.
WINDOW = 30


class Trader:
    """
    Dynamic fair-value market maker.

    For products with a known fair value (EMERALDS = 10 000), use it directly.

    For everything else (TOMATOES), estimate fair value each tick with a
    rolling ordinary-least-squares line fit to the last WINDOW mid prices,
    evaluated at the current (most recent) tick.  This de-noises the price
    without hard-coding any level derived from back-testing data.

    Execution: identical take-then-make logic for all products.
      Pass 1 — aggressively take any ask < fair or bid > fair.
      Pass 2 — post passive quotes just inside the spread with all remaining
               position capacity.  No inventory price-skew; position limits
               handle sizing.
    """

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "EMERALDS": 20,
            "TOMATOES": 20,
        }
        # Rolling mid-price history for regression-based products
        self.history: Dict[str, deque] = {}

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid      = (best_bid + best_ask) / 2.0
            pos      = state.position.get(product, 0)
            limit    = self.position_limit.get(product, 20)

            # Update mid-price history (even for fixed-fair products — harmless)
            if product not in self.history:
                self.history[product] = deque(maxlen=WINDOW)
            self.history[product].append(mid)

            # Determine fair value
            if product in FIXED_FAIR:
                fair = FIXED_FAIR[product]
            else:
                fair = self._regress_fair(self.history[product])

            result[product] = self._trade(
                product, od, pos, limit, fair, best_bid, best_ask
            )

        return result, 0, ""

    # ── Rolling linear regression ─────────────────────────────────────────────
    def _regress_fair(self, prices: deque) -> float:
        """
        Fit y = a*x + b to the last WINDOW mid prices (x = 0, 1, …, n-1)
        and return the fitted value at x = n-1 (the current tick).

        This gives a de-noised estimate of the current fair value that
        follows the local linear trend without reacting to individual noisy
        ticks.  With WINDOW = 30 it is smoothed over ~3 000 timestamps.
        """
        n = len(prices)
        if n < 3:
            # Not enough history yet — fall back to the raw mid
            return prices[-1]

        pts     = list(prices)
        x_mean  = (n - 1) / 2.0
        y_mean  = sum(pts) / n

        num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(pts))
        den = sum((i - x_mean) ** 2         for i in range(n))

        if den == 0:
            return y_mean

        slope     = num / den
        intercept = y_mean - slope * x_mean

        # Evaluate the fitted line at the most recent point
        return intercept + slope * (n - 1)

    # ── Core execution logic (same for every product) ─────────────────────────
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
        p = pos  # running position updated as orders are queued

        # Pass 1 — take mispriced resting orders
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

        # Pass 2 — passive quotes with remaining capacity
        buy_cap  = limit - p
        sell_cap = p + limit

        buy_px  = best_bid + 1
        sell_px = best_ask - 1

        # Collapsed / 1-tick spread fallback
        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
