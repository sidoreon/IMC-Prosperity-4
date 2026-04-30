# r0v01emtomimitate.py: vNN counts only inside the `emtomimitate` family (round 0; other families reuse v01 independently).
# Shapes behaviour to observed backtests: aggressive emerald takes + TOB tomato MM, sparse takes vs EMA.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    Imitates the observed backtest behaviour:

    EMERALDS — staircase PNL, position held at extremes:
      Take everything mispriced vs hardcoded fair=10000.
      Post passive quotes with ALL remaining capacity.
      No inventory skew — let position sit at limit until market reverts.

    TOMATOES — continuous noisy upward drift, position oscillating frequently:
      Always quote at top of book (bid+1 / ask-1) with full capacity.
      Only take on large clear mispricings vs EMA fair (high TAKE_EDGE).
      Earn spread continuously; don't try to hold directional bets.
    """

    EMERALDS_FAIR  = 10000

    TOM_EMA_ALPHA  = 0.10
    TOM_TAKE_EDGE  = 15    # only take very clear mispricings
    # No HALF_SPREAD — always quote top-of-book

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "EMERALDS": 20,
            "TOMATOES": 20,
        }
        self.ema: Dict[str, float] = {}

    # -------------------------------------------------------------------------
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

            if product == "EMERALDS":
                result[product] = self._emeralds(od, pos, limit, best_bid, best_ask)
            else:
                result[product] = self._tomatoes(product, od, pos, limit, mid, best_bid, best_ask)

        return result, 0, ""

    # -------------------------------------------------------------------------
    def _emeralds(self, od, pos, limit, best_bid, best_ask) -> List[Order]:
        """
        Take aggressively vs fair=10000, hold position at extremes.
        Staircase PNL comes from buying low/selling high in discrete chunks.
        """
        fair   = self.EMERALDS_FAIR
        orders = []
        p      = pos

        # Take every ask below fair
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

        # Take every bid above fair
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

        # Passive with full remaining capacity — no skew, let position sit
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
        """
        Always quote at top of book for maximum fill rate (continuous MM income).
        Only take on very large mispricings vs EMA fair.
        Position oscillates frequently at smaller sizes — no directional holding.
        """
        # Update EMA
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = self.TOM_EMA_ALPHA * mid + (1 - self.TOM_EMA_ALPHA) * self.ema[product]
        fair = self.ema[product]

        orders = []
        p      = pos

        # Take only on very clear mispricings
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

        # Always quote at top of book — maximise fill rate for MM income
        buy_px  = best_bid + 1
        sell_px = best_ask - 1

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
