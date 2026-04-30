# r0v01emtomrw.py: vNN counts only inside the `emtomrw` family (round 0; other families reuse v01 independently).
# Random-walk tomato MM (spread capture, no crossing); emerald side still takes misprices vs 10000.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    Pure market-making algorithm built on the random-walk assumption for TOMATOES.

    TOMATOES — random walk:
      Price has no memory. EMA, momentum, mean-reversion signals are all noise.
      The only reliable edge is earning the bid-ask spread.
      Strategy:
        - Always quote at top of book (bid+1 / ask-1).
        - Strong inventory skew: lean hard against open position because
          there is no expected reversion to bail you out.
        - No taking — crossing the spread on a random walk has zero edge
          and costs you the spread you should be earning.

    EMERALDS — mean-reverting to 10000 (not a random walk):
      Hardcoded fair is correct. Take aggressively when mispriced.
      Passive quotes with remaining capacity.
    """

    EMERALDS_FAIR = 10000

    # Tomatoes MM params
    TOM_INV_SKEW_MAX = 4    # max ticks to skew quotes against inventory
                             # higher than usual because no mean reversion to help

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

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            pos      = state.position.get(product, 0)
            limit    = self.position_limit.get(product, 20)

            if product == "EMERALDS":
                result[product] = self._emeralds(od, pos, limit, best_bid, best_ask)
            else:
                result[product] = self._tomatoes_mm(product, od, pos, limit, best_bid, best_ask)

        return result, 0, ""

    # -------------------------------------------------------------------------
    def _emeralds(self, od, pos, limit, best_bid, best_ask) -> List[Order]:
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
    def _tomatoes_mm(self, product, od, pos, limit, best_bid, best_ask) -> List[Order]:
        """
        Pure MM on a random walk.

        No signal. No taking. Just quote both sides at top of book,
        skewed hard against inventory.

        Inventory skew logic:
          On a random walk there is no reversion — if you're long 20,
          the price is equally likely to fall further as to recover.
          So we push quotes down aggressively when long to attract sells
          and reduce exposure as fast as possible.
        """
        orders = []

        # Strong inventory skew — lean hard against open position
        inv_skew = int(-pos / limit * self.TOM_INV_SKEW_MAX)

        buy_px  = best_bid + 1 + inv_skew
        sell_px = best_ask - 1 + inv_skew

        # Safety: don't cross quotes
        if buy_px >= sell_px:
            buy_px  = best_bid + inv_skew
            sell_px = best_ask + inv_skew

        # Clamp to reasonable range around top of book
        buy_px  = min(buy_px,  best_bid + 1)
        sell_px = max(sell_px, best_ask - 1)

        if buy_px >= sell_px:
            buy_px  = best_bid
            sell_px = best_ask

        buy_cap  = limit - pos
        sell_cap = pos + limit

        if buy_cap > 0:
            orders.append(Order(product, buy_px, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, sell_px, -sell_cap))

        return orders
