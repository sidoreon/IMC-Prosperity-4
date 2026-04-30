# r0v06emerald.py: vNN counts only inside the `emerald` family (round 0; other families reuse v01 independently).
# Post-CSV: passive-only EMERALDS, mean-reversion TOMATOES, top-of-book quoting (built on r0v04emerald).
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    """
    Built on r0v04emerald, updated with CSV data findings:

    EMERALDS:
      - Removed taking pass: CSV confirms ask levels are always >= 10000
        and bid levels always <= 10000. No mispriced orders ever exist.
        Taking pass was dead code. All profit is from passive MM.
      - Passive MM unchanged: quote at bid+1 / ask-1 with full capacity.
      - Position limit raised to 80 (confirmed by competition).

    TOMATOES:
      - Removed momentum tilt: CSV shows lag-1 autocorrelation = -0.42
        (strong mean reversion). Momentum is the wrong signal here —
        negative autocorrelation means anti-momentum. Removing it.
      - EMA fair value kept: mean reversion around EMA is structurally
        correct given the -0.42 autocorrelation.
      - Top-of-book quoting kept (HALF_SPREAD = -1): market spread is
        ~13 ticks, quoting at bid+1/ask-1 puts us well inside it.
      - Inventory skew kept: lean against position, mean reversion
        helps unwind so skew doesn't need to be extreme.
      - TAKE_EDGE kept at 12: only take clear mispricings vs EMA fair.
    """

    EMERALDS_FAIR = 10000

    TOM_HALF_SPREAD  = -1
    TOM_TAKE_EDGE    = 12
    TOM_EMA_ALPHA    = 0.10
    TOM_INV_SKEW_MAX = 5
    # Momentum removed — negative autocorrelation means anti-momentum

    def __init__(self):
        self.position_limit: Dict[str, int] = {
            "TOMATOES": 80,
            "EMERALDS": 80,
        }
        self.ema_mid: Dict[str, float] = {}

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
            limit    = self.position_limit.get(product, 20)

            if product == "EMERALDS":
                result[product] = self._emeralds(order_depth, pos, limit, best_bid, best_ask)
            else:
                result[product] = self._tomatoes(product, order_depth, pos, limit, mid, best_bid, best_ask)

        return result, 0, ""

    # -------------------------------------------------------------------------
    def _emeralds(self, od, pos, limit, best_bid, best_ask) -> List[Order]:
        """
        Pure passive MM. Taking pass removed — book never has mispriced orders.
        All edge comes from quoting inside the market's spread.
        """
        orders  = []
        buy_cap  = limit - pos
        sell_cap = pos + limit

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
        EMA fair value (mean reversion signal, no momentum).
        Take on large mispricings. Passive top-of-book with inv skew.
        """
        # EMA fair value — no momentum tilt (anti-momentum asset)
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

        # Take clear mispricings vs EMA fair
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

        # Inventory skew — lean against position
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
