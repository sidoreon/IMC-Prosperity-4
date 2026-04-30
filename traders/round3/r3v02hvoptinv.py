# r3v02hvoptinv.py: vNN counts only inside the `hvoptinv` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; inventory skew and reduce logic.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict

class Trader:
    # Round 3 Strategy — v2 v1 FAILURE: Bought VELVETFRUIT + 5 VEV products to limit (long 80 each).

    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       80,
        "VELVETFRUIT_EXTRACT": 80,
    }

    HYDROGEL_FV        = 10_000
    HYDROGEL_HS        = 7
    HYDROGEL_SKEW      = 0.10
    HYDROGEL_TAKE_EDGE = 0
    HYDROGEL_SIZE      = 25

    VELVETFRUIT_HS     = 2
    VELVETFRUIT_SKEW   = 0.05
    VELVETFRUIT_SIZE   = 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            lim = self.POSITION_LIMITS.get(product, 0)
            if product == "HYDROGEL_PACK":
                orders = self._trade_hydrogel(od, pos, lim)
            else:
                orders = []
            result[product] = orders
        return result, 0, "ROUND3_v2"

    def _trade_hydrogel(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        orders: List[Order] = []
        FV   = self.HYDROGEL_FV
        HS   = self.HYDROGEL_HS
        SIZE = self.HYDROGEL_SIZE

        best_bid = max(od.buy_orders)  if od.buy_orders  else None
        best_ask = min(od.sell_orders) if od.sell_orders else None

        adj_fv  = FV - self.HYDROGEL_SKEW * pos
        our_bid = round(adj_fv - HS)
        our_ask = round(adj_fv + HS)

        for ask in sorted(od.sell_orders.keys()):
            if ask >= FV - self.HYDROGEL_TAKE_EDGE:
                break
            qty = min(abs(od.sell_orders[ask]), lim - pos)
            if qty <= 0:
                break
            orders.append(Order("HYDROGEL_PACK", ask, qty))
            pos += qty

        for bid in sorted(od.buy_orders.keys(), reverse=True):
            if bid <= FV + self.HYDROGEL_TAKE_EDGE:
                break
            qty = min(od.buy_orders[bid], lim + pos)
            if qty <= 0:
                break
            orders.append(Order("HYDROGEL_PACK", bid, -qty))
            pos -= qty

        if best_bid is not None:
            our_bid = max(our_bid, best_bid + 1)
        if best_ask is not None:
            our_ask = min(our_ask, best_ask - 1)

        our_bid = min(our_bid, round(adj_fv))
        our_ask = max(our_ask, round(adj_fv))
        our_bid = min(our_bid, FV)
        our_ask = max(our_ask, FV)

        if best_ask is not None and our_bid >= best_ask:
            our_bid = best_ask - 1
        if best_bid is not None and our_ask <= best_bid:
            our_ask = best_bid + 1
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap  = lim - pos
        sell_cap = lim + pos
        if buy_cap > 0:
            orders.append(Order("HYDROGEL_PACK", our_bid,  min(SIZE, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("HYDROGEL_PACK", our_ask, -min(SIZE, sell_cap)))

        return orders

    def _trade_velvetfruit(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        # Market-make around current mid (no directional bet). With 5-pt market spread: bid at best_bid+1, ask at best_ask-1.
        orders: List[Order] = []

        return orders