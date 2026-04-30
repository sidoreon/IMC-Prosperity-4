# r4v04hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
# Hydrogel / velvet / VEV MM with explicit Mark-flow overlays.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List, Tuple, Optional
import json
import math

class Trader:
    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    HG_SIZE = 4
    HG_TAKE_THRESHOLD = 5

    VEV4000_SIZE = 2
    VEV4000_FAIR_HALF_SPREAD = 9

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is not None:
                mids[product] = mid

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "HYDROGEL_PACK":
                orders = self._trade_hydrogel(depth, pos, data)

            elif product == "VEV_4000":
                orders = self._trade_vev4000(depth, pos)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, pos: int, data: dict) -> List[Order]:
        # Aggressive taker strategy on HYDROGEL. Always buys at ask and sells at bid — crossing the spread.
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        spread = best_ask - best_bid
        if spread < self.HG_TAKE_THRESHOLD:
            return orders

        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - pos
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + pos

        net = data.get("hg_net", 0)

        if net <= 20 and buy_room > 0:
            qty = min(self.HG_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_ask), qty))
                net += qty

        if net >= -20 and sell_room > 0:
            qty = min(self.HG_SIZE, sell_room, abs(bid_vol))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_bid), -qty))
                net -= qty

        data["hg_net"] = net
        return orders

    def _trade_vev4000(self, depth: OrderDepth, pos: int) -> List[Order]:
        # Two-sided passive MM on VEV_4000. Mark 38 buys at ask (avg 1256.8 = above mid) and sells at bid (avg 1237.7 = below...
        orders: List[Order] = []
        limit = self.POSITION_LIMITS["VEV_4000"]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = limit - pos
        sell_room = limit + pos

        if buy_room > 0:
            qty = min(self.VEV4000_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("VEV_4000", int(best_ask), qty))

        if sell_room > 0:
            qty = min(self.VEV4000_SIZE, sell_room, abs(bid_vol))
            if qty > 0:
                orders.append(Order("VEV_4000", int(best_bid), -qty))

        return orders

    def _best_bid(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.buy_orders:
            return None, 0
        px = max(depth.buy_orders.keys())
        return px, depth.buy_orders[px]

    def _best_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.sell_orders:
            return None, 0
        px = min(depth.sell_orders.keys())
        return px, depth.sell_orders[px]

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        bid, _ = self._best_bid(depth)
        ask, _ = self._best_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"hg_net": 0}
        try:
            d = json.loads(trader_data)
            if "hg_net" not in d:
                d["hg_net"] = 0
            return d
        except Exception:
            return {"hg_net": 0}
