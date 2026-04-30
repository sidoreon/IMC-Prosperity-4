# r4v05hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    VF_SELL_SIZE = 10
    VF_BUY_SIZE = 7
    VF_SELL_EDGE = 1
    VF_BUY_EDGE = 2

    VF_BUY_EVERY_N = 6

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        tick = data.get("tick", 0) + 1
        data["tick"] = tick

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_velvetfruit(depth, pos, tick)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int, tick: int) -> List[Order]:
        # Directional net seller on VELVETFRUIT. Sells large at ask (patient), occasionally buys to partially cover at bid.
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] - pos
        sell_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] + pos

        if sell_room > 0:
            qty = min(self.VF_SELL_SIZE, sell_room, abs(bid_vol))
            if qty > 0:

                sell_px = best_ask
                orders.append(Order("VELVETFRUIT_EXTRACT", int(sell_px), -qty))

        if tick % self.VF_BUY_EVERY_N == 0 and buy_room > 0:
            qty = min(self.VF_BUY_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                buy_px = best_bid
                orders.append(Order("VELVETFRUIT_EXTRACT", int(buy_px), qty))

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

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"tick": 0}
        try:
            d = json.loads(trader_data)
            if "tick" not in d:
                d["tick"] = 0
            return d
        except Exception:
            return {"tick": 0}
