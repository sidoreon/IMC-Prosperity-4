# r4v07hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    VF_BUY_SIZE = 9
    VF_BUY_EDGE = 1
    VF_LIMIT = 200

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_velvetfruit(depth, pos)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int) -> List[Order]:
        # Pure accumulator: only buys VELVETFRUIT at or slightly above mid. Lifts the ask (taker). avg slippage +0.80 vs mid ...
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_ask is None:
            return orders

        buy_room = self.VF_LIMIT - pos
        if buy_room <= 0:
            return orders

        mid = (best_bid + best_ask) / 2.0 if best_bid is not None else best_ask

        qty = min(self.VF_BUY_SIZE, buy_room, abs(ask_vol))
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))

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
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}
