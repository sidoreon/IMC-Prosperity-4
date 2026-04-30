# r4v01hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    OTM_VOUCHERS: Dict[str, int] = {
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    OTM_BUY_SIZE: Dict[str, int] = {
        "VEV_5200": 4,
        "VEV_5300": 4,
        "VEV_5400": 4,
        "VEV_5500": 4,
        "VEV_6000": 4,
        "VEV_6500": 4,
    }

    VF_EDGE = 2
    VF_SIZE = 5
    VF_LIMIT = 200

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

            if product == "VELVETFRUIT_EXTRACT":
                pos = state.position.get(product, 0)
                orders = self._trade_velvetfruit(depth, pos, mids)

            elif product in self.OTM_VOUCHERS:
                pos = state.position.get(product, 0)
                orders = self._trade_otm_voucher(product, depth, pos)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int, mids: Dict[str, float]) -> List[Order]:
        # Small delta hedge: buy below mid, sell above mid. Patient limit orders.
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = self.VF_LIMIT - pos
        sell_room = self.VF_LIMIT + pos

        bid_px = best_bid
        ask_px = best_ask

        if buy_room > 0 and bid_px <= mid - self.VF_EDGE + 1:
            qty = min(self.VF_SIZE, buy_room)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(bid_px), int(qty)))

        if sell_room > 0 and ask_px >= mid + self.VF_EDGE - 1:
            qty = min(self.VF_SIZE, sell_room)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(ask_px), -int(qty)))

        return orders

    def _trade_otm_voucher(self, product: str, depth: OrderDepth, pos: int) -> List[Order]:
        # Pure accumulator: only buys OTM options, never sells. Places limit buy at best_bid (patient) or takes at best_ask i...
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_ask is None:
            return orders

        buy_room = limit - pos
        if buy_room <= 0:
            return orders

        base_size = self.OTM_BUY_SIZE[product]

        qty = min(base_size, buy_room, abs(ask_vol))
        if qty > 0:
            orders.append(Order(product, int(best_ask), int(qty)))

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
            return {}
        try:
            return json.loads(trader_data)
        except Exception:
            return {}
