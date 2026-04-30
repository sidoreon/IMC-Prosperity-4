# r4v06hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    VF_SIZE = 6
    VF_MOM_ALPHA = 0.25
    VF_MOM_THRESHOLD = 0.5
    VF_MAX_POS = 80
    VF_REVERSAL_SIZE = 6

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_velvetfruit(depth, pos, data)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int, data: dict) -> List[Order]:
        # High-frequency momentum scalper. Always crosses the spread. Buys at ask when momentum positive, sells at bid when n...
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] - pos
        sell_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] + pos

        ema_fast = data.get("vf_ema_fast", mid)
        ema_fast = self.VF_MOM_ALPHA * mid + (1 - self.VF_MOM_ALPHA) * ema_fast
        data["vf_ema_fast"] = ema_fast

        ema_slow = data.get("vf_ema_slow", mid)
        ema_slow = 0.05 * mid + 0.95 * ema_slow
        data["vf_ema_slow"] = ema_slow

        momentum = ema_fast - ema_slow

        if pos >= self.VF_MAX_POS and sell_room > 0:
            qty = min(self.VF_REVERSAL_SIZE, sell_room, abs(bid_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))
            return orders

        if pos <= -self.VF_MAX_POS and buy_room > 0:
            qty = min(self.VF_REVERSAL_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))
            return orders

        if momentum > self.VF_MOM_THRESHOLD and buy_room > 0:
            qty = min(self.VF_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))

        elif momentum < -self.VF_MOM_THRESHOLD and sell_room > 0:
            qty = min(self.VF_SIZE, sell_room, abs(bid_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))

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
