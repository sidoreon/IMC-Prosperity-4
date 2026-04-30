# r4v03hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    OTM_SELL_VOUCHERS: Dict[str, int] = {
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    ITM_VOUCHERS: Dict[str, int] = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
    }

    OTM_SELL_SIZE = 4

    VF_SELL_SIZE = 7
    VF_BUY_SIZE = 6
    VF_SELL_FREQ_FACTOR = 4

    HG_SIZE = 4

    ITM_SIZE = 2

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        tick = data.get("tick", 0) + 1
        data["tick"] = tick

        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is not None:
                mids[product] = mid

        for product, depth in state.order_depths.items():
            orders: List[Order] = []
            pos = state.position.get(product, 0)

            if product == "HYDROGEL_PACK":
                orders = self._trade_hydrogel(depth, pos)

            elif product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_velvetfruit(depth, pos, tick)

            elif product in self.OTM_SELL_VOUCHERS:
                orders = self._trade_otm_sell(product, depth, pos)

            elif product in self.ITM_VOUCHERS:
                orders = self._trade_itm(product, depth, pos)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, pos: int) -> List[Order]:
        # Tiny two-sided on HYDROGEL — very low activity (19 trades total across 3 days).
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - pos
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + pos

        if buy_room > 0:
            qty = min(self.HG_SIZE, buy_room)
            orders.append(Order("HYDROGEL_PACK", int(best_bid), qty))

        if sell_room > 0:
            qty = min(self.HG_SIZE, sell_room)
            orders.append(Order("HYDROGEL_PACK", int(best_ask), -qty))

        return orders

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int, tick: int) -> List[Order]:
        # Net seller on VF. Sells aggressively 4x more than buys. avg sell size 6.9 @ mid+0, avg buy size 5.8 @ mid-0.5
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
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))

        if tick % self.VF_SELL_FREQ_FACTOR == 0 and buy_room > 0:
            qty = min(self.VF_BUY_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))

        return orders

    def _trade_otm_sell(self, product: str, depth: OrderDepth, pos: int) -> List[Order]:
        # Sell-only on OTM VEVs — aggressive ask-side liquidity. Never buys. Sells at ask (or inside ask), avg size 3.5.
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_ask is None:
            return orders

        sell_room = limit + pos
        if sell_room <= 0:
            return orders

        qty = min(self.OTM_SELL_SIZE, sell_room)
        if qty > 0:
            orders.append(Order(product, int(best_ask), -qty))

        return orders

    def _trade_itm(self, product: str, depth: OrderDepth, pos: int) -> List[Order]:
        # Tiny two-sided on deep-ITM options — very low activity.
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        best_bid, _ = self._best_bid(depth)
        best_ask, _ = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = limit - pos
        sell_room = limit + pos

        if buy_room > 0:
            orders.append(Order(product, int(best_bid), min(self.ITM_SIZE, buy_room)))

        if sell_room > 0:
            orders.append(Order(product, int(best_ask), -min(self.ITM_SIZE, sell_room)))

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
            return {"tick": 0}
        try:
            d = json.loads(trader_data)
            if "tick" not in d:
                d["tick"] = 0
            return d
        except Exception:
            return {"tick": 0}
