# r4v02hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
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

    HG_FAIR = 9994.0
    HG_POST_EDGE = 8
    HG_POST_SIZE = 4
    HG_TAKE_EDGE = 20

    VF_FAIR = 5247.0
    VF_POST_EDGE = 3
    VF_POST_SIZE = 6
    VF_TAKE_EDGE = 8

    VEV4000_SPREAD = 19
    VEV4000_SIZE = 2

    OTM_VOUCHERS = {"VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}
    OTM_SIZE = 4

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

            elif product == "VELVETFRUIT_EXTRACT":
                orders = self._trade_velvetfruit(depth, pos, data)

            elif product == "VEV_4000":
                orders = self._trade_vev4000(depth, pos)

            elif product in self.OTM_VOUCHERS:
                orders = self._trade_otm_buy(product, depth, pos)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, pos: int, data: dict) -> List[Order]:
        # Passive MM: post bid/ask around fair. Earns spread of ~16 ticks.
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0

        ema = data.get("hg_ema", mid)
        ema = 0.05 * mid + 0.95 * ema
        data["hg_ema"] = ema
        fair = ema

        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - pos
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + pos

        skew = pos // 40

        if best_ask < fair - self.HG_TAKE_EDGE and buy_room > 0:
            qty = min(self.HG_POST_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_ask), qty))
                pos += qty; buy_room -= qty

        if best_bid > fair + self.HG_TAKE_EDGE and sell_room > 0:
            qty = min(self.HG_POST_SIZE, sell_room, bid_vol)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_bid), -qty))
                pos -= qty; sell_room -= qty

        bid_px = int(fair - self.HG_POST_EDGE - skew)
        ask_px = int(fair + self.HG_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(self.HG_POST_SIZE, buy_room)
            orders.append(Order("HYDROGEL_PACK", bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(self.HG_POST_SIZE, sell_room)
            orders.append(Order("HYDROGEL_PACK", ask_px, -qty))

        return orders

    def _trade_velvetfruit(self, depth: OrderDepth, pos: int, data: dict) -> List[Order]:
        # Balanced MM on VF: post tight around fair, avg size 5.5.
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        ema = data.get("vf_ema", mid)
        ema = 0.08 * mid + 0.92 * ema
        data["vf_ema"] = ema
        fair = ema

        buy_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] - pos
        sell_room = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"] + pos
        skew = pos // 34

        if best_ask < fair - self.VF_TAKE_EDGE and buy_room > 0:
            qty = min(self.VF_POST_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_ask), qty))
                pos += qty; buy_room -= qty

        if best_bid > fair + self.VF_TAKE_EDGE and sell_room > 0:
            qty = min(self.VF_POST_SIZE, sell_room, bid_vol)
            if qty > 0:
                orders.append(Order("VELVETFRUIT_EXTRACT", int(best_bid), -qty))
                pos -= qty; sell_room -= qty

        bid_px = int(fair - self.VF_POST_EDGE - skew)
        ask_px = int(fair + self.VF_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, min(self.VF_POST_SIZE, buy_room)))

        if sell_room > 0 and ask_px > best_bid:
            orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, -min(self.VF_POST_SIZE, sell_room)))

        return orders

    def _trade_vev4000(self, depth: OrderDepth, pos: int) -> List[Order]:
        # Two-sided MM on VEV_4000: buy ~1238 (bid side), sell ~1257 (ask side). The ~19 tick spread mirrors the observed buy...
        orders: List[Order] = []
        limit = self.POSITION_LIMITS["VEV_4000"]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        mid = (best_bid + best_ask) / 2.0
        buy_room = limit - pos
        sell_room = limit + pos

        bid_px = best_bid
        ask_px = best_ask

        if buy_room > 0:
            qty = min(self.VEV4000_SIZE, buy_room, abs(bid_vol) if bid_vol else self.VEV4000_SIZE)
            if qty > 0:
                orders.append(Order("VEV_4000", int(bid_px), qty))

        if sell_room > 0:
            qty = min(self.VEV4000_SIZE, sell_room, abs(ask_vol) if ask_vol else self.VEV4000_SIZE)
            if qty > 0:
                orders.append(Order("VEV_4000", int(ask_px), -qty))

        return orders

    def _trade_otm_buy(self, product: str, depth: OrderDepth, pos: int) -> List[Order]:
        # Buy-only on OTM strikes (5200-5500). Small size, patient.
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        _, ask_vol = self._best_ask(depth)
        best_ask, _ = self._best_ask(depth)
        if best_ask is None:
            return orders

        buy_room = limit - pos
        if buy_room <= 0:
            return orders

        qty = min(self.OTM_SIZE, buy_room, abs(ask_vol))
        if qty > 0:
            orders.append(Order(product, int(best_ask), qty))

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
