# r3v01hvoptema.py: vNN counts only inside the `hvoptema` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; EMA-style fair or slow state.
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

    HG_FAIR = 9991
    HG_TAKE_EDGE = 26
    HG_TAKE_SIZE = 20
    HG_POST_EDGE = 8
    HG_POST_SIZE = 20
    HG_SKEW_DIV = 40

    HG_MOM_ALPHA = 0.12
    HG_MOM_K = 0.45
    HG_MOM_CAP = 6
    HG_MOM_TAKE_BOOST = 1.3
    HG_MOM_KEY = "HYDROGEL_PACK_ema"

    VE_ANCHOR = 5262.0
    VE_ANCHOR_WEIGHT = 0.6
    VE_EMA_ALPHA = 0.08
    VE_IMBALANCE_K = 2.0
    VE_IMBALANCE_CAP = 2.0

    VE_EDGE = 7
    VE_SIZE = 20
    VE_POST_EDGE = 1
    VE_POST_SIZE = 20

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    def compute_ofi(self, depth):
        bid = sum(depth.buy_orders.values())
        ask = sum(abs(v) for v in depth.sell_orders.values())
        return (bid - ask) / (bid + ask) if (bid + ask) else 0

    def run(self, state: TradingState):
        result = {}

        for product, depth in state.order_depths.items():
            orders: List[Order] = []

            if not depth.buy_orders or not depth.sell_orders:
                result[product] = []
                continue

            best_bid = max(depth.buy_orders)
            best_ask = min(depth.sell_orders)

            if product == "VELVETFRUIT_EXTRACT":

                ofi = self.compute_ofi(depth)

                skew = max(-1, min(1, int(2 * ofi)))

                bid = best_bid + max(0, skew)
                ask = best_ask - max(0, -skew)

                orders.append(Order(product, int(bid), 18))
                orders.append(Order(product, int(ask), -18))

            elif product == "HYDROGEL_PACK":

                orders.append(Order(product, int(best_bid), 12))
                orders.append(Order(product, int(best_ask), -12))

            else:

                orders.append(Order(product, int(best_bid), 8))
                orders.append(Order(product, int(best_ask), -8))

            result[product] = orders

        return result, 0, ""

    def _trade_hydrogel(self, depth: OrderDepth, position: int, data: dict) -> List[Order]:

        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None: return orders
        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + position
        mid = (best_bid + best_ask) / 2.0
        old_ema = data.get(self.HG_MOM_KEY, mid)
        ema = (1.0 - self.HG_MOM_ALPHA) * old_ema + self.HG_MOM_ALPHA * mid
        data[self.HG_MOM_KEY] = ema
        raw_momentum = mid - ema
        mom_shift = max(-self.HG_MOM_CAP, min(self.HG_MOM_CAP, self.HG_MOM_K * raw_momentum))
        fair = self.HG_FAIR + mom_shift
        buy_take_edge = max(12.0, self.HG_TAKE_EDGE - max(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)
        sell_take_edge = max(12.0, self.HG_TAKE_EDGE + min(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)
        if best_ask < fair - buy_take_edge and buy_room > 0:
            qty = min(self.HG_TAKE_SIZE, buy_room, abs(ask_vol))
            if qty > 0: orders.append(Order("HYDROGEL_PACK", int(best_ask), int(qty)))
        if best_bid > fair + sell_take_edge and sell_room > 0:
            qty = min(self.HG_TAKE_SIZE, sell_room, bid_vol)
            if qty > 0: orders.append(Order("HYDROGEL_PACK", int(best_bid), -int(qty)))
        skew = position // self.HG_SKEW_DIV
        bid_px = int(fair - self.HG_POST_EDGE - skew)
        ask_px = int(fair + self.HG_POST_EDGE - skew)
        if buy_room > 0 and bid_px < best_ask:
            qty = min(self.HG_POST_SIZE, buy_room)
            if qty > 0: orders.append(Order("HYDROGEL_PACK", bid_px, int(qty)))
        if sell_room > 0 and ask_px > best_bid:
            qty = min(self.HG_POST_SIZE, sell_room)
            if qty > 0: orders.append(Order("HYDROGEL_PACK", ask_px, -int(qty)))
        return orders

    def _best_bid(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.buy_orders: return None, 0
        px = max(depth.buy_orders.keys())
        return px, depth.buy_orders[px]

    def _best_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.sell_orders: return None, 0
        px = min(depth.sell_orders.keys())
        return px, depth.sell_orders[px]

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        bid, _ = self._best_bid(depth)
        ask, _ = self._best_ask(depth)
        if bid is None or ask is None: return None
        return (bid + ask) / 2.0

    def _microprice(self, depth: OrderDepth) -> Optional[float]:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None: return None
        b, a = abs(bid_vol), abs(ask_vol)
        if b + a <= 0: return (bid + ask) / 2.0
        return (bid * a + ask * b) / (a + b)

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data: return {"cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}
        try: return json.loads(trader_data)
        except Exception: return {"cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}

    def _update_stats(self, data: dict, product: str, mid: float) -> None:
        pass

    def _update_and_estimate_pnl(self, data: dict, state: TradingState, mids: Dict[str, float]) -> float:
        return 0.0
