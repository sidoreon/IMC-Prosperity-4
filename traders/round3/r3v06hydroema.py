# r3v06hydroema.py: vNN counts only inside the `hydroema` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; EMA fair / smooth mid.
#
import json
from datamodel import OrderDepth, Order, TradingState
from typing import List, Dict, Tuple, Optional

class Trader:
    POSITION_LIMITS = {"HYDROGEL_PACK": 200}

    HG_FAIR = 9991
    HG_TAKE_EDGE = 26
    HG_TAKE_SIZE = 20
    HG_POST_EDGE = 8
    HG_POST_SIZE = 20
    HG_SKEW_DIV = 40

    HG_EMA_ALPHA = 0.03
    HG_ANCHOR_WEIGHT = 0.02
    HG_EMA_KEY = "hydrogel_fair"

    HG_MOM_ALPHA = 0.12
    HG_MOM_K = 0.45
    HG_MOM_CAP = 6
    HG_MOM_TAKE_BOOST = 1.3
    HG_MOM_KEY = "HYDROGEL_PACK_ema"

    def run(self, state: TradingState):
        data = json.loads(state.traderData) if state.traderData else {}
        result: Dict[str, List[Order]] = {}

        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hydrogel(
                state.order_depths["HYDROGEL_PACK"], pos, data
            )

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, position: int, data: dict) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + position

        mid = (best_bid + best_ask) / 2.0

        old_ema_fair = data.get(self.HG_EMA_KEY, self.HG_FAIR)
        ema_fair = self.HG_EMA_ALPHA * mid + (1.0 - self.HG_EMA_ALPHA) * old_ema_fair
        ema_fair = (1.0 - self.HG_ANCHOR_WEIGHT) * ema_fair + self.HG_ANCHOR_WEIGHT * self.HG_FAIR
        data[self.HG_EMA_KEY] = ema_fair

        old_ema = data.get(self.HG_MOM_KEY, mid)
        ema = (1.0 - self.HG_MOM_ALPHA) * old_ema + self.HG_MOM_ALPHA * mid
        data[self.HG_MOM_KEY] = ema

        raw_momentum = mid - ema
        mom_shift = max(-self.HG_MOM_CAP, min(self.HG_MOM_CAP, self.HG_MOM_K * raw_momentum))
        fair = ema_fair + mom_shift

        buy_take_edge = max(12.0, self.HG_TAKE_EDGE - max(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)
        sell_take_edge = max(12.0, self.HG_TAKE_EDGE + min(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)

        if best_ask < fair - buy_take_edge and buy_room > 0:
            qty = min(self.HG_TAKE_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_ask), int(qty)))
                position += qty
                buy_room -= qty

        if best_bid > fair + sell_take_edge and sell_room > 0:
            qty = min(self.HG_TAKE_SIZE, sell_room, bid_vol)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_bid), -int(qty)))
                position -= qty
                sell_room -= qty

        skew = position // self.HG_SKEW_DIV
        bid_px = int(fair - self.HG_POST_EDGE - skew)
        ask_px = int(fair + self.HG_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(self.HG_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", bid_px, int(qty)))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(self.HG_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", ask_px, -int(qty)))

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
