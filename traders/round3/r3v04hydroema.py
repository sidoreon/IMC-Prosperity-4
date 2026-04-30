# r3v04hydroema.py: vNN counts only inside the `hydroema` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; EMA fair / smooth mid.
#
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
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

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        trader_data = self.load_trader_data(state.traderData)

        od = state.order_depths.get("HYDROGEL_PACK")
        if od is not None:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self.trade_hydrogel(od, pos, trader_data)
        elif "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = []

        return result, 0, json.dumps(trader_data, separators=(",", ":"))

    def trade_hydrogel(self, od: OrderDepth, position: int, trader_data: Dict) -> List[Order]:
        orders: List[Order] = []
        best_bid = self.best_bid(od)
        best_ask = self.best_ask(od)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = POSITION_LIMITS["HYDROGEL_PACK"] + position

        mid = (best_bid + best_ask) / 2.0
        old_ema = trader_data.get(HG_MOM_KEY, mid)
        ema = (1.0 - HG_MOM_ALPHA) * old_ema + HG_MOM_ALPHA * mid
        trader_data[HG_MOM_KEY] = ema

        raw_momentum = mid - ema
        mom_shift = max(-HG_MOM_CAP, min(HG_MOM_CAP, HG_MOM_K * raw_momentum))
        fair = HG_FAIR + mom_shift

        buy_take_edge = HG_TAKE_EDGE - max(0.0, mom_shift) * HG_MOM_TAKE_BOOST
        sell_take_edge = HG_TAKE_EDGE + min(0.0, mom_shift) * HG_MOM_TAKE_BOOST

        buy_take_edge = max(12.0, buy_take_edge)
        sell_take_edge = max(12.0, sell_take_edge)

        if best_ask < fair - buy_take_edge and buy_room > 0:
            qty = min(HG_TAKE_SIZE, buy_room, abs(od.sell_orders[best_ask]))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + sell_take_edge and sell_room > 0:
            qty = min(HG_TAKE_SIZE, sell_room, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -qty))
                position -= qty
                sell_room -= qty

        skew = position // HG_SKEW_DIV
        bid_px = int(fair - HG_POST_EDGE - skew)
        ask_px = int(fair + HG_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(HG_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(HG_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", ask_px, -qty))

        return orders

    def load_trader_data(self, raw: str) -> Dict:
        if not raw:
            return {}
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def best_bid(od: Optional[OrderDepth]) -> Optional[int]:
        if od is None or not od.buy_orders:
            return None
        return max(od.buy_orders)

    @staticmethod
    def best_ask(od: Optional[OrderDepth]) -> Optional[int]:
        if od is None or not od.sell_orders:
            return None
        return min(od.sell_orders)