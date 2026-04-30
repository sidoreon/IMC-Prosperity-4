# r3v01hydroema.py: vNN counts only inside the `hydroema` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; EMA fair / smooth mid.
#
import json
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple

class Trader:
    POSITION_LIMITS: Dict[str, int] = {"HYDROGEL_PACK": 80}

    HYDROGEL_FV = 10_000.0
    HYDROGEL_HS = 7
    HYDROGEL_SKEW = 0.20
    HYDROGEL_SIZE = 20
    HYDROGEL_TAKE_SIZE = 4
    MIN_MAKE_SPREAD = 15

    SLOW_ALPHA = 0.015
    FAST_ALPHA = 0.10
    FAIR_STATIC_W = 0.35
    FAIR_SLOW_W = 0.65
    FAIR_TAKE_EDGE = 10
    FAST_TAKE_EDGE = 12
    TREND_BLOCK = 2.0
    MAX_TAKE_POS_FRAC = 0.25
    GAP_TRIGGER = 6
    GAP_BIAS = 2

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        mem = self._load_mem(state.traderData)

        for product, od in state.order_depths.items():
            if product != "HYDROGEL_PACK":
                result[product] = []
                continue
            pos = state.position.get(product, 0)
            orders, mem = self._trade_hydrogel(od, pos, self.POSITION_LIMITS[product], mem)
            result[product] = orders

        return result, 0, json.dumps(mem, separators=(",", ":"))

    def _load_mem(self, trader_data: str) -> Dict[str, float]:
        if not trader_data:
            return {}
        try:
            data = json.loads(trader_data)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _trade_hydrogel(
        self, od: OrderDepth, pos: int, lim: int, mem: Dict[str, float]
    ) -> Tuple[List[Order], Dict[str, float]]:
        orders: List[Order] = []
        if not od.buy_orders or not od.sell_orders:
            return orders, mem

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        slow = float(mem.get("slow_ema", self.HYDROGEL_FV))
        fast = float(mem.get("fast_ema", mid))
        last_mid = mem.get("last_mid")

        slow = self.SLOW_ALPHA * mid + (1.0 - self.SLOW_ALPHA) * slow
        fast = self.FAST_ALPHA * mid + (1.0 - self.FAST_ALPHA) * fast
        fair = self.FAIR_STATIC_W * self.HYDROGEL_FV + self.FAIR_SLOW_W * slow
        trend = fast - slow

        gap_bias = 0
        if last_mid is not None:
            gap = mid - float(last_mid)
            if gap >= self.GAP_TRIGGER:
                gap_bias = -self.GAP_BIAS
            elif gap <= -self.GAP_TRIGGER:
                gap_bias = self.GAP_BIAS

        mem["slow_ema"] = slow
        mem["fast_ema"] = fast
        mem["last_mid"] = mid

        max_take_pos = lim * self.MAX_TAKE_POS_FRAC
        buy_cap = lim - pos
        sell_cap = lim + pos

        if trend > -self.TREND_BLOCK and pos < max_take_pos:
            for ask in sorted(od.sell_orders):
                if ask > fair - self.FAIR_TAKE_EDGE and ask > fast - self.FAST_TAKE_EDGE:
                    break
                qty = min(abs(od.sell_orders[ask]), self.HYDROGEL_TAKE_SIZE, buy_cap)
                if qty <= 0:
                    break
                orders.append(Order("HYDROGEL_PACK", ask, qty))
                pos += qty
                buy_cap -= qty

        if trend < self.TREND_BLOCK and pos > -max_take_pos:
            for bid in sorted(od.buy_orders, reverse=True):
                if bid < fair + self.FAIR_TAKE_EDGE and bid < fast + self.FAST_TAKE_EDGE:
                    break
                qty = min(od.buy_orders[bid], self.HYDROGEL_TAKE_SIZE, sell_cap)
                if qty <= 0:
                    break
                orders.append(Order("HYDROGEL_PACK", bid, -qty))
                pos -= qty
                sell_cap -= qty

        if best_ask - best_bid < self.MIN_MAKE_SPREAD:
            return orders, mem

        adj_fv = fair - self.HYDROGEL_SKEW * pos
        our_bid = round(adj_fv - self.HYDROGEL_HS + max(0, gap_bias))
        our_ask = round(adj_fv + self.HYDROGEL_HS + min(0, gap_bias))

        if trend < -self.TREND_BLOCK:
            our_bid -= 1
            our_ask -= 1
        elif trend > self.TREND_BLOCK:
            our_bid += 1
            our_ask += 1

        our_bid = max(our_bid, best_bid + 1)
        our_ask = min(our_ask, best_ask - 1)
        our_bid = min(our_bid, round(adj_fv))
        our_ask = max(our_ask, round(adj_fv))

        if our_bid >= best_ask:
            our_bid = best_ask - 1
        if our_ask <= best_bid:
            our_ask = best_bid + 1
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        buy_cap = lim - pos
        sell_cap = lim + pos
        size = max(3, int(self.HYDROGEL_SIZE * (1.0 - abs(pos) / lim)))
        if buy_cap > 0:
            orders.append(Order("HYDROGEL_PACK", our_bid, min(size, buy_cap)))
        if sell_cap > 0:
            orders.append(Order("HYDROGEL_PACK", our_ask, -min(size, sell_cap)))

        return orders, mem
