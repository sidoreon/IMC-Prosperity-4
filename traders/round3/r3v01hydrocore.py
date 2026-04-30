# r3v01hydrocore.py: vNN counts only inside the `hydrocore` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; base fair + edge stack.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional

class Trader:
    LIMITS = {
        "HYDROGEL_PACK": 200,
    }

    HYDROGEL_FAIR = 9990.8
    HYDROGEL_EDGE = 31.94 * 0.9
    HYDROGEL_MAX_TAKE_SIZE = 100

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self.trade_hydrogel(
                state,
                "HYDROGEL_PACK",
                state.order_depths["HYDROGEL_PACK"],
            )

        return result, 0, ""

    def trade_hydrogel(
        self,
        state: TradingState,
        product: str,
        order_depth: OrderDepth,
    ) -> List[Order]:
        return self.trade_extreme_mean_reversion(
            state=state,
            product=product,
            order_depth=order_depth,
            fair_value=self.HYDROGEL_FAIR,
            edge=self.HYDROGEL_EDGE,
            max_take_size=self.HYDROGEL_MAX_TAKE_SIZE,
        )

    def trade_extreme_mean_reversion(
        self,
        state: TradingState,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        edge: float,
        max_take_size: int,
        sell_extra_edge: float = 0.0,
        extra_buy_threshold: Optional[float] = None,
        extra_buy_size: int = 0,
    ) -> List[Order]:
        orders: List[Order] = []

        limit = self.LIMITS[product]
        buy_threshold = fair_value - edge
        sell_threshold = fair_value + edge + sell_extra_edge
        current_position = state.position.get(product, 0)

        net_pos = current_position
        buy_capacity = limit - net_pos
        sell_capacity = limit + net_pos

        def add_buy(price: int, qty: int):
            nonlocal net_pos, buy_capacity, sell_capacity
            qty = int(max(0, min(qty, buy_capacity)))
            if qty > 0:
                orders.append(Order(product, int(price), qty))
                net_pos += qty
                buy_capacity -= qty
                sell_capacity += qty

        def add_sell(price: int, qty: int):
            nonlocal net_pos, buy_capacity, sell_capacity
            qty = int(max(0, min(qty, sell_capacity)))
            if qty > 0:
                orders.append(Order(product, int(price), -qty))
                net_pos -= qty
                sell_capacity -= qty
                buy_capacity += qty

        bought_from_extreme = False
        for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
            if buy_capacity <= 0:
                break
            if ask_price < buy_threshold:
                before = buy_capacity
                add_buy(ask_price, min(-ask_volume, max_take_size))
                if buy_capacity < before:
                    bought_from_extreme = True
            else:
                break

        if (
            extra_buy_threshold is not None
            and extra_buy_size > 0
            and not bought_from_extreme
            and buy_capacity > 0
            and len(order_depth.sell_orders) > 0
        ):
            best_ask, best_ask_volume = min(order_depth.sell_orders.items())
            if best_ask <= extra_buy_threshold:
                add_buy(best_ask, min(-best_ask_volume, extra_buy_size))

        for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if sell_capacity <= 0:
                break
            if bid_price > sell_threshold:
                add_sell(bid_price, min(bid_volume, max_take_size))
            else:
                break

        return orders

    def clip(self, x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x
