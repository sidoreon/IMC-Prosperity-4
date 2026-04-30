# r1v01peposmrevert.py: vNN counts only inside the `peposmrevert` family (same round can have other r1v01… files with different tags).
# Pepper (+ osmium); z-entry / revert.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json

class PepperRootStrategy:
    HARD_LIMIT = 80
    IMBALANCE_THRESHOLD = 0.3
    FRONT_RUN_TICKS = 1

    def _imbalance(self, order_depth: OrderDepth) -> float:
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(abs(v) for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        return bid_vol / total if total > 0 else 0.5

    def _market_trade_pressure(self, market_trades: list) -> int:
        net = 0
        for trade in market_trades:
            net += trade.quantity
        return net

    def run(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        market_trades: list,
    ) -> List[Order]:
        if not order_depth.sell_orders:
            return []

        imbalance = self._imbalance(order_depth)
        if imbalance < self.IMBALANCE_THRESHOLD:
            return []

        pressure = self._market_trade_pressure(market_trades)
        if pressure < 0 and imbalance < 0.3:
            return []

        best_competing_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        front_run_bid = best_competing_bid + self.FRONT_RUN_TICKS

        orders: List[Order] = []
        room = self.HARD_LIMIT - position
        if room <= 0:
            return []

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if room <= 0:
                break
            available = abs(order_depth.sell_orders[ask_price])
            buy_size = min(available, room)
            orders.append(Order(product, ask_price, buy_size))
            room -= buy_size

        if room > 0 and front_run_bid > 0:
            orders.append(Order(product, front_run_bid, room))

        return orders

class Trader:
    _pepper = PepperRootStrategy()

    def run(self, state: TradingState):
        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == "INTARIAN_PEPPER_ROOT":
                market_trades = state.market_trades.get(product, [])
                orders_by_product[product] = self._pepper.run(product, order_depth, position, market_trades)

            else:
                orders_by_product[product] = []

        return orders_by_product, 0, ""
