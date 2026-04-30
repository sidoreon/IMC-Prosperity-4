# r1v01peposmema.py: vNN counts only inside the `peposmema` family (same round can have other r1v01… files with different tags).
# Pepper (+ osmium); EMA stack.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json
import math

class PepperRootStrategy:
    HARD_LIMIT = 80
    IMBALANCE_THRESHOLD = 0.5
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

        if pressure < 0 and imbalance < 0.65:
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

class OsmiumStrategy:
    FAIR_VALUE_ANCHOR = 10000
    ROLLING_WINDOW = 200
    Z_ENTRY_THRESHOLD = 1.5
    Z_STRONG_THRESHOLD = 3.0
    BASE_SPREAD = 3
    SKEW_FACTOR = 0.3
    SOFT_LIMIT = 60
    HARD_LIMIT = 80

    def run(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        prices: List[float],
        current_timestamp: int,
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2.0
        observed_spread = best_ask - best_bid

        prices.append(mid_price)
        if len(prices) > self.ROLLING_WINDOW:
            prices.pop(0)

        n = len(prices)
        if n < 50:
            rolling_mean = float(self.FAIR_VALUE_ANCHOR)
            z_score = 0.0
        else:
            rolling_mean = sum(prices) / n
            variance = sum((p - rolling_mean) ** 2 for p in prices) / n
            rolling_std = math.sqrt(variance) if variance > 0 else 1.0
            z_score = (mid_price - rolling_mean) / rolling_std

        inv_skew = position * self.SKEW_FACTOR
        bid_price = round(rolling_mean - self.BASE_SPREAD - inv_skew)
        ask_price = round(rolling_mean + self.BASE_SPREAD - inv_skew)

        if observed_spread < 10:
            bid_price -= 4
            ask_price += 4

        abs_z = abs(z_score)
        if abs_z < self.Z_ENTRY_THRESHOLD:
            base_size = 5
        elif abs_z < self.Z_STRONG_THRESHOLD:
            base_size = 10
        else:
            base_size = 20

        bid_size = base_size
        ask_size = base_size

        if position > self.SOFT_LIMIT:
            ask_size = base_size * 2
            bid_size = max(base_size // 2, 1)
        elif position < -self.SOFT_LIMIT:
            bid_size = base_size * 2
            ask_size = max(base_size // 2, 1)

        bid_size = min(bid_size, max(0, self.HARD_LIMIT - position))
        ask_size = min(ask_size, max(0, self.HARD_LIMIT + position))

        orders: List[Order] = []
        if bid_size > 0:
            orders.append(Order(product, bid_price, bid_size))
        if ask_size > 0:
            orders.append(Order(product, ask_price, -ask_size))
        return orders

class Trader:
    _pepper = PepperRootStrategy()
    _osmium = OsmiumStrategy()

    def run(self, state: TradingState):
        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        osmium_prices: List[float] = data.get("osmium_prices", [])

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == "INTARIAN_PEPPER_ROOT":
                market_trades = state.market_trades.get(product, [])
                orders_by_product[product] = self._pepper.run(product, order_depth, position, market_trades)

            elif product == "ASH_COATED_OSMIUM":
                orders_by_product[product] = self._osmium.run(
                    product, order_depth, position, osmium_prices, state.timestamp
                )

            else:
                orders_by_product[product] = []

        trader_data = json.dumps({"osmium_prices": osmium_prices[-200:]})

        return orders_by_product, 0, trader_data
