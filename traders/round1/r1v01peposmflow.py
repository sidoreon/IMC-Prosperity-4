# r1v01peposmflow.py: vNN counts only inside the `peposmflow` family (same round can have other r1v01… files with different tags).
# Pepper (+ osmium); flow / microprice.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json
import math

class PepperRootStrategy:

    SLOPE_WINDOW = 250
    BASE_SPREAD_BID = 4
    BASE_SPREAD_ASK = 4
    INVENTORY_TARGET = 40
    SOFT_LIMIT = 70
    HARD_LIMIT = 80
    AGG_BUY_THRESHOLD = 5
    REGIME_CHANGE_WINDOW = 200
    REGIME_DEVIATION_THRESHOLD = 0.30

    def compute_ols(self, timestamps: List[int], prices: List[float]) -> Tuple[float, float]:

        n = len(timestamps)
        if n < 2:
            return 0.0, prices[-1] if prices else 0.0
        sum_x = sum(timestamps)
        sum_y = sum(prices)
        sum_xx = sum(t * t for t in timestamps)
        sum_xy = sum(t * p for t, p in zip(timestamps, prices))
        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0, sum_y / n
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    def compute_regime(
        self,
        timestamps: List[int],
        prices: List[float],
        long_slope: float,
    ) -> bool:

        w = self.REGIME_CHANGE_WINDOW
        if len(timestamps) < w or long_slope == 0:
            return False
        short_slope, _ = self.compute_ols(timestamps[-w:], prices[-w:])
        deviation = abs(short_slope - long_slope) / abs(long_slope)
        return deviation > self.REGIME_DEVIATION_THRESHOLD

    def run(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        timestamps: List[int],
        prices: List[float],
        current_timestamp: int,
    ) -> List[Order]:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2.0

        timestamps.append(current_timestamp)
        prices.append(mid_price)
        if len(timestamps) > self.SLOPE_WINDOW:
            timestamps.pop(0)
            prices.pop(0)

        slope, intercept = self.compute_ols(timestamps, prices)
        fair_value = intercept + slope * current_timestamp

        regime_flag = self.compute_regime(timestamps, prices, slope)
        if regime_flag:
            print(f"[PEPPER] REGIME CHANGE t={current_timestamp} slope={slope:.8f}")

        size_mult = 0.5 if regime_flag else 1.0
        deviation = mid_price - fair_value
        orders: List[Order] = []

        if deviation < -self.AGG_BUY_THRESHOLD and position < self.SOFT_LIMIT:

            buy_size = int(min(10 * size_mult, self.HARD_LIMIT - position))
            if buy_size > 0:
                orders.append(Order(product, best_ask, buy_size))
        else:

            bid_price = round(fair_value - self.BASE_SPREAD_BID)
            ask_spread = self.BASE_SPREAD_ASK if position < self.SOFT_LIMIT else self.BASE_SPREAD_ASK * 2
            ask_price = round(fair_value + ask_spread)

            bid_size = int(min(10 * size_mult, max(0, self.SOFT_LIMIT - position)))
            ask_size = int(min(10 * size_mult, max(0, position - self.INVENTORY_TARGET)))

            if bid_size > 0:
                orders.append(Order(product, bid_price, bid_size))
            if ask_size > 0 and position > 0:
                orders.append(Order(product, ask_price, -ask_size))

        print(
            f"[PEPPER] t={current_timestamp} fv={fair_value:.2f} mid={mid_price:.2f} "
            f"dev={deviation:.2f} pos={position} slope={slope:.8f} regime={regime_flag}"
        )
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

        print(
            f"[OSMIUM] t={current_timestamp} mean={rolling_mean:.2f} z={z_score:.3f} "
            f"pos={position} bid={bid_price} ask={ask_price} obs_spread={observed_spread}"
        )
        return orders

class RiskEngine:
    def check_position_limit(self, product: str, proposed_position: int, hard_limit: int) -> bool:

        return abs(proposed_position) <= hard_limit

    def compute_realised_pnl(self, trades_history: list) -> float:

        pnl = 0.0
        for trade in trades_history:
            pnl -= trade.price * trade.quantity
        return pnl

    def pnl_drawdown_guard(self, current_pnl: float, peak_pnl: float, threshold: float = 0.15) -> bool:

        if peak_pnl <= 0:
            return False
        return (peak_pnl - current_pnl) / peak_pnl > threshold

    def log_state(
        self,
        timestamp: int,
        positions: Dict[str, int],
        fair_values: Dict[str, float],
        z_scores: Dict[str, float],
    ) -> None:

        print(
            f"[RISK] t={timestamp} pos={positions} fv={fair_values} z={z_scores}"
        )

class Trader:

    _pepper = PepperRootStrategy()
    _osmium = OsmiumStrategy()
    _risk = RiskEngine()

    def run(self, state: TradingState):

        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        pepper_prices: List[float] = data.get("pepper_prices", [])
        pepper_timestamps: List[int] = data.get("pepper_timestamps", [])
        osmium_prices: List[float] = data.get("osmium_prices", [])
        peak_pnl: float = data.get("peak_pnl", 0.0)

        timestamp = state.timestamp
        orders_by_product: Dict[str, List[Order]] = {}

        current_pnl = 0.0
        for trades in state.own_trades.values():
            current_pnl += self._risk.compute_realised_pnl(trades)
        peak_pnl = max(peak_pnl, current_pnl)
        drawdown_guard = self._risk.pnl_drawdown_guard(current_pnl, peak_pnl)

        positions: Dict[str, int] = {}
        fair_values: Dict[str, float] = {}
        z_scores: Dict[str, float] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))
            positions[product] = position

            if product == "INTARIAN_PEPPER_ROOT":
                if drawdown_guard:
                    orders_by_product[product] = []
                    continue
                orders_by_product[product] = self._pepper.run(
                    product, order_depth, position,
                    pepper_timestamps, pepper_prices, timestamp,
                )
                if len(pepper_timestamps) >= 2:
                    slope, intercept = self._pepper.compute_ols(pepper_timestamps, pepper_prices)
                    fair_values[product] = round(intercept + slope * timestamp, 2)

            elif product == "ASH_COATED_OSMIUM":
                if drawdown_guard:
                    orders_by_product[product] = []
                    continue
                orders_by_product[product] = self._osmium.run(
                    product, order_depth, position,
                    osmium_prices, timestamp,
                )
                n = len(osmium_prices)
                if n >= 50:
                    rm = sum(osmium_prices) / n
                    var = sum((p - rm) ** 2 for p in osmium_prices) / n
                    std = math.sqrt(var) if var > 0 else 1.0
                    fair_values[product] = round(rm, 2)
                    if order_depth.buy_orders and order_depth.sell_orders:
                        mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
                        z_scores[product] = round((mid - rm) / std, 3)
                else:
                    fair_values[product] = self._osmium.FAIR_VALUE_ANCHOR

            else:
                orders_by_product[product] = []

        self._risk.log_state(timestamp, positions, fair_values, z_scores)

        trader_data = json.dumps({
            "pepper_prices": pepper_prices[-500:],
            "pepper_timestamps": pepper_timestamps[-500:],
            "osmium_prices": osmium_prices[-200:],
            "peak_pnl": peak_pnl,
        })

        return orders_by_product, 0, trader_data
