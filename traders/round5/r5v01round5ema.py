# r5v01round5ema.py: vNN counts only inside the `round5ema` family (same round can have other r5v01… files with different tags).
# Round 5 EMA-trend or smooth-signal layer on Round-5 universe.
#
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json

class Trader:
    LIMIT = 10
    TAKE_PROFIT = 15000
    STOP_LOSS = 5000
    FLATTEN_AFTER = 995_000
    TURN_FILTERS = {
        "UV": (3, 50.0),
        "OXYGEN": (3, 50.0),
        "PANEL": (3, 50.0),
    }

    BASKETS: Dict[str, Tuple[List[str], Optional[int], int, Optional[int]]] = {
        "UV": (
            [
                "UV_VISOR_YELLOW",
                "UV_VISOR_AMBER",
                "UV_VISOR_ORANGE",
                "UV_VISOR_RED",
                "UV_VISOR_MAGENTA",
            ],
            500,
            300,
            None,
        ),
        "OXYGEN": (
            [
                "OXYGEN_SHAKE_MORNING_BREATH",
                "OXYGEN_SHAKE_EVENING_BREATH",
                "OXYGEN_SHAKE_MINT",
                "OXYGEN_SHAKE_CHOCOLATE",
                "OXYGEN_SHAKE_GARLIC",
            ],
            500,
            400,
            None,
        ),
        "PANEL": (
            ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
            1000,
            600,
            None,
        ),
        "ROBOT": (
            [
                "ROBOT_VACUUMING",
                "ROBOT_MOPPING",
                "ROBOT_DISHES",
                "ROBOT_LAUNDRY",
                "ROBOT_IRONING",
            ],
            500,
            300,
            None,
        ),
        "PEBBLES": (
            ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
            None,
            0,
            50000,
        ),
    }

    def bid(self):
        return 15

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        orders: Dict[str, List[Order]] = {}
        products = self._products()
        projected = {product: state.position.get(product, 0) for product in products}
        pnl_estimate = self._update_cash_and_pnl(data, state, products)

        if state.timestamp >= self.FLATTEN_AFTER or pnl_estimate >= self.TAKE_PROFIT or pnl_estimate <= -self.STOP_LOSS:
            self._flatten(state.order_depths, projected, orders)
            data["halt"] = True
            return orders, 0, json.dumps(data, separators=(",", ":"))

        if data.get("halt"):
            self._flatten(state.order_depths, projected, orders)
            return orders, 0, json.dumps(data, separators=(",", ":"))

        for name, (basket, span, threshold, constant_fair) in self.BASKETS.items():
            quote = self._basket_quote(state.order_depths, basket)
            if quote is None:
                continue

            mid_sum, bid_sum, ask_sum, bid_prices, ask_prices, bid_qty, ask_qty = quote
            fair = float(constant_fair) if constant_fair is not None else data["ema"].get(name, mid_sum)
            can_buy, can_sell = self._turn_filter(data, name, mid_sum)

            buy_edge = fair - ask_sum
            sell_edge = bid_sum - fair

            if can_buy and buy_edge >= threshold and buy_edge >= sell_edge:
                qty = self._basket_buy_capacity(basket, projected, ask_qty)
                if qty > 0:
                    for product in basket:
                        orders.setdefault(product, []).append(Order(product, ask_prices[product], qty))
                        projected[product] = projected.get(product, 0) + qty
            elif can_sell and sell_edge >= threshold:
                qty = self._basket_sell_capacity(basket, projected, bid_qty)
                if qty > 0:
                    for product in basket:
                        orders.setdefault(product, []).append(Order(product, bid_prices[product], -qty))
                        projected[product] = projected.get(product, 0) - qty

            if span is not None:
                alpha = 2.0 / (span + 1.0)
                data["ema"][name] = alpha * mid_sum + (1.0 - alpha) * fair

            self._remember_mid(data, name, mid_sum)

        return orders, 0, json.dumps(data, separators=(",", ":"))

    def _products(self) -> List[str]:
        products: List[str] = []
        for basket, _, _, _ in self.BASKETS.values():
            products.extend(basket)
        return products

    def _load_data(self, encoded: str) -> Dict:
        if not encoded:
            return {"ema": {}, "mid": {}, "cash": {}, "seen": []}
        try:
            data = json.loads(encoded)
            data.setdefault("ema", {})
            data.setdefault("mid", {})
            data.setdefault("cash", {})
            data.setdefault("seen", [])
            return data
        except Exception:
            return {"ema": {}, "mid": {}, "cash": {}, "seen": []}

    def _turn_filter(self, data: Dict, basket_name: str, mid_sum: float) -> Tuple[bool, bool]:
        rule = self.TURN_FILTERS.get(basket_name)
        if rule is None:
            return True, True

        lookback, tolerance = rule
        history = data.setdefault("mid", {}).get(basket_name, [])
        if len(history) < lookback:
            return False, False

        momentum = mid_sum - history[-lookback]
        return momentum >= -tolerance, momentum <= tolerance

    def _remember_mid(self, data: Dict, basket_name: str, mid_sum: float) -> None:
        history = data.setdefault("mid", {}).setdefault(basket_name, [])
        history.append(mid_sum)
        if len(history) > 10:
            del history[:-10]

    def _update_cash_and_pnl(self, data: Dict, state: TradingState, products: List[str]) -> float:
        seen = set(data.get("seen", []))
        cash = data.setdefault("cash", {})

        for product, trades in state.own_trades.items():
            for trade in trades:
                key = f"{product}:{trade.timestamp}:{trade.price}:{trade.quantity}:{trade.buyer}:{trade.seller}"
                if key in seen:
                    continue
                seen.add(key)
                if trade.buyer == "SUBMISSION":
                    cash[product] = cash.get(product, 0.0) - trade.price * trade.quantity
                elif trade.seller == "SUBMISSION":
                    cash[product] = cash.get(product, 0.0) + trade.price * trade.quantity

        data["seen"] = list(seen)[-200:]

        pnl = 0.0
        for product in products:
            depth = state.order_depths.get(product)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                continue
            mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
            pnl += cash.get(product, 0.0) + state.position.get(product, 0) * mid
        return pnl

    def _flatten(self, depths: Dict[str, OrderDepth], projected: Dict[str, int], orders: Dict[str, List[Order]]) -> None:
        for product, position in projected.items():
            depth = depths.get(product)
            if depth is None:
                continue
            if position > 0 and depth.buy_orders:
                best_bid = max(depth.buy_orders)
                qty = min(position, depth.buy_orders[best_bid])
                if qty > 0:
                    orders.setdefault(product, []).append(Order(product, best_bid, -qty))
            elif position < 0 and depth.sell_orders:
                best_ask = min(depth.sell_orders)
                qty = min(-position, -depth.sell_orders[best_ask])
                if qty > 0:
                    orders.setdefault(product, []).append(Order(product, best_ask, qty))

    def _basket_quote(
        self, depths: Dict[str, OrderDepth], basket: List[str]
    ) -> Optional[Tuple[float, int, int, Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]]:
        mid_sum = 0.0
        bid_sum = 0
        ask_sum = 0
        bid_prices: Dict[str, int] = {}
        ask_prices: Dict[str, int] = {}
        bid_qty: Dict[str, int] = {}
        ask_qty: Dict[str, int] = {}

        for product in basket:
            depth = depths.get(product)
            if depth is None or not depth.buy_orders or not depth.sell_orders:
                return None
            bid = max(depth.buy_orders)
            ask = min(depth.sell_orders)
            bid_volume = depth.buy_orders[bid]
            ask_volume = -depth.sell_orders[ask]
            if bid_volume <= 0 or ask_volume <= 0:
                return None

            bid_sum += bid
            ask_sum += ask
            mid_sum += (bid + ask) / 2.0
            bid_prices[product] = bid
            ask_prices[product] = ask
            bid_qty[product] = bid_volume
            ask_qty[product] = ask_volume

        return mid_sum, bid_sum, ask_sum, bid_prices, ask_prices, bid_qty, ask_qty

    def _basket_buy_capacity(self, basket: List[str], projected: Dict[str, int], ask_qty: Dict[str, int]) -> int:
        qty = self.LIMIT
        for product in basket:
            qty = min(qty, self.LIMIT - projected.get(product, 0), ask_qty[product])
        return max(0, qty)

    def _basket_sell_capacity(self, basket: List[str], projected: Dict[str, int], bid_qty: Dict[str, int]) -> int:
        qty = self.LIMIT
        for product in basket:
            qty = min(qty, self.LIMIT + projected.get(product, 0), bid_qty[product])
        return max(0, qty)