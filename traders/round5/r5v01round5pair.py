# r5v01round5pair.py: vNN counts only inside the `round5pair` family (same round can have other r5v01… files with different tags).
# Round 5 paired-asset logic (cointegration / residual style) on listed underlyings.
#
import json
import numpy as np
import math
from datamodel import TradingState, Order
from typing import Dict, List

class Trader:

    def trade_pairs(self, state: TradingState, product_a: str, product_b: str) -> List[Order]:
        orders = []

        depth_a = state.order_depths.get(product_a)
        depth_b = state.order_depths.get(product_b)
        if depth_a is None or depth_b is None:
            return orders

        if not depth_a.buy_orders or not depth_a.sell_orders:
            return orders
        if not depth_b.buy_orders or not depth_b.sell_orders:
            return orders

        best_bid_a = max(depth_a.buy_orders.keys())
        best_ask_a = min(depth_a.sell_orders.keys())
        mid_a = (best_bid_a + best_ask_a) / 2

        best_bid_b = max(depth_b.buy_orders.keys())
        best_ask_b = min(depth_b.sell_orders.keys())
        mid_b = (best_bid_b + best_ask_b) / 2

        ratio = mid_a / mid_b if mid_b != 0 else 1

        self.ratio_history.append(ratio)

        window = 50
        if len(self.ratio_history) > window:
            self.ratio_history = self.ratio_history[-window:]
        if len(self.ratio_history) < window:
            return orders

        hist = self.ratio_history[-window:]
        mean_ratio = sum(hist) / window
        var = sum((x - mean_ratio) ** 2 for x in hist) / window
        std_ratio = var ** 0.5

        if std_ratio == 0:
            return orders

        z = (ratio - mean_ratio) / std_ratio

        pos_a = state.position.get(product_a, 0)
        pos_b = state.position.get(product_b, 0)
        max_position = 10

        entry_z = 1.5
        exit_z = 0.5
        size = 3

        if z > entry_z:
            sell_a = min(size, max_position + pos_a)
            buy_b = min(size, max_position - pos_b)

            if sell_a > 0:
                orders.append(Order(product_a, best_bid_a, -sell_a))
            if buy_b > 0:
                orders.append(Order(product_b, best_ask_b, buy_b))

        elif z < -entry_z:
            buy_a = min(size, max_position - pos_a)
            sell_b = min(size, max_position + pos_b)

            if buy_a > 0:
                orders.append(Order(product_a, best_ask_a, buy_a))
            if sell_b > 0:
                orders.append(Order(product_b, best_bid_b, -sell_b))

        elif abs(z) < exit_z:

            if pos_a > 0:
                orders.append(Order(product_a, best_bid_a, -pos_a))
            elif pos_a < 0:
                orders.append(Order(product_a, best_ask_a, -pos_a))

            if pos_b > 0:
                orders.append(Order(product_b, best_bid_b, -pos_b))
            elif pos_b < 0:
                orders.append(Order(product_b, best_ask_b, -pos_b))

        return orders

    def trade_pebbles_basket(self, state: TradingState) -> List[Order]:
        orders = []

        pebbles = ["PEBBLES_M", "PEBBLES_S", "PEBBLES_L", "PEBBLES_XS", "PEBBLES_XL"]

        depths = {}
        mids = {}

        for p in pebbles:
            depth = state.order_depths.get(p)
            if depth is None:
                return orders
            if not depth.buy_orders or not depth.sell_orders:
                return orders

            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid = (best_bid + best_ask) / 2

            depths[p] = depth
            mids[p] = mid

        total_price = sum(mids.values())
        target_total = 50000

        basket_edge = total_price - target_total

        max_position = 10
        positions = {p: state.position.get(p, 0) for p in pebbles}

        threshold = 5
        if basket_edge > threshold:
            for p in pebbles:
                depth = depths[p]
                best_bid = max(depth.buy_orders.keys())

                pos = positions[p]
                sell_qty = min(2, max_position + pos)

                if sell_qty > 0:
                    orders.append(Order(p, best_bid, -sell_qty))
                    positions[p] -= sell_qty

        elif basket_edge < -threshold:
            for p in pebbles:
                depth = depths[p]
                best_ask = min(depth.sell_orders.keys())

                pos = positions[p]
                buy_qty = min(2, max_position - pos)

                if buy_qty > 0:
                    orders.append(Order(p, best_ask, buy_qty))
                    positions[p] += buy_qty

        adjustment = (target_total - total_price) / len(pebbles)
        for p in pebbles:
            depth = depths[p]
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            fair_price = mids[p] + adjustment
            pos = positions[p]
            edge_buy = fair_price - best_ask
            edge_sell = best_bid - fair_price
            local_threshold = 2

            if edge_buy > local_threshold and pos < max_position:
                qty = min(3, max_position - pos)
                if qty > 0:
                    orders.append(Order(p, best_ask, qty))
                    positions[p] += qty

            if edge_sell > local_threshold and pos > -max_position:
                qty = min(3, max_position + pos)
                if qty > 0:
                    orders.append(Order(p, best_bid, -qty))
                    positions[p] -= qty

        return orders

    def trade_mean_reverting(self, state: TradingState, product: str) -> List[Order]:
        orders = []
        product_position = state.position.get(product, 0)
        order_depth = state.order_depths[product]
        max_position = 10
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = -order_depth.sell_orders[best_ask]
        imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
        imbalance = max(-0.5, min(0.5, imbalance))

        spread = max(2, best_ask - best_bid)
        mid = (best_bid + best_ask) / 2

        k = spread * 0.15
        fair_price = mid + k * imbalance

        arb_limit = 2
        arb_threshold = spread * 0.45

        for price, quantity in sorted(order_depth.sell_orders.items()):
            edge = fair_price - price
            if edge > arb_threshold and product_position < arb_limit:
                buy_quantity = min(max_position - product_position, -quantity)
                orders.append(Order(product, price, buy_quantity))
                product_position += buy_quantity

        for price, quantity in sorted(order_depth.buy_orders.items(), reverse=True):
            edge = price - fair_price
            if edge > arb_threshold and product_position > -arb_limit:
                sell_quantity = min(max_position + product_position, quantity)
                orders.append(Order(product, price, -sell_quantity))
                product_position -= sell_quantity

        if order_depth.buy_orders and order_depth.sell_orders:
            inv = product_position / max_position
            kappa = 0.5
            base_size = 3
            bid_size = max(0, int(base_size * (1 - inv)))
            ask_size = max(0, int(base_size * (1 + inv)))

            spread *= 0.9
            reservation_price = fair_price - kappa * inv

            if bid_size > 0:
                bid_price = round(reservation_price - spread / 2)
                orders.append(Order(product, bid_price, bid_size))

            if ask_size > 0:
                ask_price = round(reservation_price + spread / 2)
                orders.append(Order(product, ask_price, -ask_size))

        return orders

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        self.ratio_history = saved.get("ratio_history", [])

        result = {}

        for product in state.order_depths:
            result[product] = []

        mean_reverting = [
            "UV_VISOR_ORANGE",
            "ROBOT_VACUUMING",
            "ROBOT_DISHES",
            "PANEL_2X2",
            "OXYGEN_SHAKE_MINT",
            "MICROCHIP_TRIANGLE"
        ]

        for product in mean_reverting:
            if product in state.order_depths:
                orders = self.trade_mean_reverting(state, product)
                for o in orders:
                    result[o.symbol].append(o)

        orders = self.trade_pairs(state, "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY")
        for o in orders:
            if o.symbol not in result:
                result[o.symbol] = []
            result[o.symbol].append(o)

        orders = self.trade_pebbles_basket(state)
        for o in orders:
            if o.symbol not in result:
                result[o.symbol] = []
            result[o.symbol].append(o)

        traderData = json.dumps({"ratio_history": self.ratio_history}, separators=(",", ":"))
        conversions = 0
        return result, conversions, traderData