# r1v01osmiumflow.py: vNN counts only inside the `osmiumflow` family (same round can have other r1v01… files with different tags).
# Osmium book; flow / microprice.
#
from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Dict, List
import json
import math

class OsmiumStrategy:
    PRODUCT = "ASH_COATED_OSMIUM"
    POSITION_LIMIT = 80

    HISTORY_WINDOW = 200
    MIN_HISTORY = 20
    EMA_ALPHA = 0.1

    TAKE_EDGE = 1
    CLEAR_WIDTH = 1

    BASE_PASSIVE_SIZE = 15
    INVENTORY_SKEW = 0.02
    SOFT_POSITION_LIMIT = 20
    SIZE_ADJUST_RATE = 8

    AR_WEIGHT = 0.7
    AR_MIN_HISTORY = 30

    MOMENTUM_WINDOW = 6
    MOMENTUM_WEIGHT = 0.2

    SPREAD_VOL_TIGHT = 0.3
    SPREAD_VOL_WIDE  = 0.6
    MOMENTUM_LEAN    = 0.1

    @staticmethod
    def _ema(values: List[float], alpha: float) -> float:
        ema = values[0]
        for x in values[1:]:
            ema = alpha * x + (1 - alpha) * ema
        return ema

    @staticmethod
    def _stdev(values: List[float]) -> float:
        if len(values) < 2:
            return 1.0
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        return max(math.sqrt(var), 1.0)

    def _microprice(self, order_depth: OrderDepth) -> float:
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        bid_vol = order_depth.buy_orders[best_bid]
        ask_vol = abs(order_depth.sell_orders[best_ask])
        total = bid_vol + ask_vol
        if total == 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    def _ar1_beta(self, prices: List[float]) -> float:
        n = len(prices)
        if n < self.AR_MIN_HISTORY:
            return 0.0
        mean = sum(prices) / n
        devs = [prices[i] - mean for i in range(n - 1)]
        rets = [prices[i + 1] - prices[i] for i in range(n - 1)]
        ss_dev = sum(d * d for d in devs)
        if ss_dev == 0:
            return 0.0
        beta = sum(d * r for d, r in zip(devs, rets)) / ss_dev
        return max(-1.0, min(0.0, beta))

    def _lagrange_momentum(self, prices: List[float]) -> float:
        # Estimate price trend velocity using the slope of a first-order Lagrange polynomial (linear OLS) fit through the las...
        n = self.MOMENTUM_WINDOW
        if len(prices) < n:
            return 0.0
        pts = prices[-n:]
        t_bar = (n - 1) / 2.0
        p_bar = sum(pts) / n
        num = sum((i - t_bar) * (pts[i] - p_bar) for i in range(n))
        den = sum((i - t_bar) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    def _recent_flow(self, market_trades: list, best_bid: float, best_ask: float) -> float:
        if not market_trades:
            return 0.0
        net = 0.0
        mid = (best_bid + best_ask) / 2
        for trade in market_trades:
            if trade.buyer == "SUBMISSION" or trade.seller == "SUBMISSION":
                continue
            if trade.price >= mid:
                net += trade.quantity
            else:
                net -= trade.quantity
        return net

    def run(
        self,
        order_depth: OrderDepth,
        position: int,
        history: List[float],
        timestamp: int,
        market_trades: List = [],
    ):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0.0, 0.0

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        spread = best_ask - best_bid

        microprice = self._microprice(order_depth)

        if len(history) >= self.MIN_HISTORY:
            ema = self._ema(history[-self.HISTORY_WINDOW:], self.EMA_ALPHA)
            fair = 0.70 * microprice + 0.30 * ema
        else:
            fair = microprice

        flow = self._recent_flow(market_trades, best_bid, best_ask)
        flow_adj = 0.1 * (flow / max(abs(flow), 1))
        fair += flow_adj

        vol = self._stdev(history[-self.HISTORY_WINDOW:]) if len(history) >= 2 else 1.0
        window = history[-self.HISTORY_WINDOW:] if history else [microprice]
        rolling_mean = sum(window) / len(window)
        z = (microprice - rolling_mean) / max(vol, 1.0)

        if len(history) >= self.AR_MIN_HISTORY:
            ar_beta = self._ar1_beta(history[-self.HISTORY_WINDOW:])
            dev_from_mean = microprice - rolling_mean
            ar_adj = ar_beta * dev_from_mean * self.AR_WEIGHT
            fair += ar_adj
        else:
            ar_beta = 0.0
            ar_adj = 0.0

        momentum = self._lagrange_momentum(history) if len(history) >= self.MOMENTUM_WINDOW else 0.0

        momentum_norm = momentum / max(vol, 1.0)

        mom_adj = -momentum_norm * self.MOMENTUM_WEIGHT
        fair += mom_adj

        buy_order_volume = 0
        sell_order_volume = 0

        buy_cap  = lambda: max(0, self.POSITION_LIMIT - position - buy_order_volume)
        sell_cap = lambda: max(0, self.POSITION_LIMIT + position - sell_order_volume)

        orders: List[Order] = []

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if buy_cap() <= 0:
                break
            edge = fair - ask_price
            if edge < self.TAKE_EDGE:
                break
            available = abs(order_depth.sell_orders[ask_price])
            size = 8 if edge < 3 else (15 if edge < 5 else available)
            qty = min(available, size, buy_cap())
            if qty > 0:
                orders.append(Order(self.PRODUCT, ask_price, qty))
                buy_order_volume += qty

        for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
            if sell_cap() <= 0:
                break
            edge = bid_price - fair
            if edge < self.TAKE_EDGE:
                break
            available = order_depth.buy_orders[bid_price]
            size = 8 if edge < 3 else (15 if edge < 5 else available)
            qty = min(available, size, sell_cap())
            if qty > 0:
                orders.append(Order(self.PRODUCT, bid_price, -qty))
                sell_order_volume += qty

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_px = int(round(fair))

        if position_after_take > 0:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if sell_cap() <= 0:
                    break
                if bid_price < fair_px - self.CLEAR_WIDTH:
                    break
                available = order_depth.buy_orders[bid_price]
                qty = min(position_after_take, available, sell_cap())
                if qty > 0:
                    orders.append(Order(self.PRODUCT, bid_price, -qty))
                    sell_order_volume += qty
                    position_after_take -= qty

        elif position_after_take < 0:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if buy_cap() <= 0:
                    break
                if ask_price > fair_px + self.CLEAR_WIDTH:
                    break
                available = abs(order_depth.sell_orders[ask_price])
                qty = min(abs(position_after_take), available, buy_cap())
                if qty > 0:
                    orders.append(Order(self.PRODUCT, ask_price, qty))
                    buy_order_volume += qty
                    position_after_take += qty

        passive_tight = max(1, int(vol * self.SPREAD_VOL_TIGHT))
        passive_wide  = max(2, int(vol * self.SPREAD_VOL_WIDE))

        skew = int(position * self.INVENTORY_SKEW)

        mom_lean = int(momentum_norm * self.MOMENTUM_LEAN)

        total_shift = skew + mom_lean

        inv_adj = abs(position) // self.SIZE_ADJUST_RATE
        bid_size_adj = (-inv_adj if position > 0 else +inv_adj) if position != 0 else 0
        ask_size_adj = (+inv_adj if position > 0 else -inv_adj) if position != 0 else 0

        def make_size(base: int, adj: int) -> int:
            return max(2, base + adj)

        tight_bid = best_bid + passive_tight - total_shift
        tight_ask = best_ask - passive_tight - total_shift

        tight_bid = min(tight_bid, best_ask - 1)
        tight_ask = max(tight_ask, best_bid + 1)
        if tight_bid >= tight_ask:
            tight_bid = best_bid
            tight_ask = best_ask

        if buy_cap() > 0:
            qty = min(make_size(self.BASE_PASSIVE_SIZE, bid_size_adj), buy_cap())
            orders.append(Order(self.PRODUCT, tight_bid, qty))
            buy_order_volume += qty

        if sell_cap() > 0:
            qty = min(make_size(self.BASE_PASSIVE_SIZE, ask_size_adj), sell_cap())
            orders.append(Order(self.PRODUCT, tight_ask, -qty))
            sell_order_volume += qty

        wide_bid = int(fair) - passive_wide - total_shift
        wide_ask = int(fair) + passive_wide - total_shift

        wide_bid = min(wide_bid, best_ask - 1)
        wide_ask = max(wide_ask, best_bid + 1)
        if wide_bid >= wide_ask:
            wide_bid = best_bid
            wide_ask = best_ask

        if buy_cap() > 0:
            qty = min(make_size(self.BASE_PASSIVE_SIZE + 4, bid_size_adj), buy_cap())
            orders.append(Order(self.PRODUCT, wide_bid, qty))
            buy_order_volume += qty

        if sell_cap() > 0:
            qty = min(make_size(self.BASE_PASSIVE_SIZE + 4, ask_size_adj), sell_cap())
            orders.append(Order(self.PRODUCT, wide_ask, -qty))
            sell_order_volume += qty

        history.append(microprice)
        if len(history) > self.HISTORY_WINDOW:
            history.pop(0)

        print(
            f"[OPUS] t={timestamp} mid={microprice:.2f} fair={fair:.2f} z={z:.2f} "
            f"ar={ar_adj:.2f} mom={momentum:.3f} mom_adj={mom_adj:.3f} "
            f"pos={position} spread={spread} orders={len(orders)}"
        )

        return orders, fair, z

class Trader:
    def __init__(self):
        self.osmium = OsmiumStrategy()

    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        history = data.get("osmium_history", [])

        orders_by_product: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = int(state.position.get(product, 0))

            if product == "ASH_COATED_OSMIUM":
                market_trades = state.market_trades.get(product, [])
                orders, fair, z = self.osmium.run(
                    order_depth, position, history, state.timestamp, market_trades
                )
                orders_by_product[product] = orders
            else:
                orders_by_product[product] = []

        trader_data = json.dumps({
            "osmium_history": history[-250:]
        })

        return orders_by_product, 0, trader_data
