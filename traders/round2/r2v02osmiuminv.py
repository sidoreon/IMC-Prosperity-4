# r2v02osmiuminv.py: vNN counts only inside the `osmiuminv` family (same round can have other r2v01… files with different tags).
# Osmium book; inventory / penny / passive mix.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json

PRODUCT = "ASH_COATED_OSMIUM"
POSITION_LIMIT = 80

MIN_PENNY_SPREAD = 4
PASSIVE_OFFSET   = 1
BASE_PASSIVE_SIZE = 12
REDUCE_TRIGGER   = 30
INVENTORY_SKEW   = 0.04
IMBALANCE_THRESHOLD = 0.2

def _imbalance(order_depth: OrderDepth) -> float:
    bid_vol = sum(order_depth.buy_orders.values())
    ask_vol = -sum(order_depth.sell_orders.values())
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return (bid_vol - ask_vol) / total

class Trader:
    def run(self, state: TradingState):
        order_depth = state.order_depths.get(PRODUCT)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return {PRODUCT: []}, 0, ""

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        spread   = best_ask - best_bid

        position = state.position.get(PRODUCT, 0)
        buy_cap  = POSITION_LIMIT - position
        sell_cap = POSITION_LIMIT + position

        orders: List[Order] = []

        if spread >= MIN_PENNY_SPREAD:
            imbalance = _imbalance(order_depth)
            skew      = int(position * INVENTORY_SKEW)

            if imbalance > IMBALANCE_THRESHOLD:
                bid_price = best_bid + 1
                ask_price = best_ask - PASSIVE_OFFSET
            elif imbalance < -IMBALANCE_THRESHOLD:
                bid_price = best_bid + PASSIVE_OFFSET
                ask_price = best_ask - 1
            else:
                bid_price = best_bid + PASSIVE_OFFSET
                ask_price = best_ask - PASSIVE_OFFSET

            bid_price -= skew
            ask_price -= skew

            bid_price = min(bid_price, best_ask - 1)
            ask_price = max(ask_price, best_bid + 1)

            passive_size = BASE_PASSIVE_SIZE // 2 if abs(position) > REDUCE_TRIGGER else BASE_PASSIVE_SIZE

            if buy_cap > 0:
                orders.append(Order(PRODUCT, bid_price, min(passive_size, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(PRODUCT, ask_price, -min(passive_size, sell_cap)))

        return {PRODUCT: orders}, 0, ""
