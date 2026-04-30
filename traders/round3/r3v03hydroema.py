# r3v03hydroema.py: vNN counts only inside the `hydroema` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; EMA fair / smooth mid.
#
from datamodel import OrderDepth, UserId, TradingState, Order
import json
from collections import defaultdict

class Logger:
    def __init__(self):
        self.logs = ""

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str):
        print(json.dumps([
            [
                state.timestamp,
                state.traderData,
                [[l.symbol, l.product, l.denomination] for l in state.listings.values()],
                {s: [od.buy_orders, od.sell_orders] for s, od in state.order_depths.items()},
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for trades in state.own_trades.values() for t in trades],
                [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp] for trades in state.market_trades.values() for t in trades],
                dict(state.position),
                [state.observations.plainValueObservations, {
                    p: [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff, o.importTariff, o.sugarPrice, o.sunlightIndex]
                    for p, o in state.observations.conversionObservations.items()
                }],
            ],
            [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr],
            conversions,
            trader_data,
            self.logs,
        ], separators=(",", ":")))
        self.logs = ""

logger = Logger()

EMA_ALPHA = 0.30
PASSIVE_SPREAD = 5
QUOTE_LIM = 15
POS_LIM = 200

class Trader:
    def run(self, state: TradingState):
        result = defaultdict(list)
        conversions = 0

        shared = {}
        if state.traderData:
            try:
                shared = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        result['HYDROGEL_PACK'], hp_data = self.hydrogel(state, shared)

        traderData = json.dumps(hp_data)
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def hydrogel(self, state: TradingState, shared: dict):
        product = 'HYDROGEL_PACK'
        result = []

        if product not in state.order_depths:
            return result, shared

        od = state.order_depths[product]
        bids = sorted(od.buy_orders, reverse=True)
        asks = sorted(od.sell_orders)

        if bids and asks:
            mid = (bids[0] + asks[0]) / 2.0
        elif bids:
            mid = float(bids[0])
        elif asks:
            mid = float(asks[0])
        else:
            mid = shared.get("hp_ema", 10000.0)

        ema = shared.get("hp_ema", mid)
        ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * ema
        shared["hp_ema"] = ema
        fair = ema

        pos = state.position.get(product, 0)
        buy_cap = POS_LIM - pos
        sell_cap = POS_LIM + pos

        for ask in asks:
            if ask < fair and buy_cap > 0:
                qty = min(buy_cap, abs(od.sell_orders[ask]))
                result.append(Order(product, ask, qty))
                buy_cap -= qty
        for bid in bids:
            if bid > fair and sell_cap > 0:
                qty = min(sell_cap, od.buy_orders[bid])
                result.append(Order(product, bid, -qty))
                sell_cap -= qty

        fair_int = round(fair)
        if buy_cap > 0:
            result.append(Order(product, fair_int - PASSIVE_SPREAD, min(QUOTE_LIM, buy_cap)))
        if sell_cap > 0:
            result.append(Order(product, fair_int + PASSIVE_SPREAD, -min(QUOTE_LIM, sell_cap)))

        return result, shared