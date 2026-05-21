"""
Audit V37a: V36 with INV_SKEW = 0.030 (peak of osm4k sweep, was 0.025 in V36).
Minor param tweak. Local sweep showed 0.030 gives +14 more than 0.025 (54,385 vs 54,371).

Based on osm4k V10 research findings (`research/osmium_4k_push.md`):
  - wall_mid (deepest bid/ask) feeds EMA, smoother than raw_mid
  - fair_value = EMA(wall_mid) - position * INV_SKEW
  - INV_SKEW = 0.025 (peak at 0.025-0.030 in sweep, wide plateau, 3-of-3 days +)
  - OSMIUM_SOFT_LIMIT = 80 (paired with inv_skew per V10 best config)

Key safety: OSMIUM make pricing UNCHANGED (best_bid+1 / best_ask-1, sacred).
The inv_skew shifts FV used for take-width thresholds only — when long, FV
drops slightly so we're more eager to take rich bids and less eager to take
cheap asks. This is inventory-aware market making, not directional bias.

PEPPER: V29/V34 unchanged (asymmetric bid+2/ask-1, aggressive warmup).
Crash hardening: V34 type-guards on traderData restoration retained.

Local evidence (osm4k V10 + sweep):
  - Baseline (V4 = V34 OSMIUM equivalent): 52,777 OSMIUM 3-day
  - V10 best (wall_mid + inv=0.030 + soft=80): 54,385 OSMIUM 3-day = +1,608
  - inv=0.025: 54,371 (within 14 of peak — flat plateau)
  - All 3 days positive, parameter robustness confirmed

Expected live: +50 to +200 (per Agent 1 audit, applying ~0.1-0.2% transfer
ratio observed for OSMIUM FV-side mechanism class).
"""
import json
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict


class OnlineSlopeEstimator:
    def __init__(self, warmup=500):
        self.warmup = warmup
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.co_xy = 0.0

    def update(self, timestamp, mid_price):
        self.n += 1
        dx = timestamp - self.mean_x
        self.mean_x += dx / self.n
        dy = mid_price - self.mean_y
        self.mean_y += dy / self.n
        dx2 = timestamp - self.mean_x
        self.co_xy += dx * (mid_price - self.mean_y)
        self.m2_x += dx * dx2

    @property
    def slope(self):
        if self.n < 2 or self.m2_x == 0:
            return None
        return self.co_xy / self.m2_x

    def to_dict(self):
        return {"w": self.warmup, "n": self.n, "mx": self.mean_x,
                "my": self.mean_y, "m2x": self.m2_x, "cxy": self.co_xy}

    @classmethod
    def from_dict(cls, d):
        obj = cls(warmup=d["w"])
        obj.n = d["n"]
        obj.mean_x = d["mx"]
        obj.mean_y = d["my"]
        obj.m2_x = d["m2x"]
        obj.co_xy = d["cxy"]
        return obj


class Trader:
    PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
    POSITION_LIMIT = 80

    PEPPER_TAKE_BUY_WIDTH = 1.0
    PEPPER_DRIFT = 0.001
    PEPPER_INTERCEPT_ALPHA = 2.0 / 101.0

    OSMIUM_TAKE_WIDTH = 0.5
    OSMIUM_SOFT_LIMIT = 80           # V36: per V10 best (was 70 in V34)
    OSMIUM_EMA_ALPHA = 2.0 / 251.0
    OSMIUM_INV_SKEW = 0.030          # V37a: local sweep peak

    PEPPER_WARMUP_END = 1000
    PEPPER_AGGRESSIVE_WIDTH = -10

    PEPPER_MAKE_BID_OFFSET = 2

    def bid(self):
        return 1011  

    def run(self, state: TradingState):
        td = {}
        if state.traderData:
            try:
                parsed = json.loads(state.traderData)
                if isinstance(parsed, dict):
                    td = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        osmium_slope_key = "sl_ASH_COATED_OSMIUM"
        try:
            slope_state = td.get(osmium_slope_key)
            if isinstance(slope_state, dict):
                osmium_slope = OnlineSlopeEstimator.from_dict(slope_state)
            else:
                osmium_slope = OnlineSlopeEstimator()
        except (KeyError, TypeError, ValueError):
            osmium_slope = OnlineSlopeEstimator()

        if not isinstance(td.get("ip"), (int, float)):
            td.pop("ip", None)
        if not isinstance(td.get("ema_ASH_COATED_OSMIUM"), (int, float)):
            td.pop("ema_ASH_COATED_OSMIUM", None)

        result: Dict[str, List[Order]] = {}

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            position = state.position.get(product, 0)
            orders: List[Order] = []
            buy_allocated = 0
            sell_allocated = 0

            if product == "INTARIAN_PEPPER_ROOT":
                # V29/V34 PEPPER unchanged
                det = mid_price - self.PEPPER_DRIFT * state.timestamp
                key = "ip"
                if key in td:
                    intercept = self.PEPPER_INTERCEPT_ALPHA * det + \
                                (1 - self.PEPPER_INTERCEPT_ALPHA) * td[key]
                else:
                    intercept = det
                td[key] = intercept
                fair_value = intercept + self.PEPPER_DRIFT * state.timestamp

                max_buy = max(self.POSITION_LIMIT - position, 0)
                max_sell = max(self.POSITION_LIMIT + position, 0)

                if state.timestamp <= self.PEPPER_WARMUP_END:
                    tw = self.PEPPER_AGGRESSIVE_WIDTH
                else:
                    tw = self.PEPPER_TAKE_BUY_WIDTH

                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fair_value - tw:
                        ask_vol = -od.sell_orders[ask_price]
                        can_buy = max_buy - buy_allocated
                        if can_buy <= 0:
                            break
                        qty = min(ask_vol, can_buy)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_allocated += qty
                    else:
                        break

                make_bid = best_bid + self.PEPPER_MAKE_BID_OFFSET
                make_ask = best_ask - 1
                if make_bid < make_ask:
                    can_buy = max_buy - buy_allocated
                    if can_buy > 0:
                        orders.append(Order(product, make_bid, can_buy))
                    can_sell = max_sell - sell_allocated
                    if can_sell > 0:
                        orders.append(Order(product, make_ask, -can_sell))

            else:
                # V36 OSMIUM: wall_mid + inv_skew + soft=80
                wall_bid = min(od.buy_orders.keys())
                wall_ask = max(od.sell_orders.keys())
                wall_mid = (wall_bid + wall_ask) / 2.0

                ema_key = "ema_ASH_COATED_OSMIUM"
                if ema_key in td:
                    ema = self.OSMIUM_EMA_ALPHA * wall_mid + \
                          (1 - self.OSMIUM_EMA_ALPHA) * td[ema_key]
                else:
                    ema = wall_mid
                td[ema_key] = ema

                # Inventory-skewed FV (sacred make logic untouched, only shifts take threshold)
                fair_value = ema - position * self.OSMIUM_INV_SKEW

                osmium_slope.update(state.timestamp, mid_price)

                soft_limit = self.OSMIUM_SOFT_LIMIT
                max_buy = min(soft_limit - position, self.POSITION_LIMIT - position)
                max_sell = min(soft_limit + position, self.POSITION_LIMIT + position)
                max_buy = max(max_buy, 0)
                max_sell = max(max_sell, 0)

                take_width = self.OSMIUM_TAKE_WIDTH

                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fair_value - take_width:
                        ask_vol = -od.sell_orders[ask_price]
                        can_buy = max_buy - buy_allocated
                        if can_buy <= 0:
                            break
                        qty = min(ask_vol, can_buy)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_allocated += qty
                    else:
                        break

                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= fair_value + take_width:
                        bid_vol = od.buy_orders[bid_price]
                        can_sell = max_sell - sell_allocated
                        if can_sell <= 0:
                            break
                        qty = min(bid_vol, can_sell)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_allocated += qty
                    else:
                        break

                # SACRED: make pricing unchanged
                make_bid = best_bid + 1
                make_ask = best_ask - 1
                if make_bid < make_ask:
                    can_buy = max_buy - buy_allocated
                    if can_buy > 0:
                        orders.append(Order(product, make_bid, can_buy))
                    can_sell = max_sell - sell_allocated
                    if can_sell > 0:
                        orders.append(Order(product, make_ask, -can_sell))

            result[product] = orders

        td[osmium_slope_key] = osmium_slope.to_dict()

        return result, 0, json.dumps(td)