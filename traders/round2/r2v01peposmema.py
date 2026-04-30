# r2v01peposmema.py: vNN counts only inside the `peposmema` family (same round can have other r2v01… files with different tags).
# Pepper (+ osmium); EMA stack.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List
import json

class Trader:
    PRODUCTS = ["INTARIAN_PEPPER_ROOT", "ASH_COATED_OSMIUM"]
    POSITION_LIMIT = 80

    WARMUP_TICKS = 30

    OSMIUM_TAKE_WIDTH = 0.5
    OSMIUM_SOFT_LIMIT = 80
    OSMIUM_EMA_ALPHA  = 2.0 / 251.0
    OSMIUM_INV_SKEW   = 0.030

    PEPPER_TAKE_BUY_WIDTH    = 1.0
    PEPPER_DRIFT             = 0.001
    PEPPER_INTERCEPT_ALPHA   = 2.0 / 101.0
    PEPPER_WARMUP_END        = 1000
    PEPPER_AGGRESSIVE_WIDTH  = -10
    PEPPER_MAKE_BID_OFFSET   = 2

    def run(self, state: TradingState):
        td = {}
        if state.traderData:
            try:
                parsed = json.loads(state.traderData)
                if isinstance(parsed, dict):
                    td = parsed
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if not isinstance(td.get("ip"), (int, float)):
            td.pop("ip", None)
        if not isinstance(td.get("ema_osm"), (int, float)):
            td.pop("ema_osm", None)

        osm_history = td.get("osm_history", [])

        result: Dict[str, List[Order]] = {}

        for product in self.PRODUCTS:
            if product not in state.order_depths:
                continue
            od = state.order_depths[product]
            if not od.buy_orders or not od.sell_orders:
                result[product] = []
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            mid_price = (best_bid + best_ask) / 2.0
            position = state.position.get(product, 0)
            orders: List[Order] = []
            buy_allocated = 0
            sell_allocated = 0

            if product == "INTARIAN_PEPPER_ROOT":

                det = mid_price - self.PEPPER_DRIFT * state.timestamp
                if "ip" in td:
                    intercept = self.PEPPER_INTERCEPT_ALPHA * det +\
                                (1 - self.PEPPER_INTERCEPT_ALPHA) * td["ip"]
                else:
                    intercept = det
                td["ip"] = intercept
                fair_value = intercept + self.PEPPER_DRIFT * state.timestamp

                max_buy  = max(self.POSITION_LIMIT - position, 0)
                max_sell = max(self.POSITION_LIMIT + position, 0)

                tw = self.PEPPER_AGGRESSIVE_WIDTH if state.timestamp <= self.PEPPER_WARMUP_END\
                     else self.PEPPER_TAKE_BUY_WIDTH

                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fair_value - tw:
                        can_buy = max_buy - buy_allocated
                        if can_buy <= 0:
                            break
                        qty = min(-od.sell_orders[ask_price], can_buy)
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

                wall_bid = min(od.buy_orders.keys())
                wall_ask = max(od.sell_orders.keys())
                wall_mid = (wall_bid + wall_ask) / 2.0

                if len(osm_history) < self.WARMUP_TICKS:

                    ema = td.get("ema_osm", wall_mid)
                    ema = self.OSMIUM_EMA_ALPHA * wall_mid + (1 - self.OSMIUM_EMA_ALPHA) * ema
                    td["ema_osm"] = ema
                    fair_value = ema - position * self.OSMIUM_INV_SKEW
                else:

                    n   = len(osm_history)
                    sx  = sum(h[0] for h in osm_history)
                    sy  = sum(h[1] for h in osm_history)
                    sxx = sum(h[0] * h[0] for h in osm_history)
                    sxy = sum(h[0] * h[1] for h in osm_history)
                    denom = n * sxx - sx * sx
                    if denom != 0:
                        slope     = (n * sxy - sx * sy) / denom
                        intercept = (sy - slope * sx) / n
                        fair_value = slope * state.timestamp + intercept
                    else:
                        fair_value = sy / n
                    fair_value -= position * self.OSMIUM_INV_SKEW

                osm_history.append([state.timestamp, wall_mid])
                if len(osm_history) > self.WARMUP_TICKS:
                    osm_history.pop(0)

                max_buy  = max(min(self.OSMIUM_SOFT_LIMIT - position, self.POSITION_LIMIT - position), 0)
                max_sell = max(min(self.OSMIUM_SOFT_LIMIT + position, self.POSITION_LIMIT + position), 0)

                for ask_price in sorted(od.sell_orders.keys()):
                    if ask_price <= fair_value - self.OSMIUM_TAKE_WIDTH:
                        can_buy = max_buy - buy_allocated
                        if can_buy <= 0:
                            break
                        qty = min(-od.sell_orders[ask_price], can_buy)
                        if qty > 0:
                            orders.append(Order(product, ask_price, qty))
                            buy_allocated += qty
                    else:
                        break

                for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_price >= fair_value + self.OSMIUM_TAKE_WIDTH:
                        can_sell = max_sell - sell_allocated
                        if can_sell <= 0:
                            break
                        qty = min(od.buy_orders[bid_price], can_sell)
                        if qty > 0:
                            orders.append(Order(product, bid_price, -qty))
                            sell_allocated += qty
                    else:
                        break

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

        td["osm_history"] = osm_history
        return result, 0, json.dumps(td)
