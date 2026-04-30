# r2v03peposminv.py: vNN counts only inside the `peposminv` family (same round can have other r2v01… files with different tags).
# Pepper (+ osmium); inventory / paired risk.
#
from datamodel import OrderDepth, TradingState, Order
import json
import math

OSM_FAIR       = 10001
OSM_AR1        = -0.5011
OSM_HALF_SPD   = 8
OSM_INV_SKEW   = 0.020
OSM_LIMIT      = 80
OSM_MAKE_CAP   = 50

OSM_TAKE_W     = 3

PEP_AR1        = -0.4988
PEP_LIMIT      = 80
PEP_BASE_W     = 2
PEP_STOP_DRAW  = 480

PEP_REARM      = 200
PEP_ASK_THRESH = 70

class Trader:

    def run(self, state: TradingState):
        result = {}
        td = json.loads(state.traderData) if state.traderData else {}

        osm_orders = []
        if "ASH_COATED_OSMIUM" in state.order_depths:
            od: OrderDepth = state.order_depths["ASH_COATED_OSMIUM"]
            pos = state.position.get("ASH_COATED_OSMIUM", 0)

            best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid  = (best_bid + best_ask) / 2.0
                prev = td.get("osm_prev", mid)

                fair = mid + OSM_AR1 * (mid - prev)

                skew   = -pos * OSM_INV_SKEW
                bid_px = round(fair + skew - OSM_HALF_SPD)
                ask_px = round(fair + skew + OSM_HALF_SPD)

                if best_ask <= fair - OSM_TAKE_W and pos < OSM_LIMIT:
                    qty = min(-od.sell_orders[best_ask], OSM_LIMIT - pos)
                    if qty > 0:
                        osm_orders.append(Order("ASH_COATED_OSMIUM", best_ask, qty))

                if best_bid >= fair + OSM_TAKE_W and pos > -OSM_LIMIT:
                    qty = min(od.buy_orders[best_bid], pos + OSM_LIMIT)
                    if qty > 0:
                        osm_orders.append(Order("ASH_COATED_OSMIUM", best_bid, -qty))

                make_b = max(min(OSM_MAKE_CAP - pos, OSM_LIMIT - pos), 0)
                make_a = max(min(OSM_MAKE_CAP + pos, OSM_LIMIT + pos), 0)

                if make_b > 0:
                    osm_orders.append(Order("ASH_COATED_OSMIUM", bid_px, make_b))
                if make_a > 0:
                    osm_orders.append(Order("ASH_COATED_OSMIUM", ask_px, -make_a))

                td["osm_prev"] = mid

        result["ASH_COATED_OSMIUM"] = osm_orders

        pep_orders = []
        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            od: OrderDepth = state.order_depths["INTARIAN_PEPPER_ROOT"]
            pos = state.position.get("INTARIAN_PEPPER_ROOT", 0)

            best_bid = max(od.buy_orders.keys())  if od.buy_orders  else None
            best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid  = (best_bid + best_ask) / 2.0
                prev = td.get("pep_prev", mid)

                delta   = mid - prev
                ar1_adj = -PEP_AR1 * delta
                buy_w   = max(PEP_BASE_W + ar1_adj, 0)
                sell_w  = max(PEP_BASE_W - ar1_adj, 0)

                stopped    = td.get("pep_stopped",    False)
                stop_level = td.get("pep_stop_level", mid)
                peak       = td.get("pep_peak",       mid)

                if pos > 0:
                    peak = max(peak, mid)
                    if peak - mid >= PEP_STOP_DRAW:
                        stopped    = True
                        stop_level = mid
                elif pos < 0:
                    peak = min(peak, mid)
                    if mid - peak >= PEP_STOP_DRAW:
                        stopped    = True
                        stop_level = mid
                else:
                    peak = mid

                if stopped and (abs(mid - stop_level) >= PEP_REARM or pos == 0):
                    stopped = False

                if not stopped:
                    fair_pep = mid + PEP_AR1 * delta

                    if best_ask <= fair_pep - buy_w and pos < PEP_LIMIT:
                        qty = min(-od.sell_orders[best_ask], PEP_LIMIT - pos)
                        if qty > 0:
                            pep_orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, qty))

                    if best_bid >= fair_pep + sell_w and pos > -PEP_LIMIT:
                        qty = min(od.buy_orders[best_bid], pos + PEP_LIMIT)
                        if qty > 0:
                            pep_orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))

                    make_bid_px = math.floor(fair_pep) - 1
                    make_ask_px = math.ceil(fair_pep)  + 1

                    make_b = max(PEP_LIMIT - pos, 0)
                    if make_b > 0:
                        pep_orders.append(Order("INTARIAN_PEPPER_ROOT", make_bid_px, make_b))

                    if pos >= PEP_ASK_THRESH:
                        make_a = min(pos + PEP_LIMIT, 2 * PEP_LIMIT)
                        if make_a > 0:
                            pep_orders.append(Order("INTARIAN_PEPPER_ROOT", make_ask_px, -make_a))

                td["pep_prev"]       = mid
                td["pep_peak"]       = peak
                td["pep_stopped"]    = stopped
                td["pep_stop_level"] = stop_level

        result["INTARIAN_PEPPER_ROOT"] = pep_orders

        return result, 0, json.dumps(td)
