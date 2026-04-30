# r3v06hvoptinv.py: vNN counts only inside the `hvoptinv` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; inventory skew and reduce logic.
#
from datamodel import OrderDepth, TradingState, Order
from typing import List
import json
import math

class Trader:

    def run(self, state: TradingState):
        result = {}
        data = json.loads(state.traderData) if state.traderData else {}

        for product in state.order_depths:
            result[product] = []

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self.trade_hydrogel(
                order_depth=state.order_depths["HYDROGEL_PACK"],
                position=state.position.get("HYDROGEL_PACK", 0),
                data=data,
            )

        self.trade_velvetfruit_and_options(state=state, result=result, data=data)

        return result, 0, json.dumps(data)

    def trade_hydrogel(self, order_depth: OrderDepth, position: int, data: dict) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        bb = max(order_depth.buy_orders)
        ba = min(order_depth.sell_orders)
        mid = (bb + ba) / 2.0
        spread = ba - bb

        pstate = data.setdefault("hg", {})

        ALPHA_EMA  = 2.0 / 61.0
        ALPHA_SLOW = 1.0 - 0.5 ** (1.0 / 350.0)
        ALPHA_VOL  = 2.0 / 101.0

        ema       = (1 - ALPHA_EMA)  * float(pstate.get("ema",       mid)) + ALPHA_EMA  * mid
        slow_mean = (1 - ALPHA_SLOW) * float(pstate.get("slow_mean", mid)) + ALPHA_SLOW * mid
        prev_var  = float(pstate.get("slow_var", 1.0))
        resid     = mid - slow_mean
        slow_var  = max((1 - ALPHA_SLOW) * prev_var + ALPHA_SLOW * resid * resid, 1.0)

        prev_mid = pstate.get("prev_mid")
        prev_vol = float(pstate.get("vol", 1.0))
        vol = max((1 - ALPHA_VOL) * prev_vol + ALPHA_VOL * abs(mid - float(prev_mid)), 0.5)\
              if prev_mid is not None else max(prev_vol, 0.5)

        ticks = int(pstate.get("ticks", 0)) + 1
        pstate.update({"ema": ema, "slow_mean": slow_mean, "slow_var": slow_var,
                       "vol": vol, "prev_mid": mid, "ticks": ticks})

        if ticks < 60:
            return orders

        bsz = abs(order_depth.buy_orders[bb])
        asz = abs(order_depth.sell_orders[ba])
        tot = bsz + asz
        micro = (asz * bb + bsz * ba) / tot if tot > 0 else mid

        std = max(slow_var ** 0.5, 1.0)
        z   = (mid - slow_mean) / std
        if abs(z) >= 1.25:
            pull   = -0.12 * (mid - slow_mean)
            cap    = 2.0 * max(vol, 1.0)
            mr_off = max(-cap, min(cap, pull))
        else:
            mr_off = 0.0

        LIMIT     = 200
        inv_skew  = 1.2 * (position / float(LIMIT)) * max(vol, 1.0)
        micro_off = 0.30 * (micro - ema)
        expected  = ema + micro_off + mr_off - inv_skew

        base_make = 4.0 * 0.75
        make_edge = base_make + 0.08 * max(vol, 0.0)

        if (int(state.timestamp) % 1_000_000) > 990_000:
            make_edge *= 1.25

        max_bid = expected - make_edge
        min_ask = expected + make_edge

        if spread >= 3:
            cand_b = bb + 1
            mb = cand_b if cand_b <= max_bid else (bb if bb <= max_bid else None)
            cand_a = ba - 1
            ma = cand_a if cand_a >= min_ask else (ba if ba >= min_ask else None)
        else:
            mb = bb if bb <= max_bid else None
            ma = ba if ba >= min_ask else None

        if mb is not None and mb >= ba:
            mb = None
        if ma is not None and ma <= bb:
            ma = None
        if mb is not None and ma is not None and mb >= ma:
            mb, ma = None, None

        buy_room  = LIMIT - position
        sell_room = LIMIT + position
        base_size = 5
        ms_buy  = base_size + 1 if position < 0 else base_size
        ms_sell = base_size + 1 if position > 0 else base_size

        if mb is not None and buy_room > 0:
            orders.append(Order("HYDROGEL_PACK", int(mb), min(ms_buy, buy_room)))
        if ma is not None and sell_room > 0:
            orders.append(Order("HYDROGEL_PACK", int(ma), -min(ms_sell, sell_room)))

        return orders

    def trade_velvetfruit_and_options(self, state: TradingState, result: dict, data: dict) -> None:
        underlying = "VELVETFRUIT_EXTRACT"
        option_products = [p for p in state.order_depths if p.startswith("VEV_")]

        if underlying not in state.order_depths:
            return
        order_depth = state.order_depths[underlying]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid = (best_bid + best_ask) / 2

        BASE_FAIR = 5250
        ALPHA = 0.002

        if "fair" not in data:
            data["fair"] = BASE_FAIR
        fair = ALPHA * mid + (1 - ALPHA) * data["fair"]
        data["fair"] = fair

        if "last_mid" not in data:
            data["last_mid"] = mid
        momentum = mid - data["last_mid"]
        data["last_mid"] = mid

        MOMENTUM_SKEW = 0.1
        signal = fair - mid - MOMENTUM_SKEW * momentum

        if state.timestamp < 5000:
            return

        position = state.position.get(underlying, 0)
        UNDERLYING_LIMIT = 200
        PASSIVE_SIZE = 10
        PASSIVE_EDGE = 2
        INVENTORY_SKEW = 0.1

        buy_room  = UNDERLYING_LIMIT - position
        sell_room = UNDERLYING_LIMIT + position
        adjusted_fair = fair - INVENTORY_SKEW * position + MOMENTUM_SKEW * momentum

        bid_price = int(round(adjusted_fair - PASSIVE_EDGE))
        ask_price = int(round(adjusted_fair + PASSIVE_EDGE))
        bid_price = min(bid_price, best_ask - 1)
        ask_price = max(ask_price, best_bid + 1)

        if buy_room > 0:
            result[underlying].append(Order(underlying, bid_price, min(PASSIVE_SIZE, buy_room)))
        if sell_room > 0:
            result[underlying].append(Order(underlying, ask_price, -min(PASSIVE_SIZE, sell_room)))

        OPTION_LIMIT = 300
        T = 4 / 252
        ASSUMED_VOL = 0.15
        BASE_OPTION_SIZE = 15
        BUY_CALL_THRESHOLD = 12
        SELL_CALL_THRESHOLD = -5

        for option in option_products:
            if option not in state.order_depths:
                continue
            try:
                strike = int(option.split("_")[1])
            except Exception:
                continue

            depth = state.order_depths[option]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            opt_bid = max(depth.buy_orders.keys())
            opt_ask = min(depth.sell_orders.keys())
            opt_bid_volume =  depth.buy_orders[opt_bid]
            opt_ask_volume = -depth.sell_orders[opt_ask]
            opt_position  = state.position.get(option, 0)
            opt_buy_room  = OPTION_LIMIT - opt_position
            opt_sell_room = OPTION_LIMIT + opt_position

            d1 = (math.log(mid / strike) + 0.5 * ASSUMED_VOL ** 2 * T) / (ASSUMED_VOL * math.sqrt(T))
            delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

            if delta < 0.05:
                continue

            option_signal = signal * delta
            option_size   = max(5, int(BASE_OPTION_SIZE * delta))

            if option_signal > BUY_CALL_THRESHOLD and opt_buy_room > 0:
                qty = min(option_size, opt_ask_volume, opt_buy_room)
                if qty > 0:
                    result[option].append(Order(option, opt_ask, qty))

            if option_signal < SELL_CALL_THRESHOLD and opt_sell_room > 0:
                qty = min(option_size, opt_bid_volume, opt_sell_room)
                if qty > 0:
                    result[option].append(Order(option, opt_bid, -qty))