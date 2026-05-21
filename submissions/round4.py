from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


class Trader:
    OPP_EVAL_HORIZON = 5000
    OPP_MIN_SAMPLES = 8
    OPP_WEIGHT_CAP = 1.75

    INFORMATIVE_FLOW_WEIGHTS: Dict[Tuple[str, str, str], float] = {
        ("VELVETFRUIT_EXTRACT", "Mark 67", "buyer"): 1.35,
        ("VELVETFRUIT_EXTRACT", "Mark 22", "buyer"): 0.85,
        ("VELVETFRUIT_EXTRACT", "Mark 22", "seller"): -0.55,
        ("VELVETFRUIT_EXTRACT", "Mark 01", "seller"): 0.30,
        ("VELVETFRUIT_EXTRACT", "Mark 55", "buyer"): -0.20,
        ("VELVETFRUIT_EXTRACT", "Mark 55", "seller"): 0.20,
    }

    WEAK_CROSSER_WEIGHTS: Dict[Tuple[str, str, str], float] = {
        ("HYDROGEL_PACK", "Mark 38", "seller"): 1.25,
        ("HYDROGEL_PACK", "Mark 38", "buyer"): -1.25,
        ("VEV_4000", "Mark 38", "seller"): 1.10,
        ("VEV_4000", "Mark 38", "buyer"): -1.10,
        ("VELVETFRUIT_EXTRACT", "Mark 55", "seller"): 0.45,
        ("VELVETFRUIT_EXTRACT", "Mark 55", "buyer"): -0.45,
    }

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {product: [] for product in state.order_depths}
        data = json.loads(state.traderData) if state.traderData else {}

        self.log_new_own_trades(state, data)
        self.update_opponent_alpha(state, data)

        if "HYDROGEL_PACK" in state.order_depths:
            result["HYDROGEL_PACK"] = self.trade_hydrogel(
                state,
                state.order_depths["HYDROGEL_PACK"],
                state.position.get("HYDROGEL_PACK", 0),
                data,
            )

        self.trade_velvetfruit_and_options(state, result, data)
        return result, 0, json.dumps(data, separators=(",", ":"))

    def _mid_from_depth(self, depth: Optional[OrderDepth]) -> Optional[float]:
        if depth is None or not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update_opponent_alpha(self, state: TradingState, data: dict) -> None:
        market_trades = getattr(state, "market_trades", {}) or {}
        data.setdefault("opp_stats", {})
        data.setdefault("opp_flow", {})
        data.setdefault("opp_misprice", {})
        data.setdefault("opp_pending", [])
        data.setdefault("opp_online", {})
        data.setdefault("seen_market_trades", [])

        self._settle_pending_opponent_events(state, data)

        seen = set(data["seen_market_trades"])
        new_seen = list(data["seen_market_trades"])
        current_flow: Dict[str, float] = {}
        current_misprice: Dict[str, float] = {}

        for symbol, trades in market_trades.items():
            mid = self._mid_from_depth(state.order_depths.get(symbol))
            if mid is None:
                continue

            for tr in trades:
                ts = int(getattr(tr, "timestamp", state.timestamp))
                price = float(getattr(tr, "price", 0))
                qty = int(getattr(tr, "quantity", 0))
                buyer = str(getattr(tr, "buyer", ""))
                seller = str(getattr(tr, "seller", ""))
                key = f"{symbol}|{ts}|{price}|{qty}|{buyer}|{seller}"
                if key in seen:
                    continue
                seen.add(key)
                new_seen.append(key)

                edge = price - mid
                self._update_opp_stat(data, symbol, buyer, "buyer", edge, qty, ts)
                self._update_opp_stat(data, symbol, seller, "seller", edge, qty, ts)
                self._register_pending_event(data, symbol, buyer, "buyer", ts, mid, edge, qty)
                self._register_pending_event(data, symbol, seller, "seller", ts, mid, edge, qty)

                buy_flow_w = self._opponent_weight(data, symbol, buyer, "buyer", informative=True)
                sell_flow_w = self._opponent_weight(data, symbol, seller, "seller", informative=True)
                current_flow[symbol] = current_flow.get(symbol, 0.0) + buy_flow_w * qty - sell_flow_w * qty

                buy_mis_w = self._opponent_weight(data, symbol, buyer, "buyer", informative=False)
                sell_mis_w = self._opponent_weight(data, symbol, seller, "seller", informative=False)
                scale = max(1.0, mid / 100.0)

                if buy_mis_w != 0.0 and edge > 0.0:
                    current_misprice[symbol] = current_misprice.get(symbol, 0.0) + buy_mis_w * edge * qty / scale
                if sell_mis_w != 0.0 and edge < 0.0:
                    current_misprice[symbol] = current_misprice.get(symbol, 0.0) + sell_mis_w * (-edge) * qty / scale

        for symbol, shock in current_flow.items():
            data["opp_flow"][symbol] = 0.86 * float(data["opp_flow"].get(symbol, 0.0)) + 0.14 * shock
        for symbol in list(data["opp_flow"].keys()):
            if symbol not in current_flow:
                data["opp_flow"][symbol] = 0.94 * float(data["opp_flow"].get(symbol, 0.0))

        for symbol, shock in current_misprice.items():
            data["opp_misprice"][symbol] = 0.82 * float(data["opp_misprice"].get(symbol, 0.0)) + 0.18 * shock
        for symbol in list(data["opp_misprice"].keys()):
            if symbol not in current_misprice:
                data["opp_misprice"][symbol] = 0.90 * float(data["opp_misprice"].get(symbol, 0.0))

        data["seen_market_trades"] = new_seen[-400:]

    def _update_opp_stat(self, data: dict, symbol: str, participant: str, role: str, edge: float, qty: int, timestamp: int) -> None:
        if not participant:
            return
        key = f"{symbol}|{participant}|{role}"
        stats = data["opp_stats"].get(key, {"count": 0, "edge_ema": 0.0, "qty_ema": 0.0, "burst_ema": 0.0, "last_ts": -1})
        prev_ts = int(stats.get("last_ts", -1))
        dt = 999999 if prev_ts < 0 else max(0, timestamp - prev_ts)
        burst = 1.0 if dt <= 5000 else 0.0
        stats["count"] = int(stats["count"]) + 1
        stats["edge_ema"] = 0.92 * float(stats["edge_ema"]) + 0.08 * edge
        stats["qty_ema"] = 0.90 * float(stats["qty_ema"]) + 0.10 * abs(qty)
        stats["burst_ema"] = 0.80 * float(stats["burst_ema"]) + 0.20 * burst
        stats["last_ts"] = int(timestamp)
        data["opp_stats"][key] = stats

    def _register_pending_event(self, data: dict, symbol: str, participant: str, role: str, timestamp: int, mid: float, edge: float, qty: int) -> None:
        if not participant:
            return
        data["opp_pending"].append(
            {
                "symbol": symbol,
                "participant": participant,
                "role": role,
                "eval_ts": int(timestamp) + self.OPP_EVAL_HORIZON,
                "mid": float(mid),
                "edge": float(edge),
                "qty": int(abs(qty)),
            }
        )

    def _settle_pending_opponent_events(self, state: TradingState, data: dict) -> None:
        remaining = []
        for event in data["opp_pending"]:
            current_mid = self._mid_from_depth(state.order_depths.get(event["symbol"]))
            if current_mid is None or int(state.timestamp) < int(event["eval_ts"]):
                remaining.append(event)
                continue

            side_sign = 1.0 if event["role"] == "buyer" else -1.0
            move = side_sign * (current_mid - float(event["mid"]))
            edge = float(event["edge"])
            qty = float(event["qty"])
            informative_outcome = self._clip(move / 4.0, -1.5, 1.5)
            if event["role"] == "buyer":
                cross_badness = self._clip((max(0.0, edge) - move) / 4.0, -1.5, 1.5)
            else:
                cross_badness = self._clip((max(0.0, -edge) - move) / 4.0, -1.5, 1.5)

            self._update_online_weight(data, event["symbol"], event["participant"], event["role"], True, informative_outcome, qty)
            self._update_online_weight(data, event["symbol"], event["participant"], event["role"], False, cross_badness, qty)

        data["opp_pending"] = remaining

    def _update_online_weight(self, data: dict, symbol: str, participant: str, role: str, informative: bool, outcome: float, qty: float) -> None:
        if not participant:
            return
        family = "flow" if informative else "mis"
        key = f"{family}|{symbol}|{participant}|{role}"
        record = data["opp_online"].get(key, {"score": 0.0, "count": 0.0, "weight": 0.0})
        qty_weight = self._clip(qty / 8.0, 0.35, 1.5)
        score = 0.94 * float(record["score"]) + 0.06 * outcome
        count = min(float(record["count"]) + qty_weight, 200.0)
        confidence = self._clip(count / self.OPP_MIN_SAMPLES, 0.0, 1.0)
        record["score"] = score
        record["count"] = count
        record["weight"] = confidence * self._clip(score, -self.OPP_WEIGHT_CAP, self.OPP_WEIGHT_CAP)
        data["opp_online"][key] = record

    def _opponent_weight(self, data: dict, symbol: str, participant: str, role: str, informative: bool) -> float:
        if not participant:
            return 0.0
        priors = self.INFORMATIVE_FLOW_WEIGHTS if informative else self.WEAK_CROSSER_WEIGHTS
        base = float(priors.get((symbol, participant, role), 0.0))
        family = "flow" if informative else "mis"
        learned = float(data["opp_online"].get(f"{family}|{symbol}|{participant}|{role}", {}).get("weight", 0.0))
        if base == 0.0:
            return learned
        return self._clip(0.65 * base + 0.70 * learned, -self.OPP_WEIGHT_CAP, self.OPP_WEIGHT_CAP)

    def log_new_own_trades(self, state: TradingState, data: dict) -> None:
        own_trades = getattr(state, "own_trades", {}) or {}
        data.setdefault("seen_own_trade_logs", [])
        seen = set(data["seen_own_trade_logs"])
        new_seen = list(data["seen_own_trade_logs"])
        watched = {"Mark 38", "Mark 67", "Mark 22", "Mark 14", "Mark 55", "Mark 01"}

        for symbol, trades in own_trades.items():
            for tr in trades:
                ts = int(getattr(tr, "timestamp", state.timestamp))
                price = int(getattr(tr, "price", 0))
                qty = int(getattr(tr, "quantity", 0))
                buyer = str(getattr(tr, "buyer", ""))
                seller = str(getattr(tr, "seller", ""))
                key = f"{symbol}|{ts}|{price}|{qty}|{buyer}|{seller}"
                if key in seen:
                    continue
                seen.add(key)
                new_seen.append(key)
                counterparty = None
                side = None
                if buyer == "SUBMISSION" and seller in watched:
                    counterparty = seller
                    side = "BUY"
                elif seller == "SUBMISSION" and buyer in watched:
                    counterparty = buyer
                    side = "SELL"
                if counterparty is not None:
                    print(
                        f"FILL ts={ts} symbol={symbol} side={side} qty={qty} px={price} "
                        f"counterparty={counterparty} opp_flow={self._opp_flow(data, symbol):.3f} "
                        f"opp_mis={self._opp_misprice(data, symbol):.3f}"
                    )

        data["seen_own_trade_logs"] = new_seen[-300:]

    def _opp_flow(self, data: dict, symbol: str) -> float:
        return float(data["opp_flow"].get(symbol, 0.0))

    def _opp_misprice(self, data: dict, symbol: str) -> float:
        return float(data["opp_misprice"].get(symbol, 0.0))

    def trade_hydrogel(self, state: TradingState, order_depth: OrderDepth, position: int, data: dict) -> List[Order]:
        orders: List[Order] = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        bb = max(order_depth.buy_orders)
        ba = min(order_depth.sell_orders)
        bid_vol = order_depth.buy_orders[bb]
        ask_vol = -order_depth.sell_orders[ba]
        mid = (bb + ba) / 2.0

        pstate = data.setdefault("hg", {})
        alpha_ema = 2.0 / 61.0
        alpha_slow = 1.0 - 0.5 ** (1.0 / 350.0)
        alpha_vol = 2.0 / 101.0

        ema = (1 - alpha_ema) * float(pstate.get("ema", mid)) + alpha_ema * mid
        slow_mean = (1 - alpha_slow) * float(pstate.get("slow_mean", mid)) + alpha_slow * mid
        prev_var = float(pstate.get("slow_var", 1.0))
        resid = mid - slow_mean
        slow_var = max((1 - alpha_slow) * prev_var + alpha_slow * resid * resid, 1.0)
        prev_mid = pstate.get("prev_mid")
        prev_vol = float(pstate.get("vol", 1.0))
        vol = max((1 - alpha_vol) * prev_vol + alpha_vol * abs(mid - float(prev_mid)), 0.5) if prev_mid is not None else max(prev_vol, 0.5)
        ticks = int(pstate.get("ticks", 0)) + 1
        pstate.update({"ema": ema, "slow_mean": slow_mean, "slow_var": slow_var, "vol": vol, "prev_mid": mid, "ticks": ticks})
        if ticks < 60:
            return orders

        tot = abs(order_depth.buy_orders[bb]) + abs(order_depth.sell_orders[ba])
        micro = (abs(order_depth.sell_orders[ba]) * bb + abs(order_depth.buy_orders[bb]) * ba) / tot if tot > 0 else mid
        std = max(slow_var ** 0.5, 1.0)
        z = (mid - slow_mean) / std
        mr_off = 0.0 if abs(z) < 1.25 else self._clip(-0.12 * (mid - slow_mean), -2.0 * max(vol, 1.0), 2.0 * max(vol, 1.0))

        limit = 200
        buy_room = limit - position
        sell_room = limit + position
        opp_flow = self._opp_flow(data, "HYDROGEL_PACK")
        opp_mis = self._opp_misprice(data, "HYDROGEL_PACK")
        expected = ema + 0.30 * (micro - ema) + mr_off - 1.2 * (position / float(limit)) * max(vol, 1.0) + 1.25 * opp_flow + 2.40 * opp_mis
        take_edge = 2.3 + 0.10 * max(vol, 0.0)

        if buy_room > 0 and ba <= expected - take_edge:
            qty = min(buy_room, ask_vol, 10)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(ba), int(qty)))
                buy_room -= qty
        elif buy_room > 0 and opp_mis > 0.35 and ba <= expected + 0.75:
            qty = min(buy_room, ask_vol, 3)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(ba), int(qty)))
                buy_room -= qty

        if sell_room > 0 and bb >= expected + take_edge:
            qty = min(sell_room, bid_vol, 10)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(bb), -int(qty)))
        elif sell_room > 0 and opp_mis < -0.35 and bb >= expected - 0.75:
            qty = min(sell_room, bid_vol, 3)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(bb), -int(qty)))

        return orders

    def trade_velvetfruit_and_options(self, state: TradingState, result: dict, data: dict) -> None:
        underlying = "VELVETFRUIT_EXTRACT"
        option_products = [p for p in state.order_depths if p.startswith("VEV_")]
        if underlying not in state.order_depths:
            return
        depth = state.order_depths[underlying]
        if not depth.buy_orders or not depth.sell_orders:
            return

        best_bid = max(depth.buy_orders)
        best_ask = min(depth.sell_orders)
        bid_vol = depth.buy_orders[best_bid]
        ask_vol = -depth.sell_orders[best_ask]
        mid = (best_bid + best_ask) / 2.0

        base_fair = 5250.0
        alpha = 0.0005
        data["fair"] = alpha * mid + (1 - alpha) * float(data.get("fair", base_fair))
        if "last_mid" not in data:
            data["last_mid"] = mid
        momentum = mid - float(data["last_mid"])
        data["last_mid"] = mid

        opp_flow = self._opp_flow(data, underlying)
        opp_mis = self._opp_misprice(data, underlying)
        fair = float(data["fair"])
        signal = fair - mid - 0.1 * momentum + 5.0 * opp_flow + 2.0 * opp_mis
        if state.timestamp < 5000:
            return

        position = state.position.get(underlying, 0)
        limit = 200
        buy_room = limit - position
        sell_room = limit + position
        adjusted_fair = fair - 0.04 * position + 0.1 * momentum + 3.0 * opp_flow + 1.2 * opp_mis
        take_threshold = 4.0

        if buy_room > 0 and best_ask <= adjusted_fair - take_threshold:
            qty = min(buy_room, ask_vol, 12)
            if qty > 0:
                result[underlying].append(Order(underlying, int(best_ask), int(qty)))
        elif buy_room > 0 and signal > 7.0:
            qty = min(buy_room, ask_vol, 8)
            if qty > 0:
                result[underlying].append(Order(underlying, int(best_ask), int(qty)))

        if sell_room > 0 and best_bid >= adjusted_fair + take_threshold:
            qty = min(sell_room, bid_vol, 12)
            if qty > 0:
                result[underlying].append(Order(underlying, int(best_bid), -int(qty)))
        elif sell_room > 0 and signal < -7.0:
            qty = min(sell_room, bid_vol, 8)
            if qty > 0:
                result[underlying].append(Order(underlying, int(best_bid), -int(qty)))

        option_limit = 300
        t = 4 / 252
        assumed_vol = 0.25
        base_option_size = 15
        buy_call_threshold = 15.0
        sell_call_threshold = -5.0

        for option in option_products:
            try:
                strike = int(option.split("_")[1])
            except Exception:
                continue

            opt_depth = state.order_depths.get(option)
            if opt_depth is None or not opt_depth.buy_orders or not opt_depth.sell_orders:
                continue

            opt_bid = max(opt_depth.buy_orders)
            opt_ask = min(opt_depth.sell_orders)
            opt_bid_volume = opt_depth.buy_orders[opt_bid]
            opt_ask_volume = -opt_depth.sell_orders[opt_ask]
            opt_position = state.position.get(option, 0)
            opt_buy_room = option_limit - opt_position
            opt_sell_room = option_limit + opt_position

            d1 = (math.log(mid / strike) + 0.5 * assumed_vol ** 2 * t) / (assumed_vol * math.sqrt(t))
            delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
            if delta < 0.05:
                continue

            option_opp_mis = self._opp_misprice(data, option)
            option_signal = signal * delta + 8.0 * option_opp_mis
            option_size = max(5, int(base_option_size * delta))

            if option == "VEV_4000":
                option_size = max(option_size, 10)
                if option_opp_mis > 0.15:
                    option_signal += 8.0
                elif option_opp_mis < -0.15:
                    option_signal -= 8.0

            if option_signal > buy_call_threshold and opt_buy_room > 0:
                qty = min(option_size, opt_ask_volume, opt_buy_room)
                if qty > 0:
                    result[option].append(Order(option, int(opt_ask), int(qty)))

            if option_signal < sell_call_threshold and opt_sell_room > 0:
                qty = min(option_size, opt_bid_volume, opt_sell_room)
                if qty > 0:
                    result[option].append(Order(option, int(opt_bid), -int(qty)))