# r3v32hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List, Tuple, Optional
import json
import math

class Trader:
    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300,
        "VEV_4500": 300,
        "VEV_5000": 300,
        "VEV_5100": 300,
        "VEV_5200": 300,
        "VEV_5300": 300,
        "VEV_5400": 300,
        "VEV_5500": 300,
        "VEV_6000": 300,
        "VEV_6500": 300,
    }

    HG_FAIR = 9991
    HG_TAKE_EDGE = 26
    HG_TAKE_SIZE = 20
    HG_POST_EDGE = 8
    HG_POST_SIZE = 20
    HG_SKEW_DIV = 40

    HG_MOM_ALPHA = 0.12
    HG_MOM_K = 0.45
    HG_MOM_CAP = 6
    HG_MOM_TAKE_BOOST = 1.3
    HG_MOM_KEY = "HYDROGEL_PACK_ema"

    VE_ANCHOR = 5262.0
    VE_ANCHOR_WEIGHT = 0.6
    VE_EMA_ALPHA = 0.08
    VE_IMBALANCE_K = 2.0
    VE_IMBALANCE_CAP = 2.0

    VE_EDGE = 7
    VE_SIZE = 20
    VE_POST_EDGE = 1
    VE_POST_SIZE = 20

    VOUCHER_STRIKES: Dict[str, int] = {
        "VEV_4000": 4000,
        "VEV_4500": 4500,
        "VEV_5000": 5000,
        "VEV_5100": 5100,
        "VEV_5200": 5200,
        "VEV_5300": 5300,
        "VEV_5400": 5400,
        "VEV_5500": 5500,
        "VEV_6000": 6000,
        "VEV_6500": 6500,
    }

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        mids: Dict[str, float] = {}
        micros: Dict[str, float] = {}
        spreads: Dict[str, float] = {}

        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue
            micro = self._microprice(depth)
            mids[product] = mid
            micros[product] = micro if micro is not None else mid
            bid, _ = self._best_bid(depth)
            ask, _ = self._best_ask(depth)
            if bid is not None and ask is not None:
                spreads[product] = ask - bid
            self._update_stats(data, product, mid)

        total_pnl = self._update_and_estimate_pnl(data, state, mids)
        data["peak_pnl"] = max(float(data.get("peak_pnl", total_pnl)), float(total_pnl))

        extract_fair = None
        if "VELVETFRUIT_EXTRACT" in mids and "VELVETFRUIT_EXTRACT" in state.order_depths:
            extract_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            extract_fair = self._extract_fair(data, state.order_depths["VELVETFRUIT_EXTRACT"], mids, micros, extract_pos)

        if extract_fair is not None:
            data["last_extract_fair"] = extract_fair
        else:
            extract_fair = data.get("last_extract_fair")

        if extract_fair is not None:
            self._update_voucher_sigma(data, state.order_depths, extract_fair, state.timestamp)

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product == "HYDROGEL_PACK":
                orders = self._trade_hydrogel(depth, pos, data)

            elif product == "VELVETFRUIT_EXTRACT":
                fair = self._extract_fair(data, depth, mids, micros, pos)
                orders = self._trade_delta_one(
                    product, depth, pos, self.POSITION_LIMITS[product], data, fair
                )

            elif product in self.VOUCHER_STRIKES and extract_fair is not None:
                fair = self._voucher_fair(product, extract_fair, state.timestamp, data)
                delta = self._voucher_delta(product, extract_fair, state.timestamp, data)
                orders = self._trade_voucher(product, depth, pos, fair, extract_fair, delta, data)

            if orders:
                result[product] = orders
            else:
                result[product] = []

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, position: int, data: dict) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = self.POSITION_LIMITS["HYDROGEL_PACK"] - position
        sell_room = self.POSITION_LIMITS["HYDROGEL_PACK"] + position

        mid = (best_bid + best_ask) / 2.0
        old_ema = data.get(self.HG_MOM_KEY, mid)
        ema = (1.0 - self.HG_MOM_ALPHA) * old_ema + self.HG_MOM_ALPHA * mid
        data[self.HG_MOM_KEY] = ema

        raw_momentum = mid - ema
        mom_shift = max(-self.HG_MOM_CAP, min(self.HG_MOM_CAP, self.HG_MOM_K * raw_momentum))
        fair = self.HG_FAIR + mom_shift

        buy_take_edge = max(12.0, self.HG_TAKE_EDGE - max(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)
        sell_take_edge = max(12.0, self.HG_TAKE_EDGE + min(0.0, mom_shift) * self.HG_MOM_TAKE_BOOST)

        if best_ask < fair - buy_take_edge and buy_room > 0:
            qty = min(self.HG_TAKE_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_ask), int(qty)))
                position += qty
                buy_room -= qty

        if best_bid > fair + sell_take_edge and sell_room > 0:
            qty = min(self.HG_TAKE_SIZE, sell_room, bid_vol)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", int(best_bid), -int(qty)))
                position -= qty
                sell_room -= qty

        skew = position // self.HG_SKEW_DIV
        bid_px = int(fair - self.HG_POST_EDGE - skew)
        ask_px = int(fair + self.HG_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(self.HG_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", bid_px, int(qty)))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(self.HG_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order("HYDROGEL_PACK", ask_px, -int(qty)))

        return orders

    def _extract_fair(self, data: dict, depth: OrderDepth, mids: Dict[str, float], micros: Dict[str, float], pos: int) -> float:
        p = "VELVETFRUIT_EXTRACT"
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return data.get("last_extract_fair", self.VE_ANCHOR)

        mid = (best_bid + best_ask) / 2.0
        ema_key = f"{p}_ema"
        old_ema = data.get(ema_key, mid)
        ema = (1.0 - self.VE_EMA_ALPHA) * old_ema + self.VE_EMA_ALPHA * mid
        data[ema_key] = ema

        blended = self.VE_ANCHOR_WEIGHT * self.VE_ANCHOR + (1.0 - self.VE_ANCHOR_WEIGHT) * ema

        bid_vol_f = float(bid_vol)
        ask_vol_f = float(abs(ask_vol))
        total = bid_vol_f + ask_vol_f

        if total > 0:
            imbalance = (bid_vol_f - ask_vol_f) / total
            shift = max(-self.VE_IMBALANCE_CAP, min(self.VE_IMBALANCE_CAP, self.VE_IMBALANCE_K * imbalance))
        else:
            shift = 0.0

        fair = blended + shift
        return fair

    def _trade_delta_one(self, product: str, depth: OrderDepth, position: int, limit: int, data: dict, fair: float) -> List[Order]:
        orders: List[Order] = []
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_room = limit - position
        sell_room = limit + position

        if best_ask < fair - self.VE_EDGE and buy_room > 0:
            qty = min(self.VE_SIZE, buy_room, abs(ask_vol))
            if qty > 0:
                orders.append(Order(product, int(best_ask), qty))
                position += qty
                buy_room -= qty

        if best_bid > fair + self.VE_EDGE and sell_room > 0:
            qty = min(self.VE_SIZE, sell_room, bid_vol)
            if qty > 0:
                orders.append(Order(product, int(best_bid), -qty))
                position -= qty
                sell_room -= qty

        skew = position // 34
        bid_px = int(fair - self.VE_POST_EDGE - skew)
        ask_px = int(fair + self.VE_POST_EDGE - skew)

        if buy_room > 0 and bid_px < best_ask:
            qty = min(self.VE_POST_SIZE, buy_room)
            if qty > 0:
                orders.append(Order(product, bid_px, qty))

        if sell_room > 0 and ask_px > best_bid:
            qty = min(self.VE_POST_SIZE, sell_room)
            if qty > 0:
                orders.append(Order(product, ask_px, -qty))

        return orders

    def _voucher_fair(self, product: str, underlying_fair: float, timestamp: int, data: Optional[dict] = None) -> float:
        K = self.VOUCHER_STRIKES[product]
        T = self._tte_years(timestamp)
        sigma = self._sigma_for_product(data or {}, product, underlying_fair, timestamp)
        return self._black_scholes_call(underlying_fair, K, T, sigma, r=0.0)

    def _voucher_delta(self, product: str, underlying_fair: float, timestamp: int, data: Optional[dict] = None) -> float:
        K = self.VOUCHER_STRIKES[product]
        T = self._tte_years(timestamp)
        sigma = self._sigma_for_product(data or {}, product, underlying_fair, timestamp)
        return self._black_scholes_delta(underlying_fair, K, T, sigma, r=0.0)

    def _tte_years(self, timestamp: int) -> float:
        day_progress = self._clip(timestamp / 1_000_000.0, 0.0, 1.0)
        tte_days = max(4.0, 5.0 - day_progress)
        return max(1e-6, tte_days / 365.0)

    def _sigma_for_product(self, data: dict, product: str, underlying_fair: Optional[float] = None, timestamp: Optional[int] = None) -> float:
        if underlying_fair is not None and timestamp is not None and product in self.VOUCHER_STRIKES:
            K = self.VOUCHER_STRIKES[product]
            T = self._tte_years(timestamp)
            sqrt_t = math.sqrt(max(T, 1e-9))
            m = math.log(max(K, 1e-9) / max(underlying_fair, 1e-9)) / sqrt_t
            total_vol = self._smile_total_vol(m)
            residual = data.get("smile_residual", 0.0)
            product_residual = data.get("smile_residual_by_product", {}).get(product, residual)
            total_vol += self._clip(0.70 * residual + 0.30 * product_residual, -0.0025, 0.0025)
            return self._clip(total_vol / sqrt_t, 0.11, 0.42)

        return self._clip(data.get("bs_sigma", 0.24), 0.11, 0.42)

    def _smile_total_vol(self, m: float) -> float:
        x = self._clip(m, -2.2, 2.9)
        v = 0.0009 * x ** 3 - 0.0030 * x ** 2 + 0.0003 * x + 0.0328
        return self._clip(v, 0.0245, 0.0380)

    def _black_scholes_call(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        if S <= 0 or K <= 0:
            return 0.0
        intrinsic = max(0.0, S - K)
        if T <= 1e-8 or sigma <= 1e-8:
            return intrinsic
        vol_sqrt_t = sigma * math.sqrt(T)
        if vol_sqrt_t <= 1e-10:
            return intrinsic
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return S * self._norm_cdf(d1) - K * math.exp(-r * T) * self._norm_cdf(d2)

    def _black_scholes_delta(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
        if S <= 0 or K <= 0:
            return 0.0
        if T <= 1e-8 or sigma <= 1e-8:
            return 1.0 if S > K else 0.0
        vol_sqrt_t = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / max(1e-10, vol_sqrt_t)
        return self._norm_cdf(d1)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _update_voucher_sigma(self, data: dict, order_depths: Dict[str, OrderDepth], underlying_fair: float, timestamp: int) -> None:
        if "smile_residual_by_product" not in data:
            data["smile_residual_by_product"] = {}
        T = self._tte_years(timestamp)
        sqrt_t = math.sqrt(max(T, 1e-9))
        residuals = []

        for product, K in self.VOUCHER_STRIKES.items():
            depth = order_depths.get(product)
            if depth is None:
                continue
            mid = self._mid(depth)
            if mid is None or mid <= 0:
                continue

            intrinsic = max(0.0, underlying_fair - K)
            extrinsic = mid - intrinsic
            m = math.log(max(K, 1e-9) / max(underlying_fair, 1e-9)) / sqrt_t

            if extrinsic < 0.5 or abs(m) > 1.75:
                continue

            iv = self._implied_vol_call(mid, underlying_fair, K, T)
            if iv is None:
                continue

            market_total_vol = self._clip(iv * sqrt_t, 0.020, 0.045)
            model_total_vol = self._smile_total_vol(m)
            residual = self._clip(market_total_vol - model_total_vol, -0.0040, 0.0040)

            old_p = data["smile_residual_by_product"].get(product, data.get("smile_residual", 0.0))
            data["smile_residual_by_product"][product] = self._clip(0.96 * old_p + 0.04 * residual, -0.0030, 0.0030)
            residuals.append(residual)

        if residuals:
            residuals.sort()
            med = residuals[len(residuals) // 2]
            old = data.get("smile_residual", 0.0)
            data["smile_residual"] = self._clip(0.985 * old + 0.015 * med, -0.0025, 0.0025)

        if "bs_sigma" not in data:
            data["bs_sigma"] = 0.24
        if "bs_sigma_by_product" not in data:
            data["bs_sigma_by_product"] = {}

    def _implied_vol_call(self, market_price: float, S: float, K: float, T: float) -> Optional[float]:
        intrinsic = max(0.0, S - K)
        if market_price < intrinsic - 0.2:
            return None

        lo, hi = 0.01, 1.00
        price_lo = self._black_scholes_call(S, K, T, lo, 0.0)
        price_hi = self._black_scholes_call(S, K, T, hi, 0.0)
        if market_price <= price_lo:
            return lo
        if market_price >= price_hi:
            return hi

        for _ in range(24):
            mid = (lo + hi) / 2.0
            price = self._black_scholes_call(S, K, T, mid, 0.0)
            if price < market_price:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    def _trade_voucher(self, product: str, depth: OrderDepth, pos: int, fair: float, underlying_fair: float, delta: float, data: dict) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        K = self.VOUCHER_STRIKES[product]
        spread = best_ask - best_bid
        passive_only_strikes = {"VEV_5300", "VEV_5400", "VEV_5500", "VEV_6000", "VEV_6500"}
        is_passive_only = product in passive_only_strikes

        total_pnl = float(data.get("last_total_pnl", 0.0))
        peak_pnl = float(data.get("peak_pnl", total_pnl))
        risk_scale = 1.0
        lock_only_unwind = False
        if peak_pnl >= 8000.0:
            risk_scale = 0.0
            lock_only_unwind = True
        elif peak_pnl >= 5000.0:
            risk_scale = 0.50

        strike_multiplier = 1.0
        extra_buy_edge = 0.0
        extra_sell_edge = 0.0
        allow_passive_bid = True
        inv_skew_k = 0.018
        hard_inventory_frac = 0.70

        if product == "VEV_5400":
            strike_multiplier = 0.25
            extra_buy_edge = 1.10
            extra_sell_edge = 0.40
        elif product == "VEV_5300":
            strike_multiplier = 0.45
            extra_buy_edge = 0.90
            extra_sell_edge = 1.25
            inv_skew_k = 0.050
            hard_inventory_frac = 0.45
        elif K >= 5500:
            strike_multiplier = 0.35
            extra_buy_edge = 1.50
            extra_sell_edge = 0.75
            allow_passive_bid = False
        elif product in ("VEV_5100", "VEV_5200", "VEV_5300"):
            strike_multiplier = 1.0
        else:
            strike_multiplier = 0.70

        if 0.25 <= delta <= 0.75:
            base_size = 28
            take_edge = 1.4
            quote_edge = 1.0
        elif 0.10 <= delta < 0.25 or 0.75 < delta <= 0.92:
            base_size = 18
            take_edge = 1.2
            quote_edge = 0.9
        elif delta > 0.92:
            base_size = 14
            take_edge = 1.0
            quote_edge = 0.8
        else:
            base_size = 7
            take_edge = 0.8
            quote_edge = 0.8

        if fair <= 1.20:
            base_size = min(base_size, 5)
            take_edge = 0.7

        base_size = max(1, int(round(base_size * strike_multiplier * max(risk_scale, 0.0))))
        take_edge += extra_buy_edge * 0.50

        buy_take_edge = take_edge + 0.35 + extra_buy_edge
        sell_take_edge = max(0.65, take_edge - 0.15 + extra_sell_edge)
        buy_quote_edge = quote_edge + 0.25 + 0.50 * extra_buy_edge
        sell_quote_edge = max(0.55, quote_edge - 0.10 + 0.35 * extra_sell_edge)

        if is_passive_only:
            buy_quote_edge = max(buy_quote_edge, 1.5)
            sell_quote_edge = max(sell_quote_edge, 1.5)
            if K >= 5500:
                base_size = max(15, base_size)

        inv_adj = fair - inv_skew_k * pos
        buy_cap = limit - pos
        sell_cap = limit + pos

        if lock_only_unwind:
            if pos < 0 and buy_cap > 0 and best_ask <= fair + 0.60:
                qty = min(-pos, buy_cap, abs(ask_vol), max(1, int(0.45 * max(1, 28 * strike_multiplier))))
                if qty > 0:
                    orders.append(Order(product, int(best_ask), int(qty)))
                    buy_cap -= qty
            if pos > 0 and sell_cap > 0 and best_bid >= fair - 0.60:
                qty = min(pos, sell_cap, abs(bid_vol), max(1, int(0.45 * max(1, 28 * strike_multiplier))))
                if qty > 0:
                    orders.append(Order(product, int(best_bid), -int(qty)))
                    sell_cap -= qty
        else:
            if not is_passive_only:
                if buy_cap > 0 and best_ask <= fair - buy_take_edge:
                    edge = fair - best_ask
                    qty = min(buy_cap, abs(ask_vol), int(base_size + 2.0 * max(0.0, edge - buy_take_edge)))
                    if qty > 0:
                        orders.append(Order(product, int(best_ask), int(qty)))
                        buy_cap -= qty

                if sell_cap > 0 and best_bid >= fair + sell_take_edge:
                    edge = best_bid - fair
                    qty = min(sell_cap, abs(bid_vol), int(base_size + 2.0 * max(0.0, edge - sell_take_edge)))
                    if qty > 0:
                        orders.append(Order(product, int(best_bid), -int(qty)))
                        sell_cap -= qty

        min_quote_spread = 1 if is_passive_only else 2
        if spread >= min_quote_spread and not (fair <= 1.05 and spread < 3):
            bid_px = min(best_bid + 1, math.floor(inv_adj - buy_quote_edge))
            ask_px = max(best_ask - 1, math.ceil(inv_adj + sell_quote_edge))

            if bid_px < ask_px:
                buy_size = min(buy_cap, self._inventory_scaled_size(base_size, pos, limit, side=1))
                sell_size = min(sell_cap, self._inventory_scaled_size(base_size, pos, limit, side=-1))

                if not allow_passive_bid:
                    buy_size = 0

                if lock_only_unwind:
                    if pos > 0:
                        buy_size = 0
                        sell_size = min(sell_cap, max(1, min(pos, sell_size)))
                    elif pos < 0:
                        sell_size = 0
                        buy_size = min(buy_cap, max(1, min(-pos, buy_size)))
                    else:
                        buy_size = 0
                        sell_size = 0

                if pos > hard_inventory_frac * limit:
                    buy_size = 0
                    sell_size = min(sell_cap, max(sell_size, base_size))
                elif pos < -hard_inventory_frac * limit:
                    sell_size = 0
                    buy_size = min(buy_cap, max(buy_size, base_size))

                fair_guard = 0.15 if is_passive_only else 0.3

                if buy_size > 0 and bid_px <= fair - fair_guard:
                    orders.append(Order(product, int(bid_px), int(buy_size)))
                if sell_size > 0 and ask_px >= fair + fair_guard:
                    orders.append(Order(product, int(ask_px), -int(sell_size)))

        return orders

    def _update_and_estimate_pnl(self, data: dict, state: TradingState, mids: Dict[str, float]) -> float:
        if "cash" not in data:
            data["cash"] = {}
        if "seen_trade_keys" not in data:
            data["seen_trade_keys"] = []

        seen = set(data.get("seen_trade_keys", []))
        new_seen = list(data.get("seen_trade_keys", []))
        own_trades = getattr(state, "own_trades", {}) or {}

        for product, trades in own_trades.items():
            if product not in self.POSITION_LIMITS:
                continue
            for tr in trades:
                ts = getattr(tr, "timestamp", state.timestamp)
                price = getattr(tr, "price", 0)
                qty = getattr(tr, "quantity", 0)
                buyer = str(getattr(tr, "buyer", ""))
                seller = str(getattr(tr, "seller", ""))
                key = f"{product}|{ts}|{price}|{qty}|{buyer}|{seller}"
                if key in seen:
                    continue
                seen.add(key)
                new_seen.append(key)

                if buyer == "SUBMISSION":
                    data["cash"][product] = float(data["cash"].get(product, 0.0)) - float(price) * float(qty)
                elif seller == "SUBMISSION":
                    data["cash"][product] = float(data["cash"].get(product, 0.0)) + float(price) * float(qty)

        if len(new_seen) > 260:
            new_seen = new_seen[-260:]
        data["seen_trade_keys"] = new_seen

        total = 0.0
        for product in self.POSITION_LIMITS:
            if product == "HYDROGEL_PACK":
                continue
            total += float(data["cash"].get(product, 0.0))
            mark = mids.get(product)
            if mark is not None:
                total += int(state.position.get(product, 0)) * float(mark)
        data["last_total_pnl"] = total
        return total

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"ema_fast": {}, "ema_slow": {}, "last_mid": {}, "ret_ema": {}, "bs_sigma": 0.24, "bs_sigma_by_product": {}, "smile_residual": 0.0, "smile_residual_by_product": {}, "cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}
        try:
            data = json.loads(trader_data)
            for k in ("ema_fast", "ema_slow", "last_mid", "ret_ema"):
                if k not in data:
                    data[k] = {}
            if "bs_sigma" not in data:
                data["bs_sigma"] = 0.20
            if "bs_sigma_by_product" not in data:
                data["bs_sigma_by_product"] = {}
            if "smile_residual" not in data:
                data["smile_residual"] = 0.0
            if "smile_residual_by_product" not in data:
                data["smile_residual_by_product"] = {}
            if "cash" not in data:
                data["cash"] = {}
            if "seen_trade_keys" not in data:
                data["seen_trade_keys"] = []
            if "peak_pnl" not in data:
                data["peak_pnl"] = 0.0
            if "last_total_pnl" not in data:
                data["last_total_pnl"] = 0.0
            return data
        except Exception:
            return {"ema_fast": {}, "ema_slow": {}, "last_mid": {}, "ret_ema": {}, "bs_sigma": 0.24, "bs_sigma_by_product": {}, "smile_residual": 0.0, "smile_residual_by_product": {}, "cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}

    def _update_stats(self, data: dict, product: str, mid: float) -> None:
        prev_fast = data["ema_fast"].get(product, mid)
        prev_slow = data["ema_slow"].get(product, mid)
        prev_mid = data["last_mid"].get(product, mid)
        prev_ret = data["ret_ema"].get(product, 0.0)

        data["ema_fast"][product] = 0.18 * mid + 0.82 * prev_fast
        data["ema_slow"][product] = 0.035 * mid + 0.965 * prev_slow
        data["ret_ema"][product] = 0.20 * (mid - prev_mid) + 0.80 * prev_ret
        data["last_mid"][product] = mid

    def _best_bid(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.buy_orders:
            return None, 0
        px = max(depth.buy_orders.keys())
        return px, depth.buy_orders[px]

    def _best_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.sell_orders:
            return None, 0
        px = min(depth.sell_orders.keys())
        return px, depth.sell_orders[px]

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        bid, _ = self._best_bid(depth)
        ask, _ = self._best_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _microprice(self, depth: OrderDepth) -> Optional[float]:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return None
        b = abs(bid_vol)
        a = abs(ask_vol)
        if b + a <= 0:
            return (bid + ask) / 2.0
        return (bid * a + ask * b) / (a + b)

    def _inventory_scaled_size(self, base: int, pos: int, limit: int, side: int) -> int:
        inv = pos / max(1, limit)
        if side == 1:
            factor = 1.0 - 0.75 * max(0.0, inv) + 0.35 * max(0.0, -inv)
        else:
            factor = 1.0 - 0.75 * max(0.0, -inv) + 0.35 * max(0.0, inv)
        return max(1, int(round(base * self._clip(factor, 0.20, 1.35))))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))