# r3v10hvoptinv.py: vNN counts only inside the `hvoptinv` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; inventory skew and reduce logic.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List, Optional, Tuple
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
    STRIKE_POSITION_CAP: Dict[str, int] = {
        "VEV_4000": 140,
        "VEV_4500": 160,
        "VEV_5000": 110,
        "VEV_5100": 100,
        "VEV_5200": 180,
        "VEV_5300": 180,
        "VEV_5400": 90,
        "VEV_5500": 70,
        "VEV_6000": 0,
        "VEV_6500": 0,
    }

    HG_ANCHOR = 9991.0
    HG_EMA_ALPHA = 0.12
    HG_EDGE = 26.0
    HG_POST_EDGE = 8.0
    HG_TAKE_SIZE = 20
    HG_POST_SIZE = 20

    VF_ANCHOR = 5262.0
    VF_ANCHOR_W = 0.55
    VF_EMA_ALPHA = 0.08
    VF_OFI_K = 1.2
    VF_OFI_CAP = 2.0
    VF_EDGE = 7
    VF_POST_EDGE = 1
    VF_TAKE_SIZE = 20
    VF_POST_SIZE = 20

    INV_SKEW_MAX = 4.0
    INV_SCALE = 0.6
    INV_FORCE_TH = 0.7
    INV_EMERGENCY_TH = 0.9
    INV_FORCE_SIZE = 25

    TOX_ALPHA = 0.2
    TOX_HIGH = 2.0
    TOX_EXTREME = 3.0
    VOL_ALPHA = 0.25
    K1 = 8.0
    K2 = 1.2

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        result: Dict[str, List[Order]] = {}

        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue
            mids[product] = mid
            self._update_stats(data, product, mid, depth)

        extract_fair = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            depth = state.order_depths["VELVETFRUIT_EXTRACT"]
            extract_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            extract_fair = self._extract_fair(data, depth, extract_pos)
            data["last_extract_fair"] = extract_fair
        else:
            extract_fair = data.get("last_extract_fair")

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue
            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product == "HYDROGEL_PACK":
                orders = self._trade_hydrogel(depth, pos, data)
            elif product == "VELVETFRUIT_EXTRACT":
                fair = extract_fair if extract_fair is not None else self._mid(depth)
                if fair is not None:
                    orders = self._trade_extract(depth, pos, fair, data)
            elif product in self.VOUCHER_STRIKES and extract_fair is not None:
                fair = self._voucher_fair(product, extract_fair, state.timestamp, data)
                delta = self._voucher_delta(product, extract_fair, state.timestamp, data)
                orders = self._trade_voucher(product, depth, pos, fair, delta)

            result[product] = orders

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _trade_hydrogel(self, depth: OrderDepth, position: int, data: dict) -> List[Order]:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return []

        mid = (bid + ask) / 2.0
        old = float(data["hg_ema_mid"].get("HYDROGEL_PACK", mid))
        ema = (1.0 - self.HG_EMA_ALPHA) * old + self.HG_EMA_ALPHA * mid
        data["hg_ema_mid"]["HYDROGEL_PACK"] = ema
        fair = self.HG_ANCHOR + self._clip(mid - ema, -6.0, 6.0)

        vol, spread_ratio, toxicity = self._micro_state("HYDROGEL_PACK", depth, data)
        edge_mult, size_mult = self._edge_size_multipliers(vol, spread_ratio)
        if toxicity >= self.TOX_EXTREME:
            return self._force_inventory_only("HYDROGEL_PACK", depth, position)

        edge = max(10.0, self.HG_EDGE * edge_mult)
        post_edge = max(4.0, self.HG_POST_EDGE * edge_mult)
        take_size = max(1, int(round(self.HG_TAKE_SIZE * size_mult)))
        post_size = max(1, int(round(self.HG_POST_SIZE * size_mult)))

        out: List[Order] = []
        limit = self.POSITION_LIMITS["HYDROGEL_PACK"]
        buy_room = limit - position
        sell_room = limit + position

        if toxicity < self.TOX_HIGH and ask <= fair - edge and buy_room > 0:
            q = min(take_size, buy_room, abs(ask_vol))
            if q > 0:
                out.append(Order("HYDROGEL_PACK", int(ask), int(q)))
                position += q
                buy_room -= q
        if toxicity < self.TOX_HIGH and bid >= fair + edge and sell_room > 0:
            q = min(take_size, sell_room, abs(bid_vol))
            if q > 0:
                out.append(Order("HYDROGEL_PACK", int(bid), -int(q)))
                position -= q
                sell_room -= q

        skew = self._inventory_skew(position, limit)
        bid_px = int(round(fair - post_edge - skew))
        ask_px = int(round(fair + post_edge - skew))
        if buy_room > 0 and bid_px < ask:
            out.append(Order("HYDROGEL_PACK", bid_px, min(post_size, buy_room)))
        if sell_room > 0 and ask_px > bid:
            out.append(Order("HYDROGEL_PACK", ask_px, -min(post_size, sell_room)))

        return self._inventory_overlay("HYDROGEL_PACK", depth, position, out)

    def _trade_extract(self, depth: OrderDepth, position: int, fair: float, data: dict) -> List[Order]:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return []

        vol, spread_ratio, toxicity = self._micro_state("VELVETFRUIT_EXTRACT", depth, data)
        edge_mult, size_mult = self._edge_size_multipliers(vol, spread_ratio)
        if toxicity >= self.TOX_EXTREME:
            return self._force_inventory_only("VELVETFRUIT_EXTRACT", depth, position)

        edge = max(3, int(round(self.VF_EDGE * edge_mult * (1.25 if toxicity >= self.TOX_HIGH else 1.0))))
        post_edge = max(1, int(round(self.VF_POST_EDGE * edge_mult * (1.25 if toxicity >= self.TOX_HIGH else 1.0))))
        take_size = max(1, int(round(self.VF_TAKE_SIZE * size_mult)))
        post_size = max(1, int(round(self.VF_POST_SIZE * size_mult)))

        out: List[Order] = []
        limit = self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        buy_room = limit - position
        sell_room = limit + position

        if ask <= fair - edge and buy_room > 0:
            q = min(take_size, buy_room, abs(ask_vol))
            if q > 0:
                out.append(Order("VELVETFRUIT_EXTRACT", int(ask), int(q)))
                position += q
                buy_room -= q
        if bid >= fair + edge and sell_room > 0:
            q = min(take_size, sell_room, abs(bid_vol))
            if q > 0:
                out.append(Order("VELVETFRUIT_EXTRACT", int(bid), -int(q)))
                position -= q
                sell_room -= q

        skew = self._inventory_skew(position, limit)
        bid_px = int(round(fair - post_edge - skew))
        ask_px = int(round(fair + post_edge - skew))
        if buy_room > 0 and bid_px < ask:
            out.append(Order("VELVETFRUIT_EXTRACT", bid_px, min(post_size, buy_room)))
        if sell_room > 0 and ask_px > bid:
            out.append(Order("VELVETFRUIT_EXTRACT", ask_px, -min(post_size, sell_room)))

        return self._inventory_overlay("VELVETFRUIT_EXTRACT", depth, position, out)

    def _trade_voucher(self, product: str, depth: OrderDepth, pos: int, fair: float, delta: float) -> List[Order]:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return []

        strike = self.VOUCHER_STRIKES[product]
        if strike >= 6000:
            return []

        if strike in (5200, 5300):
            base_size = 22
            take_edge = 1.2
            post_edge = 0.9
        elif strike in (5000, 5100):

            base_size = 10
            take_edge = 1.9
            post_edge = 1.3
        elif strike >= 5500:
            base_size = 6
            take_edge = 1.8
            post_edge = 1.6
        else:
            base_size = 12
            take_edge = 1.4
            post_edge = 1.2

        limit = min(self.POSITION_LIMITS[product], self.STRIKE_POSITION_CAP.get(product, self.POSITION_LIMITS[product]))
        buy_room = limit - pos
        sell_room = limit + pos
        out: List[Order] = []
        spread = ask - bid

        can_take = not (strike in (5000, 5100) and spread <= 2)

        if can_take and ask <= fair - take_edge and buy_room > 0:
            q = min(base_size, buy_room, abs(ask_vol))
            if q > 0:
                out.append(Order(product, int(ask), int(q)))
                pos += q
                buy_room -= q
        if can_take and bid >= fair + take_edge and sell_room > 0:
            q = min(base_size, sell_room, abs(bid_vol))
            if q > 0:
                out.append(Order(product, int(bid), -int(q)))
                pos -= q
                sell_room -= q

        skew = self._inventory_skew(pos, limit)

        bid_px = min(int(bid), int(math.floor(fair - post_edge - skew)))
        ask_px = max(int(ask), int(math.ceil(fair + post_edge - skew)))
        if bid_px < ask_px:
            if buy_room > 0 and bid_px < ask:
                out.append(Order(product, bid_px, min(base_size, buy_room)))
            if sell_room > 0 and ask_px > bid:
                out.append(Order(product, ask_px, -min(base_size, sell_room)))

        pos_norm = abs(pos) / max(1.0, float(limit))
        if pos_norm > 0.45:
            if pos > 0:
                q = min(int(abs(pos) * 0.30), abs(bid_vol), 20)
                if q > 0:
                    out.append(Order(product, int(bid), -int(q)))
            elif pos < 0:
                q = min(int(abs(pos) * 0.30), abs(ask_vol), 20)
                if q > 0:
                    out.append(Order(product, int(ask), int(q)))
        return out

    def _extract_fair(self, data: dict, depth: OrderDepth, pos: int) -> float:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return float(data.get("last_extract_fair", self.VF_ANCHOR))

        mid = (bid + ask) / 2.0
        old = float(data["vf_ema_mid"].get("VELVETFRUIT_EXTRACT", mid))
        ema = (1.0 - self.VF_EMA_ALPHA) * old + self.VF_EMA_ALPHA * mid
        data["vf_ema_mid"]["VELVETFRUIT_EXTRACT"] = ema

        b = float(abs(bid_vol))
        a = float(abs(ask_vol))
        ofi = (b - a) / max(1e-9, a + b)
        ofi_shift = self._clip(self.VF_OFI_K * ofi, -self.VF_OFI_CAP, self.VF_OFI_CAP)

        fair = self.VF_ANCHOR_W * self.VF_ANCHOR + (1.0 - self.VF_ANCHOR_W) * ema + ofi_shift
        fair -= 0.8 * self._inventory_skew(pos, self.POSITION_LIMITS["VELVETFRUIT_EXTRACT"])
        return fair

    def _voucher_fair(self, product: str, underlying_fair: float, timestamp: int, data: dict) -> float:
        k = self.VOUCHER_STRIKES[product]
        t = self._tte_years(timestamp)
        sigma = self._sigma(product, underlying_fair, t)
        return self._bs_call(underlying_fair, k, t, sigma)

    def _voucher_delta(self, product: str, underlying_fair: float, timestamp: int, data: dict) -> float:
        k = self.VOUCHER_STRIKES[product]
        t = self._tte_years(timestamp)
        sigma = self._sigma(product, underlying_fair, t)
        return self._bs_delta(underlying_fair, k, t, sigma)

    def _sigma(self, product: str, s: float, t: float) -> float:
        k = self.VOUCHER_STRIKES[product]
        m = math.log(max(k, 1e-9) / max(s, 1e-9)) / max(math.sqrt(max(t, 1e-9)), 1e-9)

        base = 0.215 + 0.03 * abs(m)
        return self._clip(base, 0.12, 0.55)

    def _micro_state(self, product: str, depth: OrderDepth, data: dict) -> Tuple[float, float, float]:
        mid = self._mid(depth)
        bid, _ = self._best_bid(depth)
        ask, _ = self._best_ask(depth)
        if mid is None or bid is None or ask is None:
            return 0.0, 1.0, 1.0

        prev_mid = float(data["last_mid"].get(product, mid))
        abs_ret = abs(mid - prev_mid)
        old_vol = float(data["vol_ema_abs_ret"].get(product, abs_ret))
        vol = self.VOL_ALPHA * abs_ret + (1.0 - self.VOL_ALPHA) * old_vol
        data["vol_ema_abs_ret"][product] = vol

        spread = float(ask - bid)
        old_spread = float(data["spread_ema"].get(product, spread))
        avg_spread = 0.08 * spread + 0.92 * old_spread
        data["spread_ema"][product] = avg_spread
        spread_ratio = spread / max(1e-9, avg_spread)

        old_jump = float(data["tox_jump_ema"].get(product, max(1e-6, abs_ret)))
        avg_jump = self.TOX_ALPHA * abs_ret + (1.0 - self.TOX_ALPHA) * old_jump
        data["tox_jump_ema"][product] = max(1e-6, avg_jump)
        toxicity = abs_ret / max(1e-6, avg_jump)
        return vol, spread_ratio, toxicity

    def _edge_size_multipliers(self, vol: float, spread_ratio: float) -> Tuple[float, float]:
        adjusted = 2.0 + self.K1 * vol + self.K2 * spread_ratio
        if adjusted < 3.0:
            return 0.95, 1.10
        if adjusted < 5.0:
            return 1.05, 1.0
        return 1.2, 0.8

    def _inventory_skew(self, pos: int, limit: int) -> float:
        pos_norm = pos / max(1.0, float(limit))
        return self.INV_SKEW_MAX * math.tanh(pos_norm / self.INV_SCALE)

    def _force_inventory_only(self, product: str, depth: OrderDepth, pos: int) -> List[Order]:
        limit = self.POSITION_LIMITS[product]
        pos_norm = abs(pos) / max(1.0, float(limit))
        if pos_norm <= self.INV_FORCE_TH:
            return []
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return []
        if pos > 0:
            q = min(abs(pos), self.INV_FORCE_SIZE, abs(bid_vol))
            return [Order(product, int(bid), -int(q))] if q > 0 else []
        if pos < 0:
            q = min(abs(pos), self.INV_FORCE_SIZE, abs(ask_vol))
            return [Order(product, int(ask), int(q))] if q > 0 else []
        return []

    def _inventory_overlay(self, product: str, depth: OrderDepth, pos: int, orders: List[Order]) -> List[Order]:
        limit = self.POSITION_LIMITS[product]
        pos_norm = abs(pos) / max(1.0, float(limit))
        if pos_norm <= self.INV_FORCE_TH:
            return orders
        forced = self._force_inventory_only(product, depth, pos)
        if pos_norm > self.INV_EMERGENCY_TH:
            return forced
        return orders + forced

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {
                "last_mid": {},
                "vol_ema_abs_ret": {},
                "spread_ema": {},
                "tox_jump_ema": {},
                "hg_ema_mid": {},
                "vf_ema_mid": {},
            }
        try:
            data = json.loads(trader_data)
            data.setdefault("last_mid", {})
            data.setdefault("vol_ema_abs_ret", {})
            data.setdefault("spread_ema", {})
            data.setdefault("tox_jump_ema", {})
            data.setdefault("hg_ema_mid", {})
            data.setdefault("vf_ema_mid", {})
            return data
        except Exception:
            return {
                "last_mid": {},
                "vol_ema_abs_ret": {},
                "spread_ema": {},
                "tox_jump_ema": {},
                "hg_ema_mid": {},
                "vf_ema_mid": {},
            }

    def _update_stats(self, data: dict, product: str, mid: float, depth: OrderDepth) -> None:
        data["last_mid"][product] = mid
        _ = depth

    def _best_bid(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.buy_orders:
            return None, 0
        p = max(depth.buy_orders.keys())
        return p, depth.buy_orders[p]

    def _best_ask(self, depth: OrderDepth) -> Tuple[Optional[int], int]:
        if not depth.sell_orders:
            return None, 0
        p = min(depth.sell_orders.keys())
        return p, depth.sell_orders[p]

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        bid, _ = self._best_bid(depth)
        ask, _ = self._best_ask(depth)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def _tte_years(self, timestamp: int) -> float:
        day_progress = self._clip(timestamp / 1_000_000.0, 0.0, 1.0)
        tte_days = max(4.0, 5.0 - day_progress)
        return max(1e-6, tte_days / 365.0)

    def _bs_call(self, s: float, k: float, t: float, sigma: float, r: float = 0.0) -> float:
        if s <= 0 or k <= 0:
            return 0.0
        intrinsic = max(0.0, s - k)
        if t <= 1e-8 or sigma <= 1e-8:
            return intrinsic
        vs = sigma * math.sqrt(t)
        if vs <= 1e-10:
            return intrinsic
        d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / vs
        d2 = d1 - vs
        return s * self._norm_cdf(d1) - k * math.exp(-r * t) * self._norm_cdf(d2)

    def _bs_delta(self, s: float, k: float, t: float, sigma: float, r: float = 0.0) -> float:
        if s <= 0 or k <= 0:
            return 0.0
        if t <= 1e-8 or sigma <= 1e-8:
            return 1.0 if s > k else 0.0
        vs = sigma * math.sqrt(t)
        d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / max(1e-10, vs)
        return self._norm_cdf(d1)

    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))
