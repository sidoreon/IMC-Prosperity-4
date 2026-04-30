# r3v01hydromark.py: vNN counts only inside the `hydromark` family (same round can have other r3v01… files with different tags).
# Hydrogel-focused; Mark-informed quoting.
#
from datamodel import OrderDepth, Order, TradingState
from typing import Dict, List, Tuple, Optional
import json
import math

class Trader:
    # Round 3 robust v5: volatility-smile voucher model with profit-lock risk control. Design goal:

    POSITION_LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK": 200,
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

        for product, depth in state.order_depths.items():
            if product not in self.POSITION_LIMITS:
                continue

            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product == "HYDROGEL_PACK":
                fair = self._hydrogel_fair(data, depth, mids, micros, pos)
                orders = self._trade_local_mm(
                    product, depth, pos, fair,
                    take_edge=5.0,
                    quote_edge=7.0,
                    base_size=18,
                    min_spread_to_quote=11,
                    inv_skew=0.055,
                    max_take_size=34,
                )

            if orders:
                result[product] = orders
            else:
                result[product] = []

        return result, 0, json.dumps(data, separators=(",", ":"))

    def _hydrogel_fair(self, data: dict, depth: OrderDepth, mids: Dict[str, float], micros: Dict[str, float], pos: int) -> float:
        p = "HYDROGEL_PACK"
        mid = mids.get(p, self._mid(depth))
        micro = micros.get(p, mid)
        fast = data.get("ema_fast", {}).get(p, mid)
        slow = data.get("ema_slow", {}).get(p, mid)
        ret_ema = data.get("ret_ema", {}).get(p, 0.0)

        fair = 0.54 * micro + 0.25 * mid + 0.16 * fast + 0.05 * slow

        trend = fast - slow
        fair += self._clip(-0.22 * trend, -3.0, 3.0)
        fair += self._clip(-0.25 * ret_ema, -2.0, 2.0)
        fair += 2.0 * self._imbalance(depth)
        fair -= 3.8 * (pos / self.POSITION_LIMITS[p])
        return fair

    def _trade_local_mm(self, product: str, depth: OrderDepth, pos: int, fair: float,
                        take_edge: float, quote_edge: float, base_size: int,
                        min_spread_to_quote: int, inv_skew: float, max_take_size: int) -> List[Order]:
        orders: List[Order] = []
        limit = self.POSITION_LIMITS[product]
        best_bid, bid_vol = self._best_bid(depth)
        best_ask, ask_vol = self._best_ask(depth)
        if best_bid is None or best_ask is None:
            return orders

        buy_cap = limit - pos
        sell_cap = limit + pos

        if buy_cap > 0 and best_ask <= fair - take_edge:
            edge = fair - best_ask
            qty = min(buy_cap, abs(ask_vol), max_take_size, int(base_size + 3 * max(0.0, edge - take_edge)))
            if qty > 0:
                orders.append(Order(product, int(best_ask), qty))
                buy_cap -= qty

        if sell_cap > 0 and best_bid >= fair + take_edge:
            edge = best_bid - fair
            qty = min(sell_cap, abs(bid_vol), max_take_size, int(base_size + 3 * max(0.0, edge - take_edge)))
            if qty > 0:
                orders.append(Order(product, int(best_bid), -qty))
                sell_cap -= qty

        spread = best_ask - best_bid
        if spread >= min_spread_to_quote:
            inventory_pressure = inv_skew * pos
            adj = fair - inventory_pressure
            bid_px = min(best_bid + 1, math.floor(adj - quote_edge))
            ask_px = max(best_ask - 1, math.ceil(adj + quote_edge))

            if bid_px < ask_px:
                buy_size = min(buy_cap, self._inventory_scaled_size(base_size, pos, limit, side=1))
                sell_size = min(sell_cap, self._inventory_scaled_size(base_size, pos, limit, side=-1))

                if pos > 0.75 * limit:
                    buy_size = 0
                    sell_size = min(sell_cap, max(sell_size, base_size))
                elif pos < -0.75 * limit:
                    sell_size = 0
                    buy_size = min(buy_cap, max(buy_size, base_size))

                if buy_size > 0:
                    orders.append(Order(product, int(bid_px), int(buy_size)))
                if sell_size > 0:
                    orders.append(Order(product, int(ask_px), -int(sell_size)))

        return orders

    def _update_and_estimate_pnl(self, data: dict, state: TradingState, mids: Dict[str, float]) -> float:
        # Approximate live mark-to-market PnL for risk control only.
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
            total += float(data["cash"].get(product, 0.0))
            mark = mids.get(product)
            if mark is not None:
                total += int(state.position.get(product, 0)) * float(mark)
        data["last_total_pnl"] = total
        return total

    def _load_data(self, trader_data: str) -> dict:
        if not trader_data:
            return {"ema_fast": {}, "ema_slow": {}, "last_mid": {}, "ret_ema": {}, "cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}
        try:
            data = json.loads(trader_data)
            for k in ("ema_fast", "ema_slow", "last_mid", "ret_ema"):
                if k not in data:
                    data[k] = {}
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
            return {"ema_fast": {}, "ema_slow": {}, "last_mid": {}, "ret_ema": {}, "cash": {}, "seen_trade_keys": [], "peak_pnl": 0.0, "last_total_pnl": 0.0}

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

    def _imbalance(self, depth: OrderDepth) -> float:
        bid, bid_vol = self._best_bid(depth)
        ask, ask_vol = self._best_ask(depth)
        if bid is None or ask is None:
            return 0.0
        b = abs(bid_vol)
        a = abs(ask_vol)
        if b + a <= 0:
            return 0.0
        return self._clip((b - a) / (b + a), -1.0, 1.0)

    def _inventory_scaled_size(self, base: int, pos: int, limit: int, side: int) -> int:
        inv = pos / max(1, limit)
        if side == 1:

            factor = 1.0 - 0.75 * max(0.0, inv) + 0.35 * max(0.0, -inv)
        else:

            factor = 1.0 - 0.75 * max(0.0, -inv) + 0.35 * max(0.0, inv)
        return max(1, int(round(base * self._clip(factor, 0.20, 1.35))))

    def _clip(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))