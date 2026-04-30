# r3v34hvoptvol.py: vNN counts only inside the `hvoptvol` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; vol- and ladder-aware sizing.
#
from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math

def _mid(od: Optional[OrderDepth]) -> Optional[float]:
    if od is None or not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0

HYDROGEL = "HYDROGEL_PACK"
VE = "VELVETFRUIT_EXTRACT"
LIQUID_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEV: Dict[int, str] = {K: f"VEV_{K}" for K in ALL_STRIKES}

HYDROGEL_LIMIT = 200
VE_LIMIT = 200
VOUCHER_LIMIT = 300

class Trader:

    RESIDUAL_THRESHOLD = 0.7
    MAX_POS_BASE = 30
    SMOOTHING_ALPHA = 0.3
    HEDGE_DEAD_ZONE = 15
    HEDGE_COOLDOWN = 100
    MM_BAND_HALF_WIDTH = 30
    SKEW_K = 0.05

    def run(self, state: TradingState):

        try:
            data: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        hydrogel_ema: float = float(data.get("hydrogel_ema", 10000.0))
        last_hedge_ts: int = int(data.get("last_hedge_timestamp", 0))

        ts = state.timestamp
        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}

        if HYDROGEL in state.order_depths:
            hyd_orders, hydrogel_ema = self._hydrogel(
                state.order_depths[HYDROGEL],
                int(state.position.get(HYDROGEL, 0)),
                hydrogel_ema,
            )
            result[HYDROGEL] = hyd_orders

        target_hedge = 0

        if VE in state.order_depths:
            ve_orders, last_hedge_ts = self._ve_mm(
                state.order_depths[VE],
                int(state.position.get(VE, 0)),
                target_hedge,
                ts,
                last_hedge_ts,
            )
            result[VE] = ve_orders

        trader_data = json.dumps({
            "hydrogel_ema": hydrogel_ema,
            "last_hedge_timestamp": last_hedge_ts,
        })

        return result, 0, trader_data

    def _hydrogel(
        self, od: OrderDepth, pos: int, ema: float
    ) -> Tuple[List[Order], float]:
        if not od.buy_orders or not od.sell_orders:
            return [], ema

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = (best_bid + best_ask) / 2.0

        ema = 0.01 * mid + 0.99 * ema
        fv = 0.9 * 10000.0 + 0.1 * ema

        skew = pos * self.SKEW_K
        bid_px = round(best_bid + 1 - skew)
        ask_px = round(best_ask - 1 - skew)

        size = max(0, int(50 * (1.0 - abs(pos) / HYDROGEL_LIMIT)))
        bid_size = min(size, max(0, HYDROGEL_LIMIT - pos))
        ask_size = min(size, max(0, HYDROGEL_LIMIT + pos))

        orders: List[Order] = []
        if bid_px < ask_px:
            if bid_size > 0:
                orders.append(Order(HYDROGEL, bid_px, bid_size))
            if ask_size > 0:
                orders.append(Order(HYDROGEL, ask_px, -ask_size))
        return orders, ema

    def _vouchers(
        self,
        state: TradingState,
        S: float,
        T_years: float,
        TTE_days: float,
        prev_targets: Dict[int, float],
    ) -> Tuple[Dict[str, List[Order]], Dict[int, float], int, Dict[int, int]]:
        sqrtT = math.sqrt(T_years)

        ivs: Dict[int, float] = {}
        ms: Dict[int, float] = {}
        for K in LIQUID_STRIKES:
            od = state.order_depths.get(VEV[K])
            mid = _mid(od)
            if mid is None:
                continue
            iv = _implied_vol(mid, S, K, T_years)
            if iv is None:
                continue
            ivs[K] = iv
            ms[K] = math.log(K / S) / sqrtT

        fitted: Dict[int, float] = {}
        if len(ivs) >= 4:
            m_arr = [ms[K] for K in ivs]
            v_arr = [ivs[K] for K in ivs]
            a, b, c = np.polyfit(m_arr, v_arr, 2)
            for K in ivs:
                m = ms[K]
                fitted[K] = a * m * m + b * m + c

        residuals: Dict[int, float] = {}
        residual_std = 1.0
        if fitted:
            for K in fitted:
                residuals[K] = ivs[K] - fitted[K]
            vals = list(residuals.values())
            if len(vals) >= 2:
                mean_r = sum(vals) / len(vals)
                var = sum((r - mean_r) ** 2 for r in vals) / len(vals)
                residual_std = math.sqrt(var) if var > 0 else 1.0

        MAX_POS = int(self.MAX_POS_BASE * math.sqrt(max(TTE_days, 0.5) / 5.0))
        new_targets = dict(prev_targets)
        pending: Dict[int, int] = {}
        vev_orders: Dict[str, List[Order]] = {}

        for K in LIQUID_STRIKES:
            product = VEV[K]
            od = state.order_depths.get(product)
            pos = int(state.position.get(product, 0))
            r = residuals.get(K)

            if r is not None:
                ratio = abs(r) / residual_std
                if r < -self.RESIDUAL_THRESHOLD * residual_std:
                    comp = float(min(MAX_POS, ratio * 20))
                elif r > self.RESIDUAL_THRESHOLD * residual_std:
                    comp = float(-min(MAX_POS, ratio * 20))
                else:
                    comp = 0.0
            else:
                comp = 0.0

            prev = prev_targets.get(K, 0.0)
            smooth = self.SMOOTHING_ALPHA * comp + (1.0 - self.SMOOTHING_ALPHA) * prev
            new_targets[K] = smooth

            order_qty = 0
            if od and od.buy_orders and od.sell_orders:
                mid = _mid(od)
                gap = int(round(smooth)) - pos
                buy_cap = VOUCHER_LIMIT - pos
                sell_cap = VOUCHER_LIMIT + pos
                if gap > 0 and buy_cap > 0:
                    order_qty = min(gap, buy_cap)
                    vev_orders[product] = [Order(product, round(mid + 1), order_qty)]
                elif gap < 0 and sell_cap > 0:
                    order_qty = -min(-gap, sell_cap)
                    vev_orders[product] = [Order(product, round(mid - 1), order_qty)]
                else:
                    vev_orders[product] = []
            else:
                vev_orders[product] = []

            pending[K] = order_qty

        portfolio_delta = 0.0
        for K in LIQUID_STRIKES:
            proj_pos = int(state.position.get(VEV[K], 0)) + pending.get(K, 0)
            fiv = fitted.get(K)
            if proj_pos == 0 or fiv is None or fiv <= 0:
                continue
            portfolio_delta += proj_pos * _bs_delta(S, K, T_years, fiv)

        target_hedge = int(-round(portfolio_delta))
        return vev_orders, new_targets, target_hedge, pending

    def _ve_mm(
        self,
        od: OrderDepth,
        pos: int,
        target_hedge: int,
        ts: int,
        last_hedge_ts: int,
    ) -> Tuple[List[Order], int]:
        if not od.buy_orders or not od.sell_orders:
            return [], last_hedge_ts

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        gap = target_hedge - pos
        hedge_qty = 0
        new_last_ts = last_hedge_ts

        if abs(gap) > self.HEDGE_DEAD_ZONE and ts - last_hedge_ts > self.HEDGE_COOLDOWN:
            remaining = abs(gap)
            filled = 0
            if gap > 0:
                buy_cap = max(0, VE_LIMIT - pos)
                for ask_px in sorted(od.sell_orders.keys())[:3]:
                    avail = -od.sell_orders[ask_px]
                    take = min(remaining, avail, buy_cap - filled)
                    if take <= 0:
                        break
                    filled += take
                    remaining -= take
                hedge_qty = filled
            else:
                sell_cap = max(0, VE_LIMIT + pos)
                for bid_px in sorted(od.buy_orders.keys(), reverse=True)[:3]:
                    avail = od.buy_orders[bid_px]
                    take = min(remaining, avail, sell_cap - filled)
                    if take <= 0:
                        break
                    filled += take
                    remaining -= take
                hedge_qty = -filled

            if hedge_qty != 0:
                new_last_ts = ts

        proj_ve = pos + hedge_qty

        band_lo = max(-VE_LIMIT, target_hedge - self.MM_BAND_HALF_WIDTH)
        band_hi = min(VE_LIMIT, target_hedge + self.MM_BAND_HALF_WIDTH)
        mm_skew = (proj_ve - target_hedge) * self.SKEW_K

        mm_buy_cap = max(0, VE_LIMIT - proj_ve)
        mm_sell_cap = max(0, VE_LIMIT + proj_ve)

        bid_px = round(best_bid + 1 - mm_skew)
        ask_px = round(best_ask - 1 - mm_skew)

        mm_orders: List[Order] = []
        if band_lo <= proj_ve <= band_hi:
            if bid_px < ask_px:
                if mm_buy_cap > 0:
                    mm_orders.append(Order(VE, bid_px, min(20, mm_buy_cap)))
                if mm_sell_cap > 0:
                    mm_orders.append(Order(VE, ask_px, -min(20, mm_sell_cap)))
        elif proj_ve > band_hi:
            if mm_sell_cap > 0:
                mm_orders.append(Order(VE, ask_px, -min(20, mm_sell_cap)))
        else:
            if mm_buy_cap > 0:
                mm_orders.append(Order(VE, bid_px, min(20, mm_buy_cap)))

        orders: List[Order] = []
        if hedge_qty != 0:
            remaining = abs(hedge_qty)
            if hedge_qty > 0:
                for ask_px_l in sorted(od.sell_orders.keys())[:3]:
                    avail = -od.sell_orders[ask_px_l]
                    take = min(remaining, avail)
                    if take <= 0:
                        break
                    orders.append(Order(VE, ask_px_l, take))
                    remaining -= take
            else:
                for bid_px_l in sorted(od.buy_orders.keys(), reverse=True)[:3]:
                    avail = od.buy_orders[bid_px_l]
                    take = min(remaining, avail)
                    if take <= 0:
                        break
                    orders.append(Order(VE, bid_px_l, -take))
                    remaining -= take

        orders.extend(mm_orders)
        return orders, new_last_ts

if __name__ == "__main__":
    import csv
    from collections import defaultdict
    from pathlib import Path

    def _repo_root() -> Path:
        p = Path(__file__).resolve()
        for cur in p.parents:
            if cur.name == "traders":
                return cur.parent
        return p.parent.parent

    DATASET_DIR = _repo_root() / "datasets" / "round3"
    DAYS = [0, 1, 2]

    tick_map: dict = defaultdict(list)
    for day in DAYS:
        csv_path = DATASET_DIR / f"prices_round_3_day_{day}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tick_map[(day, int(row["timestamp"]))].append(row)

    trader = Trader()
    positions: Dict[str, int] = {}
    cash: Dict[str, float] = {}
    last_mid: Dict[str, float] = {}
    trader_data_str = ""

    for (day, csv_ts) in sorted(tick_map.keys()):
        rows = tick_map[(day, csv_ts)]
        sim_ts = day * 1_000_000 + csv_ts

        order_depths: Dict[str, OrderDepth] = {}
        for row in rows:
            product = row["product"]
            od = OrderDepth()
            for i in range(1, 4):
                bp = row.get(f"bid_price_{i}", "").strip()
                bv = row.get(f"bid_volume_{i}", "").strip()
                if bp and bv:
                    try:
                        od.buy_orders[int(float(bp))] = int(float(bv))
                    except ValueError:
                        pass
                ap = row.get(f"ask_price_{i}", "").strip()
                av = row.get(f"ask_volume_{i}", "").strip()
                if ap and av:
                    try:
                        od.sell_orders[int(float(ap))] = -int(float(av))
                    except ValueError:
                        pass
            order_depths[product] = od
            try:
                last_mid[product] = float(row["mid_price"])
            except (ValueError, KeyError):
                pass

        state = TradingState(
            traderData=trader_data_str,
            timestamp=sim_ts,
            listings={},
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=dict(positions),
            observations={},
        )

        orders_by_product, _, trader_data_str = trader.run(state)

        for product, orders in orders_by_product.items():
            od = order_depths.get(product)
            if od is None:
                continue
            pos = positions.get(product, 0)
            c = cash.get(product, 0.0)

            for order in orders:
                if order.quantity > 0:
                    remaining = order.quantity
                    for ask_px in sorted(od.sell_orders.keys()):
                        if ask_px > order.price or remaining <= 0:
                            break
                        avail = -od.sell_orders[ask_px]
                        fill = min(remaining, avail)
                        pos += fill
                        c -= fill * ask_px
                        remaining -= fill
                else:
                    remaining = abs(order.quantity)
                    for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                        if bid_px < order.price or remaining <= 0:
                            break
                        avail = od.buy_orders[bid_px]
                        fill = min(remaining, avail)
                        pos -= fill
                        c += fill * bid_px
                        remaining -= fill

            positions[product] = pos
            cash[product] = c

    print("\n=== PnL by product ===")
    total = 0.0
    for product in sorted(cash.keys()):
        mid = last_mid.get(product, 0.0)
        pos = positions.get(product, 0)
        pnl = cash[product] + pos * mid
        print(f"  {product:<35}  PnL = {pnl:>12.2f}  (pos={pos:>5}, cash={cash[product]:>12.2f})")
        total += pnl
    print(f"\n  {'TOTAL':<35}  PnL = {total:>12.2f}")