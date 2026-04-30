# r3v01hvoptflow.py: vNN counts only inside the `hvoptflow` family (same round can have other r3v01… files with different tags).
# Hydrogel / velvet / VEV MM; tape / flow tilt on quotes.
#
import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Tuple

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300,
    "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
    "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
}
SKIP_PRODUCTS = {"VEV_6000", "VEV_6500"}

PRODUCT_CONFIG: Dict[str, Dict] = {
    "VELVETFRUIT_EXTRACT": {"half_spread": 2, "order_size": 12, "skew": 2.0, "max_fill": 0.65},
    "HYDROGEL_PACK": {"half_spread": 4, "order_size": 10, "skew": 2.0, "max_fill": 0.8},
    "VEV_4000": {"half_spread": 8, "order_size": 6, "skew": 2.5, "max_fill": 0.65},
    "VEV_4500": {"half_spread": 6, "order_size": 6, "skew": 2.5, "max_fill": 0.65},
    "VEV_5000": {"half_spread": 8, "order_size": 6, "skew": 2.0, "max_fill": 0.7},
    "VEV_5100": {"half_spread": 1, "order_size": 10, "skew": 2.0, "max_fill": 0.7},
    "VEV_5200": {"half_spread": 1, "order_size": 8, "skew": 2.0, "max_fill": 0.7},
    "VEV_5300": {"half_spread": 1, "order_size": 5, "skew": 2.0, "max_fill": 0.7},
    "VEV_5400": {"half_spread": 1, "order_size": 8, "skew": 2.5, "max_fill": 0.6},
    "VEV_5500": {"half_spread": 1, "order_size": 6, "skew": 2.5, "max_fill": 0.6},
}

OPTION_DELTA: Dict[str, float] = {
    "VEV_5000": 0.95, "VEV_5100": 0.84, "VEV_5200": 0.63,
    "VEV_5300": 0.38, "VEV_5400": 0.17, "VEV_5500": 0.06,
}

COUNTERPARTY_FOLLOW: set = set()
COUNTERPARTY_FADE:   set = set()
CT_WEIGHT = 1.5
CT_DECAY  = 0.85

VFE_Z_WINDOW       = 800
VFE_Z_BLOCK_SIZE   = 100
VFE_Z_ENTRY        = 1.5
VFE_Z_EMERGENCY    = 5.0
VFE_Z_SHIFT_FACTOR = 4.5
VFE_Z_MAX_SHIFT    = 7

HG_Z_WINDOW        = 5000
HG_Z_BLOCK_SIZE    = 100
HG_Z_ENTRY         = 1.5
HG_Z_EMERGENCY     = 6.0
HG_Z_SHIFT_FACTOR  = 2.5
HG_Z_MAX_SHIFT     = 8

VFE_SEED_BLOCKS = [
    [100, 527462.5, 2782167743.25],
    [100, 526724.0, 2774384168.5],
    [100, 524902.5, 2755229726.25],
    [100, 526126.0, 2768087075.5],
    [100, 526945.5, 2776717674.25],
    [100, 526090.0, 2767708363.0],
    [100, 525442.5, 2760898739.25],
    [100, 526500.0, 2772023762.0],
]

HG_SEED_BLOCKS = [
    [100, 1000049.5, 10000991852.75], [100, 1000749.0, 10014989094.5],
    [100, 1001529.5, 10030616391.75], [100, 1001805.5, 10036145130.75],
    [100, 1001279.5, 10025609061.25], [100, 1001311.0, 10026242103.0],
    [100, 1000405.0, 10008105512.0],  [100, 1000852.5, 10017059271.25],
    [100, 1000321.5, 10006431953.75], [100, 1001367.5, 10027384676.25],
    [100, 1002282.0, 10045694954.0],  [100, 1000001.5, 10000041242.25],
    [100, 1001060.5, 10021229323.75], [100, 1002161.5, 10043280951.75],
    [100, 1000802.0, 10016062615.5],  [100, 998930.0,  9978615501.0],
    [100, 1001199.0, 10024004349.5],  [100, 1000590.5, 10011818165.75],
    [100, 999639.0,  9992784193.5],   [100, 1002227.0, 10044606053.0],
    [100, 1004628.5, 10092787882.25], [100, 1004779.0, 10095813740.0],
    [100, 1007022.5, 10140945608.25], [100, 1004585.0, 10091919244.5],
    [100, 1004939.0, 10099026076.5],  [100, 1006264.5, 10125686628.25],
    [100, 1005681.0, 10113945748.0],  [100, 1006776.5, 10135991083.75],
    [100, 1006479.5, 10130013406.25], [100, 1003944.0, 10079043833.5],
    [100, 1001101.0, 10022033153.5],  [100, 1000306.0, 10006123566.0],
    [100, 1000656.5, 10013137405.75], [100, 1002725.0, 10054585315.0],
    [100, 1002757.5, 10055227707.75], [100, 1005078.0, 10101834338.0],
    [100, 1005450.0, 10109307286.5],  [100, 1002035.5, 10040769144.25],
    [100, 999911.5,  9998235007.75],  [100, 999236.5,  9984741054.25],
    [100, 1001065.5, 10021327181.25], [100, 1002837.0, 10056826094.5],
    [100, 1000458.0, 10009165542.0],  [100, 1000408.5, 10008174518.75],
    [100, 1001448.5, 10028993561.25], [100, 1002065.0, 10041352827.5],
    [100, 1001545.5, 10030952015.25], [100, 999743.5,  9994875293.25],
    [100, 1001780.0, 10035639132.5],  [100, 1002267.5, 10045405205.25],
]

def weighted_mid(depth: OrderDepth) -> Optional[float]:
    if not depth.buy_orders or not depth.sell_orders:
        return None
    bid_wsum = sum(p * v for p, v in depth.buy_orders.items())
    bid_vol  = sum(depth.buy_orders.values())
    ask_wsum = sum(p * abs(v) for p, v in depth.sell_orders.items())
    ask_vol  = sum(abs(v) for v in depth.sell_orders.values())
    if bid_vol == 0 or ask_vol == 0:
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0
    return (bid_wsum / bid_vol + ask_wsum / ask_vol) / 2.0

def _vol_mid(depth: OrderDepth, fallback: float) -> float:
    bb = depth.buy_orders or {}
    sa = depth.sell_orders or {}
    bid_p = max(bb.keys(), key=lambda p: bb[p]) if bb else None
    ask_p = max(sa.keys(), key=lambda p: abs(sa[p])) if sa else None
    if bid_p is None or ask_p is None:
        return fallback
    return (bid_p + ask_p) / 2.0

def _z_update(zs: dict, mid: float, seed_blocks: list, window: int, block_size: int) -> float:
    max_blocks = window // block_size
    if not zs.get('seeded'):
        zs['blocks'] = [list(b) for b in seed_blocks]
        zs['cur']    = [0, 0.0, 0.0]
        zs['seeded'] = True
    cur = zs['cur']
    cur[0] += 1; cur[1] += mid; cur[2] += mid * mid
    if cur[0] >= block_size:
        zs['blocks'].append(cur)
        while len(zs['blocks']) > max_blocks:
            zs['blocks'].pop(0)
        zs['cur'] = [0, 0.0, 0.0]
    blocks = zs['blocks']
    tot_n  = sum(b[0] for b in blocks) + zs['cur'][0]
    tot_s  = sum(b[1] for b in blocks) + zs['cur'][1]
    tot_ss = sum(b[2] for b in blocks) + zs['cur'][2]
    if tot_n < window:
        return 0.0
    mean = tot_s / tot_n
    var  = max(0.0, tot_ss / tot_n - mean * mean)
    if var < 1e-9:
        return 0.0
    return (mid - mean) / var ** 0.5

def _ct_signal(ct: dict, product: str, market_trades, decay: float, weight: float) -> float:
    # Return FV shift in ticks from counterparty order flow. Returns 0.0 when COUNTERPARTY_FOLLOW/FADE are both empty (no...
    if not COUNTERPARTY_FOLLOW and not COUNTERPARTY_FADE:
        return 0.0
    trades = market_trades.get(product, []) if market_trades else []
    net = 0.0
    for t in trades:
        qty = t.quantity
        if t.buyer in COUNTERPARTY_FOLLOW or t.seller in COUNTERPARTY_FADE:
            net += qty
        if t.seller in COUNTERPARTY_FOLLOW or t.buyer in COUNTERPARTY_FADE:
            net -= qty
    flow = ct.get(product, 0.0) * decay + net
    ct[product] = flow
    return max(-weight, min(weight, flow / 100.0 * weight))

def mm_orders(product, depth, running, fv_adj=0.0, unwind_only=False):
    cfg = PRODUCT_CONFIG.get(product)
    if cfg is None:
        return []
    limit = POSITION_LIMITS[product]
    hs = cfg["half_spread"]; sz = cfg["order_size"]
    sk = cfg["skew"];        mf = cfg["max_fill"]
    fv = weighted_mid(depth)
    if fv is None:
        return []
    fv += fv_adj
    cur        = running.get(product, 0)
    fill_ratio = cur / limit
    inv_skew   = fill_ratio * hs * sk
    bid_px = round(fv - hs - inv_skew)
    ask_px = round(fv + hs - inv_skew)
    if ask_px <= bid_px:
        ask_px = bid_px + 1
    adj_sz = max(1, int(sz * (1.0 - 0.5 * abs(fill_ratio))))
    orders: List[Order] = []
    if unwind_only:
        if fill_ratio > 0.05 and cur - adj_sz >= -limit:
            orders.append(Order(product, ask_px, -adj_sz))
        elif fill_ratio < -0.05 and cur + adj_sz <= limit:
            orders.append(Order(product, bid_px, adj_sz))
    else:
        if fill_ratio < mf and cur + adj_sz <= limit:
            orders.append(Order(product, bid_px, adj_sz))
        if fill_ratio > -mf and cur - adj_sz >= -limit:
            orders.append(Order(product, ask_px, -adj_sz))
    return orders

class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            td = {}

        vfe_z_state: dict = td.get('vfe_z', {})
        hg_z_state:  dict = td.get('hg_z',  {})
        ct_state:    dict = td.get('ct',    {})
        result: Dict[str, List[Order]] = {}
        running: Dict[str, int] = {p: state.position.get(p, 0) for p in POSITION_LIMITS}
        market_trades = getattr(state, 'market_trades', None) or {}

        vfe_fv_adj = 0.0
        vfe_unwind = False
        vfe_depth  = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if vfe_depth and vfe_depth.buy_orders and vfe_depth.sell_orders:
            fallback = vfe_z_state.get('old_price', 5250.0)
            mid = _vol_mid(vfe_depth, fallback)
            vfe_z_state['old_price'] = mid
            z = _z_update(vfe_z_state, mid, VFE_SEED_BLOCKS, VFE_Z_WINDOW, VFE_Z_BLOCK_SIZE)
            if abs(z) >= VFE_Z_EMERGENCY:
                vfe_unwind = True
            elif abs(z) > VFE_Z_ENTRY:
                excess = abs(z) - VFE_Z_ENTRY
                shift  = min(excess * VFE_Z_SHIFT_FACTOR, VFE_Z_MAX_SHIFT)
                vfe_fv_adj = -shift if z > 0 else shift

        vfe_fv_adj += _ct_signal(ct_state, "VELVETFRUIT_EXTRACT", market_trades, CT_DECAY, CT_WEIGHT)

        hg_fv_adj = 0.0
        hg_depth  = state.order_depths.get("HYDROGEL_PACK")
        if hg_depth and hg_depth.buy_orders and hg_depth.sell_orders:
            fallback = hg_z_state.get('old_price', 10000.0)
            mid = _vol_mid(hg_depth, fallback)
            hg_z_state['old_price'] = mid
            z = _z_update(hg_z_state, mid, HG_SEED_BLOCKS, HG_Z_WINDOW, HG_Z_BLOCK_SIZE)
            if abs(z) > HG_Z_ENTRY:
                excess = abs(z) - HG_Z_ENTRY
                shift  = min(excess * HG_Z_SHIFT_FACTOR, HG_Z_MAX_SHIFT)
                hg_fv_adj = -shift if z > 0 else shift

        for product, depth in state.order_depths.items():
            if product in SKIP_PRODUCTS or product not in POSITION_LIMITS:
                continue
            if not depth.buy_orders or not depth.sell_orders:
                continue

            if product == "VELVETFRUIT_EXTRACT":
                fv_adj = vfe_fv_adj
                unwind = vfe_unwind
            elif product == "HYDROGEL_PACK":
                fv_adj = hg_fv_adj
                unwind = False
            else:
                fv_adj = OPTION_DELTA.get(product, 0.0) * vfe_fv_adj
                unwind = False

            orders = mm_orders(product, depth, running, fv_adj=fv_adj, unwind_only=unwind)
            limit  = POSITION_LIMITS[product]
            product_orders: List[Order] = []
            for order in orders:
                cur = running[product]
                qty = order.quantity
                if qty > 0 and cur + qty > limit:
                    qty = max(0, limit - cur)
                elif qty < 0 and cur + qty < -limit:
                    qty = min(0, -limit - cur)
                if qty == 0:
                    continue
                product_orders.append(Order(product, order.price, qty))
                running[product] += qty
            if product_orders:
                result[product] = product_orders

        td['vfe_z'] = vfe_z_state
        td['hg_z']  = hg_z_state
        td['ct']    = ct_state
        return result, 0, json.dumps(td, separators=(',', ':'))