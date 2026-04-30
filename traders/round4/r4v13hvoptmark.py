# r4v13hvoptmark.py: vNN counts only inside the `hvoptmark` family (same round can have other r4v01… files with different tags).
# Hydrogel / velvet / VEV MM with explicit Mark-flow overlays.
#
from datamodel import OrderDepth, Order, Trade, TradingState
from typing import Dict, List, Optional, Tuple
import json

class Trader:
    TARGET = "Mark 22"
    BASE_SIZE = 10
    ACTIVE = frozenset({"HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"})
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

    def run(self, state: TradingState):
        data = self._load_data(state.traderData)
        out: Dict[str, List[Order]] = {k: [] for k in state.order_depths}
        for hit in data["pending_hits"]:
            sym = str(hit["symbol"])
            if sym not in state.order_depths or sym not in self.ACTIVE:
                continue
            want = str(hit["side"])
            pos = state.position.get(sym, 0)
            lim = self.POSITION_LIMITS.get(sym, 0)
            qty = min(int(hit["qty"]), self.BASE_SIZE)
            depth = state.order_depths[sym]
            if want == "sell" and pos > -lim:
                bid, _ = self._best_bid(depth)
                if bid is not None:
                    q = min(qty, lim + pos)
                    if q > 0:
                        out[sym].append(Order(sym, int(bid), -q))
            elif want == "buy" and pos < lim:
                ask, _ = self._best_ask(depth)
                if ask is not None:
                    q = min(qty, lim - pos)
                    if q > 0:
                        out[sym].append(Order(sym, int(ask), q))
        nxt: List[Dict] = []
        seen = set(data["seen_keys"])
        sk = list(data["seen_keys"])
        for sym, trs in (state.market_trades or {}).items():
            if sym not in self.ACTIVE:
                continue
            for tr in trs:
                k = self._key(sym, tr)
                if k in seen:
                    continue
                seen.add(k)
                sk.append(k)
                q = abs(int(getattr(tr, "quantity", 0)))
                if q <= 0:
                    continue
                b, s = str(getattr(tr, "buyer", "")), str(getattr(tr, "seller", ""))
                if b == self.TARGET:
                    nxt.append({"symbol": sym, "side": "sell", "qty": min(q, self.BASE_SIZE)})
                if s == self.TARGET:
                    nxt.append({"symbol": sym, "side": "buy", "qty": min(q, self.BASE_SIZE)})
        data["pending_hits"] = nxt
        data["seen_keys"] = sk[-500:]
        return out, 0, json.dumps(data, separators=(",", ":"))

    def _best_bid(self, d: OrderDepth) -> Tuple[Optional[int], int]:
        if not d.buy_orders:
            return None, 0
        p = max(d.buy_orders.keys())
        return p, int(d.buy_orders[p])
    def _best_ask(self, d: OrderDepth) -> Tuple[Optional[int], int]:
        if not d.sell_orders:
            return None, 0
        p = min(d.sell_orders.keys())
        return p, int(d.sell_orders[p])
    def _key(self, sym: str, tr: Trade) -> str:
        return f"{sym}|{getattr(tr,'timestamp',0)}|{getattr(tr,'price',0)}|{getattr(tr,'quantity',0)}|{getattr(tr,'buyer','')}|{getattr(tr,'seller','')}"
    def _load_data(self, s: str) -> dict:
        if not s:
            return {"pending_hits": [], "seen_keys": []}
        try:
            o = json.loads(s)
            o.setdefault("pending_hits", [])
            o.setdefault("seen_keys", [])
            return o
        except Exception:
            return {"pending_hits": [], "seen_keys": []}
