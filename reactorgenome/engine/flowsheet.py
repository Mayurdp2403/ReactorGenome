"""Steady-state solver for a network of reactors drawn on the design board.

Blocks
    feed     Q (L/min), CA, CB (mol/L), T (°C)
    cstr     V (L), T (°C). Jacket on: Ta (°C), UA (J/min/K); T is then the
             result of the energy balance, not a set point.
    pfr      V (L), T (°C), plug flow
    bed      V (L), T (°C), Pe (axial dispersion)
    split    f = fraction sent to the side outlet (recycle or bypass)
    product  collects whatever arrives

Several pipes may arrive at one block; they mix before it. Each outlet port
goes to exactly one place. Flows come from one linear system; compositions
from repeated passes through the network (accelerated with Wegstein's method)
until nothing changes. A loop that has no way out is reported, not solved.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from .pfr import _plug, dispersion_profile
from .rate import NU, Kinetics, rate
from .units import EngineError, c_to_k, k_to_c

UNIT_KINDS = ("cstr", "pfr", "bed")
LABEL = {"feed": "Feed", "cstr": "CSTR", "pfr": "PFR", "bed": "Packed bed", "split": "Splitter", "product": "Product"}
PORTS = {"feed": ["out"], "cstr": ["out"], "pfr": ["out"], "bed": ["out"], "split": ["out", "side"], "product": []}


def _f(node, key, default):
    try:
        v = float(node.get(key, default))
        return v if np.isfinite(v) else float(default)
    except (TypeError, ValueError):
        return float(default)


def names(nodes):
    count, out = {}, {}
    for n in nodes:
        count[n["kind"]] = count.get(n["kind"], 0) + 1
        out[n["id"]] = n.get("name") or f"{LABEL.get(n['kind'], n['kind'])} {count[n['kind']]}"
    return out


def validate(plant: dict) -> list[str]:
    nodes = plant.get("nodes", [])
    edges = plant.get("edges", [])
    nm = names(nodes)
    ids = {n["id"] for n in nodes}
    errs = []
    if not any(n["kind"] == "feed" for n in nodes):
        errs.append("Add a Feed block. Nothing can flow without one.")
    if not any(n["kind"] == "product" for n in nodes):
        errs.append("Add a Product block so the stream has somewhere to leave.")
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            errs.append("A pipe points at a block that no longer exists. Delete it.")
        if e["from"] == e["to"]:
            errs.append(f"{nm.get(e['from'], '?')} is piped into itself. Use a Splitter to make a recycle.")
    seen = set()
    for e in edges:
        key = (e["from"], e.get("port", "out"))
        if key in seen:
            errs.append(f"{nm[e['from']]} sends one outlet to two places. Put a Splitter there.")
        seen.add(key)
    for n in nodes:
        k = n["kind"]
        ins = [e for e in edges if e["to"] == n["id"]]
        if k == "feed" and ins:
            errs.append(f"{nm[n['id']]} has a pipe coming in. Feeds only send.")
        if k not in ("feed",) and not ins:
            errs.append(f"{nm[n['id']]} has nothing coming in.")
        for p in PORTS[k]:
            if not any(e["from"] == n["id"] and e.get("port", "out") == p for e in edges):
                where = "side outlet" if p == "side" else "outlet"
                errs.append(f"The {where} of {nm[n['id']]} is not connected.")
        if k == "feed":
            if _f(n, "Q", 0) <= 0:
                errs.append(f"{nm[n['id']]} needs a flow above zero.")
            if _f(n, "CA", 0) < 0 or _f(n, "CB", 0) < 0:
                errs.append(f"{nm[n['id']]} has a negative concentration.")
        if k in UNIT_KINDS and _f(n, "V", 0) <= 0:
            errs.append(f"{nm[n['id']]} needs a volume above zero.")
        if k == "bed" and _f(n, "Pe", 0) <= 0:
            errs.append(f"{nm[n['id']]} needs a Peclet number above zero.")
        if k == "split" and not (0.0 <= _f(n, "f", 0.5) < 1.0):
            errs.append(f"{nm[n['id']]}: side fraction must be at least 0 and below 1.")
    return list(dict.fromkeys(errs))


def flows(plant: dict) -> dict:
    nodes = plant["nodes"]
    edges = plant["edges"]
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    b = np.zeros(n)
    for nd in nodes:
        if nd["kind"] == "feed":
            b[idx[nd["id"]]] = _f(nd, "Q", 0)
    for e in edges:
        src = nodes[idx[e["from"]]]
        w = 1.0
        if src["kind"] == "split":
            f = _f(src, "f", 0.5)
            w = f if e.get("port", "out") == "side" else 1.0 - f
        A[idx[e["to"]], idx[e["from"]]] += w
    M = np.eye(n) - A
    if abs(np.linalg.det(M)) < 1e-10 or np.linalg.cond(M) > 1e10:
        raise EngineError("There is a loop that liquid can enter but never leave. Lower a splitter's side fraction or add an exit.")
    Q = np.linalg.solve(M, b)
    if np.any(Q < -1e-9):
        raise EngineError("Flow balance gave a negative flow. Check the pipe directions.")
    edge_Q = []
    for e in edges:
        src = nodes[idx[e["from"]]]
        w = 1.0
        if src["kind"] == "split":
            f = _f(src, "f", 0.5)
            w = f if e.get("port", "out") == "side" else 1.0 - f
        edge_Q.append(w * Q[idx[e["from"]]])
    return {"node_Q": Q, "edge_Q": np.array(edge_Q), "idx": idx}


def _tank_iso(c_in, Q, V, k, K):
    ca_in, cb_in, cc_in, cd_in = c_in
    d = cb_in - ca_in
    lo = max(0.0, -d)
    hi = ca_in + min(cc_in, cd_in)

    def comp(ca):
        x = ca_in - ca
        return np.array([ca, ca + d, cc_in + x, cd_in + x])

    def g(ca):
        return Q * (ca_in - ca) - V * rate(comp(ca), k, K)

    if hi - lo < 1e-15:
        return comp(lo)
    return comp(brentq(g, lo, hi, xtol=1e-15, rtol=1e-13))


def _tank_jacket(c_in, T_in, Q, V, kin: Kinetics, UA, Ta):
    rc = kin.rho_cp

    def f(_t, y):
        r = rate(y[:4], kin.k(y[4]), kin.k2_ratio)
        dc = Q / V * (np.asarray(c_in) - y[:4]) + NU * r
        dT = Q / V * (T_in - y[4]) + (-kin.dH) * r / rc - UA * (y[4] - Ta) / (rc * V)
        return np.append(dc, dT)

    y0 = np.append(np.asarray(c_in, dtype=float), T_in)
    sol = solve_ivp(f, (0.0, 60.0 * V / Q), y0, method="BDF", rtol=1e-8, atol=1e-11)
    if not sol.success:
        raise EngineError(f"Jacketed tank did not settle: {sol.message}")
    y = sol.y[:, -1]
    pol = root(lambda z: f(0, z), y, tol=1e-12)
    if pol.success and np.max(np.abs(f(0, pol.x))) < 1e-8 and np.all(pol.x[:4] > -1e-9):
        y = pol.x
    if np.max(np.abs(f(0, y))) > 1e-5:
        raise EngineError("A jacketed tank kept changing after 60 residence times. It may be oscillating.")
    return np.clip(y[:4], 0, None), float(y[4])


def _unit(node, c_in, T_in, Q, kin: Kinetics):
    k = node["kind"]
    if k in ("feed",):
        return np.array([_f(node, "CA", 0), _f(node, "CB", 0), 0.0, 0.0]), c_to_k(_f(node, "T", 30)), None
    if k in ("split", "product") or Q <= 0:
        return np.asarray(c_in, float), T_in, None
    V = _f(node, "V", 1.0)
    tau = V / Q
    if k == "cstr":
        if node.get("jacket"):
            c, T = _tank_jacket(c_in, T_in, Q, V, kin, _f(node, "UA", 0.0), c_to_k(_f(node, "Ta", 30)))
            return c, T, {"z": np.array([0.0, 1.0]), "c": np.column_stack([c, c]), "T": np.array([T, T])}
        Tu = c_to_k(_f(node, "T", 30))
        c = _tank_iso(c_in, Q, V, kin.k(Tu), kin.k2_ratio)
        return c, Tu, {"z": np.array([0.0, 1.0]), "c": np.column_stack([c, c]), "T": np.array([Tu, Tu])}
    Tu = c_to_k(_f(node, "T", 30))
    if k == "pfr":
        s, y = _plug(c_in, Tu, tau, kin, False, n=41)
        return y[:4, -1], Tu, {"z": s / tau, "c": y[:4], "T": np.full(len(s), Tu)}
    if k == "bed":
        eta, prof = dispersion_profile(c_in, kin.k(Tu), kin.k2_ratio, tau, _f(node, "Pe", 40), n=41)
        return prof[:, -1], Tu, {"z": eta, "c": prof, "T": np.full(len(eta), Tu)}
    raise EngineError(f"Unknown block '{k}'.")


def _order(plant):
    """Feeds first, then breadth-first along pipes. Back edges are the tears."""
    nodes = plant["nodes"]
    edges = plant["edges"]
    order, seen = [], set()
    queue = [n["id"] for n in nodes if n["kind"] == "feed"]
    while queue:
        i = queue.pop(0)
        if i in seen:
            continue
        seen.add(i)
        order.append(i)
        queue.extend(e["to"] for e in edges if e["from"] == i)
    order += [n["id"] for n in nodes if n["id"] not in seen]
    return order


def solve(plant: dict, kin: Kinetics, tol: float = 1e-10, max_iter: int = 400) -> dict:
    errs = validate(plant)
    if errs:
        raise EngineError(" ".join(errs))
    nodes = plant["nodes"]
    edges = plant["edges"]
    nm = names(nodes)
    fl = flows(plant)
    idx, Qn, Qe = fl["idx"], fl["node_Q"], fl["edge_Q"]
    order = _order(plant)
    n = len(nodes)
    feeds = [nd for nd in nodes if nd["kind"] == "feed"]
    T_feed = np.mean([c_to_k(_f(nd, "T", 30)) for nd in feeds])
    # state: outlet c (4) + T for every node
    X = np.zeros((n, 5))
    X[:, 4] = T_feed
    incoming = {nd["id"]: [j for j, e in enumerate(edges) if e["to"] == nd["id"]] for nd in nodes}
    profiles = {}

    def sweep(Xs):
        Xs = Xs.copy()
        for nid in order:
            i = idx[nid]
            nd = nodes[i]
            ins = incoming[nid]
            Qin = sum(Qe[j] for j in ins)
            if ins and Qin > 0:
                c_in = sum(Qe[j] * Xs[idx[edges[j]["from"]], :4] for j in ins) / Qin
                T_in = sum(Qe[j] * Xs[idx[edges[j]["from"]], 4] for j in ins) / Qin
            else:
                c_in, T_in = np.zeros(4), T_feed
            c, T, prof = _unit(nd, c_in, T_in, Qin if nd["kind"] != "feed" else Qn[i], kin)
            Xs[i, :4], Xs[i, 4] = c, T
            if prof is not None:
                profiles[nid] = dict(prof, c_in=c_in, T_in=T_in, Q=Qin)
            elif nd["kind"] in ("split", "product"):
                profiles[nid] = {"c_in": c_in, "T_in": T_in, "Q": Qin}
        return Xs

    x_prev, g_prev = None, None
    x = X.ravel()
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        g = sweep(x.reshape(n, 5)).ravel()
        if np.max(np.abs(g - x)) < tol:
            x = g
            converged = True
            break
        if x_prev is not None:
            dx = x - x_prev
            dg = g - g_prev
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(np.abs(dx) > 1e-14, dg / dx, 0.0)
                q = np.where(np.abs(s - 1) > 1e-12, s / (s - 1), 0.0)
            q = np.clip(q, -5.0, 0.0)
            new = q * x + (1 - q) * g
        else:
            new = g
        x_prev, g_prev = x, g
        x = new
        x.reshape(n, 5)[:, :4] = np.clip(x.reshape(n, 5)[:, :4], 0, None)
    if not converged:
        raise EngineError(f"The recycle did not converge in {max_iter} passes. Try a smaller side fraction.")
    X = x.reshape(n, 5)
    sweep(X)  # refresh profiles at the converged state

    streams = []
    for j, e in enumerate(edges):
        src = X[idx[e["from"]]]
        streams.append({
            "From": nm[e["from"]] + (" (side)" if e.get("port") == "side" else ""), "To": nm[e["to"]],
            "Q (L/min)": Qe[j], "NaOH (mol/L)": src[0], "Ester (mol/L)": src[1],
            "Acetate (mol/L)": src[2], "Ethanol (mol/L)": src[3], "T (°C)": k_to_c(src[4]),
        })
    units = []
    for nd in nodes:
        if nd["kind"] not in UNIT_KINDS:
            continue
        i = idx[nd["id"]]
        p = profiles.get(nd["id"], {})
        ca_in = p.get("c_in", np.zeros(4))[0]
        Qin = p.get("Q", 0.0)
        V = _f(nd, "V", 1.0)
        duty = None
        if nd["kind"] == "cstr" and nd.get("jacket"):
            duty = _f(nd, "UA", 0) * (X[i, 4] - c_to_k(_f(nd, "Ta", 30)))  # J/min removed by jacket
        units.append({
            "id": nd["id"], "name": nm[nd["id"]], "kind": nd["kind"], "V": V, "Q": Qin,
            "tau": V / Qin if Qin > 0 else np.inf, "T_C": k_to_c(X[i, 4]),
            "X_pass": 1 - X[i, 0] / ca_in if ca_in > 0 else np.nan,
            "CA_in": ca_in, "CA_out": X[i, 0], "duty_J_min": duty,
        })
    fa_in = sum(_f(nd, "Q", 0) * _f(nd, "CA", 0) for nd in feeds)
    fb_in = sum(_f(nd, "Q", 0) * _f(nd, "CB", 0) for nd in feeds)
    prods = [nd for nd in nodes if nd["kind"] == "product"]
    out = {i: sum(Qn[idx[p["id"]]] * X[idx[p["id"]], i] for p in prods) for i in range(4)}
    X_total = 1 - out[0] / fa_in if fa_in > 0 else np.nan
    sodium = abs(out[0] + out[2] - fa_in) / max(fa_in, 1e-30)
    acetyl = abs(out[1] + out[2] - fb_in) / max(fb_in, 1e-30) if fb_in > 0 else 0.0
    return {
        "X": X_total, "streams": streams, "units": units, "profiles": profiles, "names": nm,
        "iterations": it, "balance_error": max(sodium, acetyl), "node_state": X, "node_Q": Qn,
        "product_Q": sum(Qn[idx[p["id"]]] for p in prods),
    }


def main_path(plant: dict) -> list[str]:
    """Feed → … → product following main outlets, for the profile plot."""
    nodes = {n["id"]: n for n in plant["nodes"]}
    edges = plant["edges"]
    start = next((n["id"] for n in plant["nodes"] if n["kind"] == "feed"), None)
    path, cur = [], start
    while cur and cur not in path:
        path.append(cur)
        nxt = [e["to"] for e in edges if e["from"] == cur and e.get("port", "out") == "out"]
        cur = nxt[0] if nxt else None
    return [p for p in path if nodes[p]["kind"] in UNIT_KINDS]
