"""Continuous stirred tank, ideal and non-ideal, steady and transient.

Flow picture (each piece can be switched off):

    feed ──┬──────────────────────────────── bypass β Q ─────────┐
           └─ (1-β)Q ─> tank 1 ─> tank 2 ... tank N ─────────────┴─> outlet
                         ↕ exchange  R·(1-β)Q
                       stagnant zone (volume αV)

Active volume (1-α)V is split equally over N tanks. With R = 0 the stagnant
zone is dead volume: it holds liquid but takes no part. Every zone obeys

    Vz dc/dt = Σ Q_in c_in - Q_out c + Vz ν r(c, T)

and, when the jacket is switched on,

    ρcp Vz dT/dt = ρcp Σ Q_in (T_in - T) + (-ΔH) r Vz - UA (Vz/V) (T - Ta)

With N = 1, β = 0, α = 0, k2 = 0 and constant T this is the textbook CSTR,
whose steady state is X/((1-X)(M-X)) = k CA0 τ. The tests check that.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, root

from .rate import NU, Kinetics, rate
from .units import EngineError, c_to_k, check_all, k_to_c


@dataclass(frozen=True)
class CSTRModel:
    V: float  # L, total liquid volume
    qA: float  # L/min, NaOH stream
    qB: float  # L/min, ethyl acetate stream
    cA_feed: float  # mol/L in the NaOH bottle
    cB_feed: float  # mol/L in the ester bottle
    T_C: float  # isothermal: tank temperature. Jacket on: feed temperature.
    kin: Kinetics = field(default_factory=lambda: Kinetics(k_ref=6.0))
    n_tanks: int = 1
    bypass: float = 0.0
    stagnant: float = 0.0
    exchange: float = 0.0  # stagnant exchange flow / through flow. 0 = dead volume
    jacket: bool = False
    UA: float = 0.0  # J/min/K for the whole vessel
    T_jacket_C: float = 30.0

    @property
    def Q(self) -> float:
        return self.qA + self.qB

    @property
    def CA0(self) -> float:
        return self.cA_feed * self.qA / self.Q if self.Q > 0 else 0.0

    @property
    def CB0(self) -> float:
        return self.cB_feed * self.qB / self.Q if self.Q > 0 else 0.0

    @property
    def M(self) -> float:
        return self.CB0 / self.CA0 if self.CA0 > 0 else float("nan")

    @property
    def tau(self) -> float:
        return self.V / self.Q if self.Q > 0 else float("inf")

    @property
    def has_exchange(self) -> bool:
        return self.stagnant > 0 and self.exchange > 0

    def problems(self) -> list[str]:
        out = check_all(
            [
                ("V", self.V), ("q", self.qA), ("q", self.qB), ("C", self.cA_feed), ("C", self.cB_feed),
                ("T_C", self.T_C), ("k", self.kin.k_ref), ("bypass", self.bypass), ("dead", self.stagnant),
                ("N", self.n_tanks), ("UA", self.UA), ("T_C", self.T_jacket_C),
            ]
        )
        if self.Q <= 0:
            out.append("Both feed flows are zero, so nothing passes through the tank.")
        if self.CA0 <= 0:
            out.append("There is no NaOH in the mixed feed, so conversion of NaOH is undefined.")
        if self.exchange < 0:
            out.append("Exchange flow cannot be negative.")
        return out


def _zones(m: CSTRModel):
    Va = (1.0 - m.stagnant) * m.V
    N = int(m.n_tanks)
    Vi = Va / N
    Qr = (1.0 - m.bypass) * m.Q
    Vs = m.stagnant * m.V
    Qe = m.exchange * Qr if m.has_exchange else 0.0
    return N, Vi, Qr, Vs, Qe


def n_states(m: CSTRModel) -> tuple[int, int]:
    """(number of zones, values per zone)."""
    N, *_ = _zones(m)
    return N + (1 if m.has_exchange else 0), (5 if m.jacket else 4)


def rhs_factory(m: CSTRModel):
    N, Vi, Qr, Vs, Qe = _zones(m)
    nz, per = n_states(m)
    cin = np.array([m.CA0, m.CB0, 0.0, 0.0])
    Tin = c_to_k(m.T_C)
    Ta = c_to_k(m.T_jacket_C)
    kin = m.kin
    k_iso = kin.k(Tin)
    rc = kin.rho_cp

    def f(_t, y):
        y = y.reshape(nz, per)
        dy = np.zeros_like(y)
        for z in range(nz):
            c = y[z, :4]
            T = y[z, 4] if m.jacket else Tin
            k = kin.k(T) if m.jacket else k_iso
            r = rate(c, k, kin.k2_ratio)
            if z < N:
                Vz = Vi
                c_up = cin if z == 0 else y[z - 1, :4]
                T_up = Tin if z == 0 else (y[z - 1, 4] if m.jacket else Tin)
                dc = Qr / Vz * (c_up - c) + NU * r
                dT = Qr / Vz * (T_up - T) if m.jacket else 0.0
                if z == 0 and Qe > 0:
                    cs = y[N, :4]
                    dc += Qe / Vz * (cs - c)
                    if m.jacket:
                        dT += Qe / Vz * (y[N, 4] - T)
            else:
                Vz = Vs
                c0 = y[0, :4]
                dc = Qe / Vz * (c0 - c) + NU * r
                dT = Qe / Vz * (y[0, 4] - T) if m.jacket else 0.0
            dy[z, :4] = dc
            if m.jacket:
                dT += (-kin.dH) * r / rc - m.UA * (Vz / m.V) * (T - Ta) / (rc * Vz)
                dy[z, 4] = dT
        return dy.ravel()

    return f


def outlet(m: CSTRModel, y):
    """Outlet concentrations and temperature from a state vector (or array of them)."""
    nz, per = n_states(m)
    N, *_ = _zones(m)
    y = np.asarray(y, dtype=float)
    single = y.ndim == 1
    Y = y.reshape(-1, nz, per) if not single else y.reshape(1, nz, per)
    cin = np.array([m.CA0, m.CB0, 0.0, 0.0])
    last = Y[:, N - 1, :4]
    c = (1.0 - m.bypass) * last + m.bypass * cin
    if m.jacket:
        T = (1.0 - m.bypass) * Y[:, N - 1, 4] + m.bypass * c_to_k(m.T_C)
    else:
        T = np.full(len(Y), c_to_k(m.T_C))
    return (c[0], float(T[0])) if single else (c, T)


def initial_state(m: CSTRModel, init: str = "water", T_init_C: float | None = None):
    nz, per = n_states(m)
    if init == "water":
        c = np.zeros(4)
    elif init == "feed":
        c = np.array([m.CA0, m.CB0, 0.0, 0.0])
    else:
        raise EngineError(f"Unknown start condition '{init}'.")
    y = np.zeros((nz, per))
    y[:, :4] = c
    if m.jacket:
        y[:, 4] = c_to_k(T_init_C if T_init_C is not None else m.T_C)
    return y.ravel()


# ---------------------------------------------------------------- steady state


def _tank_CA(CA_in, d, CA0, k, k2r, Q, V, Qe=0.0, CAs=None):
    """Steady CA leaving one ideal tank, using the stoichiometric invariants.

    CB = CA + d, CC = CD = CA0 - CA everywhere, because every stream in the
    vessel came from the same feed. That reduces each tank to one equation.
    """
    lo = max(0.0, -d)

    def r(ca):
        return k * (ca * (ca + d) - k2r * (CA0 - ca) ** 2)

    def g(ca):
        extra = Qe * (CAs - ca) if Qe > 0 else 0.0
        return Q * (CA_in - ca) + extra - V * r(ca)

    hi = max(CA_in, CAs if CAs is not None else CA_in)
    if hi <= lo:
        return lo
    return brentq(g, lo, hi, xtol=1e-14, rtol=1e-12)


def steady_isothermal(m: CSTRModel) -> dict:
    """Exact steady state for constant temperature, any N, bypass, stagnant zone."""
    N, Vi, Qr, Vs, Qe = _zones(m)
    CA0, CB0 = m.CA0, m.CB0
    d = CB0 - CA0
    k = m.kin.k_C(m.T_C)
    k2r = m.kin.k2_ratio
    lo = max(0.0, -d)

    def stag_of(ca1):
        def h(cs):
            return Qe * (ca1 - cs) - Vs * k * (cs * (cs + d) - k2r * (CA0 - cs) ** 2)

        if ca1 <= lo:
            return lo
        return brentq(h, lo, ca1, xtol=1e-14, rtol=1e-12)

    zones = []
    if Qe > 0:
        def g(ca1):
            cs = stag_of(ca1)
            return Qr * (CA0 - ca1) + Qe * (cs - ca1) - Vi * k * (ca1 * (ca1 + d) - k2r * (CA0 - ca1) ** 2)

        ca1 = brentq(g, lo, CA0, xtol=1e-14, rtol=1e-12) if CA0 > lo else lo
        cas = stag_of(ca1)
    else:
        ca1 = _tank_CA(CA0, d, CA0, k, k2r, Qr, Vi) if Qr > 0 else CA0
        cas = None
    zones.append(ca1)
    for _ in range(1, N):
        zones.append(_tank_CA(zones[-1], d, CA0, k, k2r, Qr, Vi))
    ca_last = zones[-1]
    ca_out = (1.0 - m.bypass) * ca_last + m.bypass * CA0
    to_c = lambda ca: np.array([ca, ca + d, CA0 - ca, CA0 - ca])  # noqa: E731
    out_c = to_c(ca_out)
    return {
        "c_out": out_c,
        "X": 1.0 - ca_out / CA0,
        "X_tanks": [1.0 - z / CA0 for z in zones],
        "X_stagnant": (1.0 - cas / CA0) if cas is not None else None,
        "T_out_C": m.T_C,
        "state": np.concatenate([to_c(z) for z in zones] + ([to_c(cas)] if cas is not None else [])),
    }


def steady_state(m: CSTRModel, init: str = "feed", T_init_C: float | None = None) -> dict:
    """Steady state. Exact for isothermal; integrate-then-polish with the jacket on.

    With the jacket on there can be more than one steady state. The one
    returned is the one the tank reaches from the given start.
    """
    probs = m.problems()
    if probs:
        raise EngineError(" ".join(probs))
    if not m.jacket:
        return steady_isothermal(m)
    f = rhs_factory(m)
    y0 = initial_state(m, init, T_init_C)
    horizon = 60.0 * max(m.tau, 1e-3)
    sol = solve_ivp(f, (0.0, horizon), y0, method="BDF", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise EngineError(f"The energy balance did not integrate: {sol.message}")
    guess = sol.y[:, -1]
    pol = root(lambda y: f(0.0, y), guess, method="hybr", tol=1e-12)
    y = pol.x if pol.success and np.max(np.abs(f(0.0, pol.x))) < 1e-8 else guess
    resid = float(np.max(np.abs(f(0.0, y))))
    if resid > 1e-5:
        raise EngineError("The tank had not settled after 60 residence times. It may be oscillating.")
    c, T = outlet(m, y)
    nz, per = n_states(m)
    N, *_ = _zones(m)
    Y = y.reshape(nz, per)
    return {
        "c_out": c,
        "X": 1.0 - c[0] / m.CA0,
        "X_tanks": [1.0 - Y[i, 0] / m.CA0 for i in range(N)],
        "X_stagnant": (1.0 - Y[N, 0] / m.CA0) if m.has_exchange else None,
        "T_out_C": k_to_c(T),
        "T_tanks_C": [k_to_c(Y[i, 4]) for i in range(N)],
        "state": y,
    }


def jacobian(f, y, eps=1e-7):
    n = len(y)
    J = np.zeros((n, n))
    f0 = f(0.0, y)
    for i in range(n):
        h = eps * max(1.0, abs(y[i]))
        yp = y.copy()
        yp[i] += h
        J[:, i] = (f(0.0, yp) - f0) / h
    return J


def settling(m: CSTRModel, ss: dict | None = None, init: str = "water", y0=None, T_init_C: float | None = None) -> dict:
    """How fast the outlet conductivity closes in on its steady value.

    Two kinds of disturbance relax at different speeds. Total sodium
    (NaOH + acetate) is not changed by reaction, so after a start from water
    it washes in on the plain residence time τ. A change of flow with the same
    feed leaves that total alone and only the reaction mode is excited; for one
    ideal tank that mode relaxes in τ(1-X)/(1+X). Rather than guess which one
    applies, the start-up actually chosen is simulated and the decay rate of
    its tail is measured. The eigenvalues of the linearised balances give
    stability and the slowest possible time constant.

    shape: the simulated outlet deviation from steady state, as fractions of
    CA0 for NaOH (dA) and acetate (dC), so any Λ0/Λ∞ can be applied later.
    """
    ss = ss or steady_state(m)
    f = rhs_factory(m)
    J = jacobian(f, np.asarray(ss["state"], dtype=float))
    ev = np.linalg.eigvals(J)
    re = ev.real
    stable = bool(np.all(re < 0))
    slow = float(-1.0 / np.max(re)) if stable else float("inf")
    tau_r = slow
    shape = None
    try:
        sim = simulate(m, t_end=40.0 * m.tau, n_out=2001, init=init, y0=y0, T_init_C=T_init_C)
        shape = {"t": sim["t"], "dA": (sim["c_out"][:, 0] - ss["c_out"][0]) / m.CA0, "dC": (sim["c_out"][:, 2] - ss["c_out"][2]) / m.CA0}
        lam = conductivity(sim["c_out"][:, 0], sim["c_out"][:, 2], m.CA0, 1.0, 0.366)
        lam_ss = conductivity(ss["c_out"][0], ss["c_out"][2], m.CA0, 1.0, 0.366)
        dev = np.abs(lam - lam_ss)
        top = dev.max()
        win = (dev < 1e-2 * top) & (dev > 1e-6 * top)
        if top > 0 and win.sum() >= 10:
            slope = np.polyfit(sim["t"][win], np.log(dev[win]), 1)[0]
            if slope < 0:
                tau_r = float(-1.0 / slope)
    except EngineError:
        pass
    return {"tau_r": tau_r, "tau_slow": slow, "stable": stable, "eigen": ev, "shape": shape}


def simulate(m: CSTRModel, t_end: float, n_out: int = 241, init: str = "water", T_init_C: float | None = None, y0=None) -> dict:
    """Transient from a start condition. Integration is done once; the
    animation plays back these arrays."""
    probs = m.problems()
    if probs:
        raise EngineError(" ".join(probs))
    if not np.isfinite(t_end) or t_end <= 0:
        raise EngineError("Simulation time must be above zero.")
    f = rhs_factory(m)
    y_start = np.asarray(y0, dtype=float) if y0 is not None else initial_state(m, init, T_init_C)
    t_eval = np.linspace(0.0, float(t_end), int(n_out))
    sol = solve_ivp(f, (0.0, float(t_end)), y_start, t_eval=t_eval, method="LSODA", rtol=1e-8, atol=1e-11)
    if not sol.success:
        raise EngineError(f"The solver stopped early: {sol.message}")
    if np.any(sol.y[:, -1] < -1e-6):
        raise EngineError("The solver produced negative concentrations, so the result is not shown.")
    Y = sol.y.T
    c, T = outlet(m, Y)
    nz, per = n_states(m)
    N, *_ = _zones(m)
    Z = Y.reshape(len(Y), nz, per)
    CA0 = m.CA0
    X_out = 1.0 - c[:, 0] / CA0
    # Conversion is undefined while the tank still holds mostly water, so the
    # animation and the plots use reacted fraction relative to feed instead.
    extent = c[:, 2] / CA0
    return {
        "t": sol.t,
        "c_out": c,
        "T_out_C": k_to_c(T),
        "X_out": X_out,
        "extent_out": extent,
        "tank_CA": Z[:, :N, 0],
        "tank_CC": Z[:, :N, 2],
        "stag_CC": Z[:, N, 2] if m.has_exchange else None,
        "stag_CA": Z[:, N, 0] if m.has_exchange else None,
        "T_tanks_C": k_to_c(Z[:, :N, 4]) if m.jacket else None,
    }


def conductivity(cA, cC, CA0, lam0, lam_inf):
    """κ of the outlet from ion amounts, anchored to measured Λ0 and Λ∞.

    Λ0 is the mixed feed (all sodium as NaOH, at CA0). Λ∞ is the same sodium
    fully as acetate. In between κ is linear in the two ion amounts, and
    pure water reads zero. At steady state this reduces to the manual's
    X = (Λ0 - Λ)/(Λ0 - Λ∞).
    """
    cA = np.asarray(cA, dtype=float)
    cC = np.asarray(cC, dtype=float)
    return lam0 * cA / CA0 + lam_inf * cC / CA0


def ideal_X(Da: float, M: float = 1.0) -> float:
    """Analytic ideal-CSTR conversion for -rA = k CA CB, Da = k CA0 τ."""
    if Da <= 0:
        return 0.0
    if abs(M - 1.0) < 1e-12:
        return float(1.0 + (1.0 - np.sqrt(1.0 + 4.0 * Da)) / (2.0 * Da))
    # Da (1-X)(M-X) = X  ->  Da X^2 - (Da(1+M) + 1) X + Da M = 0
    a = Da
    b = -(Da * (1.0 + M) + 1.0)
    c = Da * M
    return float((-b - np.sqrt(b * b - 4 * a * c)) / (2 * a))


def with_(m: CSTRModel, **kw) -> CSTRModel:
    return replace(m, **kw)
