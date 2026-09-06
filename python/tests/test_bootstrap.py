"""Bootstrap: round-trip repricing, edge cases, invalid-quote detection."""

from __future__ import annotations

import pytest

from irm import (
    FRA,
    MAX_MATURITY,
    MIN_MATURITY,
    Deposit,
    DiscountCurve,
    OISSwap,
    Swap,
    annual_schedule,
    bootstrap,
    load_curve_quotes,
    load_ois_quotes,
    par_swap_rate,
    solve_pillar_df,
)


def test_eur_round_trip(eur_curve: DiscountCurve, data_dir) -> None:
    instruments = load_curve_quotes(data_dir / "curve_quotes.csv", "EUR")
    for ins in instruments:
        assert abs(ins.residual(eur_curve)) < 1e-9
    assert len(eur_curve.times) == len(instruments)


def test_usd_round_trip(usd_curve: DiscountCurve, data_dir) -> None:
    instruments = load_curve_quotes(data_dir / "curve_quotes.csv", "USD")
    for ins in instruments:
        assert abs(ins.residual(usd_curve)) < 1e-9


def test_ois_round_trip(ois_curve: DiscountCurve, data_dir) -> None:
    for ins in load_ois_quotes(data_dir / "ois_quotes.csv"):
        assert abs(ins.residual(ois_curve)) < 1e-9


def test_eur_negative_short_end_df_above_one(eur_curve: DiscountCurve) -> None:
    # Negative deposit rates imply DF > 1 — must be supported, not clipped.
    assert eur_curve.df(0.5) > 1.0
    assert eur_curve.df(1.0) > 1.0
    assert eur_curve.zero_rate(1.0) == pytest.approx(-0.003, abs=5e-4)


def test_usd_df_positive_and_decreasing(usd_curve: DiscountCurve) -> None:
    prev = 1.0
    for t in usd_curve.times:
        df = usd_curve.df(t)
        assert 0.0 < df < prev
        prev = df


def test_bootstrapped_par_rates_match_quotes(usd_curve: DiscountCurve, data_dir) -> None:
    # Property-style: every quoted swap's par rate off the final curve
    # equals its input quote (repricing in rate space).
    for ins in load_curve_quotes(data_dir / "curve_quotes.csv", "USD"):
        if isinstance(ins, Swap):
            times = [0.0] + list(annual_schedule(ins.maturity))
            assert par_swap_rate(usd_curve, times) == pytest.approx(
                ins.rate, abs=1e-10
            )


def test_bootstrap_with_fra_strip() -> None:
    instruments = [
        Deposit(0.5, 0.030),
        FRA(0.5, 1.0, 0.032),
        FRA(1.0, 1.5, 0.034),
        Swap(3.0, 0.033),
    ]
    curve = bootstrap(instruments)
    for ins in instruments:
        assert abs(ins.residual(curve)) < 1e-9
    # FRA repricing in rate space too.
    assert curve.fwd_rate(0.5, 1.0) == pytest.approx(0.032, abs=1e-12)


def test_deposit_only_and_swap_only() -> None:
    dep_curve = bootstrap([Deposit(0.25, 0.02), Deposit(0.5, 0.021), Deposit(1.0, 0.022)])
    assert dep_curve.df(1.0) == pytest.approx(1.0 / 1.022, rel=1e-12)
    swp = [Swap(1.0, 0.02), Swap(2.0, 0.022), Swap(5.0, 0.025), Swap(10.0, 0.027)]
    swp_curve = bootstrap(swp)  # gap pillars (3,4,6..9) come from interpolation
    for ins in swp:
        assert abs(ins.residual(swp_curve)) < 1e-9


def test_fractional_swap_maturity_schedule() -> None:
    assert annual_schedule(2.0) == (1.0, 2.0)
    assert annual_schedule(2.5) == (0.5, 1.5, 2.5)
    assert annual_schedule(0.25) == (0.25,)
    with pytest.raises(ValueError):
        annual_schedule(0.0)


def test_ordering_violations_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        bootstrap([Deposit(1.0, 0.02), Deposit(0.5, 0.02)])
    with pytest.raises(ValueError, match="strictly increasing"):
        bootstrap([Deposit(1.0, 0.02), Swap(1.0, 0.02)])  # duplicate pillar
    with pytest.raises(ValueError, match="at least one"):
        bootstrap([])


def test_invalid_quotes_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-positive discount factor"):
        Deposit(1.0, -1.5)  # 1 + R*T <= 0 -> negative implied DF
    with pytest.raises(ValueError, match="non-positive discount factor"):
        FRA(1.0, 2.0, -1.2)
    with pytest.raises(ValueError):
        FRA(2.0, 1.0, 0.02)  # start >= end
    with pytest.raises(ValueError):
        Deposit(1.0, float("nan"))
    with pytest.raises(ValueError):
        Swap(-1.0, 0.02)


def test_crossed_quotes_fail_bracketing() -> None:
    # A -99% 2y swap after a sane 1y deposit needs DF(2) ~ 199 — outside any
    # admissible discount factor: bootstrap must raise, not return garbage.
    with pytest.raises(ValueError, match="crossed"):
        bootstrap([Deposit(1.0, 0.05), Swap(2.0, -0.99)])


class _NanInstrument:
    """Duck-typed instrument whose residual is NaN inside the bracket."""

    rate = 0.02

    @property
    def pillar(self) -> float:
        return 1.0

    def residual(self, curve: DiscountCurve) -> float:
        df = curve.df(1.0)
        if 0.5 < df < 1.5:
            return float("nan")
        return df - 0.97


def test_bootstrap_non_convergence_message_not_crossed_quotes() -> None:
    # A solver failure that is not a bracketing failure must not be
    # relabelled as crossed quotes (MINOR-10).
    with pytest.raises(ValueError) as exc:
        bootstrap([_NanInstrument()])
    msg = str(exc.value)
    assert "solver failed" in msg and "not finite" in msg
    assert "crossed" not in msg
    # The per-pillar kernel exposes the same classification directly.
    with pytest.raises(ValueError, match="crossed"):
        solve_pillar_df(lambda x: 1.0 + x, 2.0, "test")
    with pytest.raises(ValueError, match="solver failed"):
        solve_pillar_df(lambda x: float("inf") if x > 1.0 else -1.0, 2.0, "test")
    assert solve_pillar_df(lambda x: x - 0.9, 2.0, "test") == pytest.approx(0.9, abs=1e-13)


def test_swap_maturity_out_of_range_rejected() -> None:
    for bad in (1e12, 1e9, MAX_MATURITY * (1 + 1e-9), 1e-13, MIN_MATURITY / 2.0, float("inf")):
        with pytest.raises(ValueError):
            Swap(bad, 0.03)
        with pytest.raises(ValueError):
            OISSwap(bad, 0.03)
        with pytest.raises(ValueError):
            Deposit(bad, 0.03)
        with pytest.raises(ValueError):
            FRA(0.0, bad, 0.03)
        with pytest.raises(ValueError):
            annual_schedule(bad)
    # The bounds themselves are accepted and produce sane schedules.
    assert len(annual_schedule(MAX_MATURITY)) == 200
    assert annual_schedule(MIN_MATURITY) == (MIN_MATURITY,)
    assert len(annual_schedule(199.5)) == 200


def test_dv01_localises_in_quote_space(data_dir) -> None:
    quotes = load_curve_quotes(data_dir / "curve_quotes.csv", "USD")
    times = [0.0] + [float(i) for i in range(1, 11)]

    def bumped(pillar: float, bump: float = 1e-4):
        return bootstrap([
            type(q)(maturity=q.maturity, rate=q.rate + (bump if q.pillar == pillar else 0.0))
            for q in quotes
        ])

    base = par_swap_rate(bootstrap(quotes), times)
    assert base == pytest.approx(0.0425, abs=1e-12)
    # Bumping the 5y quote leaves the 10y par rate untouched ...
    assert par_swap_rate(bumped(5.0), times) == pytest.approx(0.0425, abs=1e-12)
    # ... bumping the 10y quote moves it by exactly the bump.
    assert par_swap_rate(bumped(10.0), times) - 0.0425 == pytest.approx(1e-4, abs=1e-12)


def test_ois_short_stub_schedule() -> None:
    # A 1M OIS has a single fixed payment with accrual = maturity.
    c = bootstrap([OISSwap(1.0 / 12.0, 0.03)])
    t = 1.0 / 12.0
    assert c.df(t) == pytest.approx(1.0 / (1.0 + 0.03 * t), rel=1e-12)
