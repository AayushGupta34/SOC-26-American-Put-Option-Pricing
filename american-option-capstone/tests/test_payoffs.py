import numpy as np

from pricing.payoffs import put_payoff, call_payoff, payoff


def test_put_payoff_never_negative():
    spots = np.linspace(0.0, 300.0, 50)
    assert np.all(put_payoff(spots, 100.0) >= 0.0)


def test_call_payoff_never_negative():
    spots = np.linspace(0.0, 300.0, 50)
    assert np.all(call_payoff(spots, 100.0) >= 0.0)


def test_put_payoff_matches_formula():
    assert put_payoff(80.0, 100.0) == 20.0
    assert put_payoff(120.0, 100.0) == 0.0


def test_call_payoff_matches_formula():
    assert call_payoff(120.0, 100.0) == 20.0
    assert call_payoff(80.0, 100.0) == 0.0


def test_payoff_rejects_unknown_option_type():
    try:
        payoff(100.0, 100.0, option_type="butterfly")
        assert False, "expected ValueError"
    except ValueError:
        pass
