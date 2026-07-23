from pricing.binomial import crr_price
from pricing.black_scholes import black_scholes_price


def test_american_not_less_than_european():
    args = dict(S0=100, K=105, T=1.0, r=0.05, sigma=0.25, steps=500)
    euro = crr_price(**args, american=False)
    amer = crr_price(**args, american=True)
    assert amer >= euro


def test_put_value_falls_as_spot_rises():
    low = crr_price(80, 100, 1.0, 0.05, 0.25, 500, american=True)
    high = crr_price(120, 100, 1.0, 0.05, 0.25, 500, american=True)
    assert low > high


def test_more_volatility_is_not_cheaper():
    low_vol = crr_price(100, 100, 1.0, 0.05, 0.15, 500, american=True)
    high_vol = crr_price(100, 100, 1.0, 0.05, 0.35, 500, american=True)
    assert high_vol >= low_vol


def test_binomial_european_converges_to_black_scholes():
    # The binomial tree's European price should approach the closed-form
    # Black-Scholes price as steps grows -- a check that doesn't depend on
    # the tree grading its own convergence.
    args = dict(S0=100, K=100, T=1.0, r=0.05, sigma=0.25)
    bs = black_scholes_price(**args, option_type="put")
    tree = crr_price(**args, steps=1000, american=False)
    assert abs(tree - bs) < 0.01


def test_zero_maturity_returns_intrinsic_value():
    price = crr_price(S0=90, K=100, T=0.0, r=0.05, sigma=0.25, steps=10, american=True)
    assert price == 10.0
