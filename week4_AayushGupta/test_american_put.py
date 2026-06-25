from american_put import crr_put_price

#check if european price less than equal to american
def test_american_put_not_less_than_european():
    args = dict(S0=100, K=105, T=1.0, r=0.05, sigma=0.25, steps=500)
    euro = crr_put_price(**args, american=False)
    amer = crr_put_price(**args, american=True)
    assert amer >= euro

#check if put value falls as stock price rises
def test_put_value_falls_as_spot_rises():
    low_spot = crr_put_price(80, 100, 1.0, 0.05, 0.25, 500, american=True)
    high_spot = crr_put_price(120, 100, 1.0, 0.05, 0.25, 500, american=True)
    assert low_spot > high_spot

#check volatility effects
def test_more_volatility_is_not_cheaper():
    low_vol = crr_put_price(100, 100, 1.0, 0.05, 0.15, 500, american=True)
    high_vol = crr_put_price(100, 100, 1.0, 0.05, 0.35, 500, american=True)
    assert high_vol >= low_vol

def convergence_table(S0=100, K=100, T=1.0, r=0.05, sigma=0.25):
    rows = []
    for steps in [25, 50, 100, 200, 500, 1000]:
        price = crr_put_price(S0, K, T, r, sigma, steps, american=True)
        rows.append((steps, price))
    return rows

test_american_put_not_less_than_european()
test_put_value_falls_as_spot_rises()
test_more_volatility_is_not_cheaper()

print("All sanity tests passed")

for steps, price in convergence_table():
    print(f"{steps:4d} steps  ->  {price:.6f}")
