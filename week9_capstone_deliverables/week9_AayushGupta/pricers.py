"""
Thin `Pricer` wrapper around week4_AayushGupta/american_put.py's
crr_put_price, so the comparison code in compare_all.py can call
`.price(contract)` the same way regardless of which pricing method it's
using (binomial today; would extend the same way for e.g. Longstaff-Schwartz
if a future week added one).
"""

import os
import sys

_WEEK4_DIR = os.path.join(os.path.dirname(__file__), "..", "week4_AayushGupta")
if _WEEK4_DIR not in sys.path:
    sys.path.insert(0, _WEEK4_DIR)

from american_put import crr_put_price  # noqa: E402

from option_contract import OptionContract


class PricingResult:
    def __init__(self, price: float, metadata: dict = None):
        self.price = float(price)
        self.metadata = metadata or {}


class BinomialAmericanPutPricer:
    name = "crr_binomial_american_put"

    def price(self, contract: OptionContract) -> PricingResult:
        price = crr_put_price(
            contract.S0, contract.K, contract.T, contract.r, contract.sigma,
            contract.steps, american=(contract.option_type == "put"),
        )
        return PricingResult(price)
