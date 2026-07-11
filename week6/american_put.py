import math
import numpy as np

def crr_put_price(S0, K, T, r, sigma, steps, american=True):
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive for the CRR model")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")

    #CRR parameters
    steps = int(steps)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)

    if not (0.0 < p < 1.0):
        raise ValueError("Invalid risk-neutral probability; increase steps or check inputs")

    # Terminal layer: j up moves and steps-j down moves.
    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)

    # Roll the option values back to today.
    for i in range(steps - 1, -1, -1):
        value = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])

        if american:
            j = np.arange(i + 1)
            stock = S0 * (u ** j) * (d ** (i - j))
            exercise = np.maximum(K - stock, 0.0)
            value = np.maximum(value, exercise)

    return float(value[0])
