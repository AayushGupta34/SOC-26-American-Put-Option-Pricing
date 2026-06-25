from american_put import crr_put_price
import numpy as np
import matplotlib.pyplot as plt
import math

S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.25
steps = 500

euro = crr_put_price(S0, K, T, r, sigma, steps, american=False)
amer = crr_put_price(S0, K, T, r, sigma, steps, american=True)

print("European Put Price =", euro)
print("American Put Price =", amer)

premium = amer - euro

print("Early Exercise Premium =", premium)

#convergence table
def convergence_table(S0=100, K=100, T=1.0, r=0.05, sigma=0.25):
    rows = []
    for steps in [25, 50, 100, 200, 500, 1000]:
        price = crr_put_price(S0, K, T, r, sigma, steps, american=True)
        rows.append((steps, price))
    return rows

for steps, price in convergence_table():
    print(f"{steps:4d} steps -> {price:.6f}")

#price surface
def price_grid(K=100, r=0.05, sigma=0.25, steps=300):
    spots = np.linspace(60, 140, 41)
    maturities = np.linspace(0.05, 2.0, 40)
    prices = np.zeros((len(maturities), len(spots)))

    for i, T in enumerate(maturities):
        for j, S0 in enumerate(spots):
            prices[i, j] = crr_put_price(S0, K, T, r, sigma, steps, american=True)

    return spots, maturities, prices

spots, maturities, prices = price_grid()
S_mesh, T_mesh = np.meshgrid(spots, maturities)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

ax.plot_surface(S_mesh, T_mesh, prices, cmap="viridis", linewidth=0)

ax.set_xlabel("Spot S")
ax.set_ylabel("Time to maturity T")
ax.set_zlabel("American put price")
ax.set_title("CRR American Put Price Surface")

plt.tight_layout()

plt.savefig("figures/price_surface.png", dpi=160, bbox_inches="tight")

plt.show()

#Exercise Boundary
def crr_put_with_boundary(S0, K, T, r, sigma, steps):
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)
    boundary = []

    for i in range(steps - 1, -1, -1):
        continuation = disc * (p * value[1:i + 2] + (1.0 - p) * value[0:i + 1])
        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = np.maximum(K - stock, 0.0)
        exercise_now = exercise > continuation + 1e-10

        if np.any(exercise_now):
            boundary_stock = float(np.max(stock[exercise_now]))
            boundary.append((i * dt, boundary_stock))

        value = np.maximum(continuation, exercise)

    boundary.reverse()
    return float(value[0]), boundary

price, boundary = crr_put_with_boundary(100, 100, 1.0, 0.05, 0.25, 500)

times = [t for t, _ in boundary]
spots = [s for _, s in boundary]

plt.figure(figsize=(7, 4))

plt.plot(times, spots, marker='.')

plt.axhline(100, color="gray", linestyle="--", linewidth=1, label="Strike K")

plt.xlabel("Time")
plt.ylabel("Highest exercise stock")
plt.title("American Put Exercise Boundary")

plt.legend()

plt.tight_layout()

plt.savefig("figures/exercise_boundary.png", dpi=160, bbox_inches="tight")

plt.show()


print("\nReflection")

print("From the results, early exercise mainly happened when the option became deep in-the-money, meaning the stock price was much lower than the strike price, and especially near expiry. This makes sense because as expiry gets closer, the extra time value of holding the option reduces a lot.\n At that point, exercising early and taking the intrinsic value immediately becomes the better choice. Also, my code is a bit slow at this point which I am hoping to fix soon")
