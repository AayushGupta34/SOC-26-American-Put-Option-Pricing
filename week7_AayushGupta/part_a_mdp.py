import numpy as np

def make_state(step, steps, spot, strike):
    """
    State features exposed to the agent.

    Returns a length-2 vector [time_fraction, moneyness]:
      - time_fraction = step / steps   (how far through the option's life, 0..1)
      - moneyness      = spot / strike  (how deep in/out of the money we are)

    Both are CURRENT, normalized quantities. No future price and no full price
    history is included — see the "no leakage" note in MDP_DEFINITION below.
    """
    time_fraction = step / steps
    moneyness = spot / strike
    return np.array([time_fraction, moneyness], dtype=np.float32)


MDP_DEFINITION = """
PART A — MDP DEFINITION  (S, A, P, R, gamma)
============================================

1. STATE (S)
   state = [time_fraction, moneyness] = [step/steps, spot/K].
   Only current, normalized quantities are exposed. The contract parameters
   (r, sigma, T) are fixed for a single contract, so they are constant and
   omitted from the state; they would be added only if training across many
   different contracts at once.

2. ACTIONS (A)
   0 = HOLD    (keep the option alive)
   1 = EXERCISE (stop now and take the immediate payoff)

3. TRANSITIONS (P)
   - After HOLD: the underlying takes one risk-neutral binomial step,
       spot -> spot * u   with probability p
       spot -> spot * d   with probability 1 - p
     and the step counter advances. If that step reaches expiry, the episode
     terminates (see reward below).
   - After EXERCISE: the episode ends immediately (absorbing terminal state).

4. REWARD (R) AND DISCOUNTING (gamma)
   Reward is paid EXACTLY ONCE, only when the episode stops:
     - EXERCISE at step t : reward = max(K - spot_t, 0)
     - HOLD               : reward = 0 on that step (option still alive)
     - EXPIRY (forced)    : reward = max(K - spot_T, 0)  (auto-exercise if ITM)
   The one-step discount factor is gamma = exp(-r * dt), applied inside the
   Bellman/Q-learning target: target = reward + gamma * max_a' Q(s', a').

5. WHY THE STATE DOES NOT LEAK FUTURE INFORMATION
   The agent only ever observes (t/T, S_t/K) at the CURRENT step. The next
   price move is drawn INSIDE the environment's step function, AFTER the action
   has already been chosen, so the action cannot depend on it. There is no field
   in the state that encodes any future price or path history — only where we
   are right now (time left, moneyness). This exactly matches a real option
   holder's information set: you decide whether to exercise using today's price,
   never tomorrow's.
"""


if __name__ == "__main__":
    print(MDP_DEFINITION)

    # quick demonstration of the state helper
    print("Example state features (make_state):")
    print("  at start   (step=0,  spot=100, K=100):", make_state(0, 50, 100, 100))
    print("  mid-life   (step=25, spot=90,  K=100):", make_state(25, 50, 90, 100))
    print("  near expiry(step=49, spot=80,  K=100):", make_state(49, 50, 80, 100))
