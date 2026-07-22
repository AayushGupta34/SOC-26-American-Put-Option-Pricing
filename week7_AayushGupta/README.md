# Week 7 — American Put Options as a Reinforcement Learning Problem

The idea behind this week's work is simple to state but surprisingly deep: when
you hold an American put option, you get to decide *when* to exercise it, and
that decision is genuinely hard. Exercise too early and you throw away time
value; wait too long and the chance to lock in a good payoff can slip away. This
is exactly the kind of "when do I stop?" problem that reinforcement learning is
built for, so the goal here was to reframe the early-exercise decision as a
Markov Decision Process and then see how a few different strategies stack up
against each other.

## What's in the folder

The code is split across three files, one per part of the assignment, plus my
Week 4 model as the benchmark.

- **`part_a_mdp.py`** — Part A. This is where the problem gets set up as an MDP.
  It has the small `make_state` helper and, more importantly, the written-out
  definition of the state, actions, transitions, reward, and discounting. It
  also spells out why the state can't peek at future prices (more on that
  below). Run it and it just prints the definition plus a couple of example
  state vectors so you can see what the agent actually "sees."

- **`part_b_environment.py`** — Part B. The actual `AmericanPutEnv` class lives
  here — the thing the agent interacts with through `reset()` and `step_env()`.
  It also runs the two sanity-check tests and prints five sample episodes so you
  can watch the environment behave.

- **`part_c_policy_comparison.py`** — Part C. This is where the comparison
  happens: four different strategies go head to head, and the file reports how
  much each one earns on average, how often each exercises, and *when* the
  interesting ones pull the trigger. It finishes with a written reflection on
  what it all means.

- **`week4_model.py`** — my Week 4 binomial American-put model, which produces
  the exercise boundary that Part C's reflection keeps referring back to.

One thing worth mentioning: I only wrote the environment once. Part C pulls it
in from Part B with a plain `from part_b_environment import AmericanPutEnv`
rather than copy-pasting it, so there's no risk of two slightly different copies
drifting apart. Part A stands on its own.

## Running it

You'll need Python 3 and NumPy — that's it. If NumPy isn't already installed,
`pip install numpy` sorts it out.

From inside this folder:

```bash
python part_a_mdp.py                 # prints the MDP definition + example states
python part_b_environment.py         # runs the tests + 5 sample episodes
python part_c_policy_comparison.py   # runs the full policy comparison
```

Part C is the one that imports from Part B, so keep the files together in the
same folder and it'll find everything on its own.

## The MDP in a nutshell

If you just want the shape of the thing without opening the code:

- **State** — `[time_fraction, moneyness]`, i.e. `[step/steps, spot/K]`. Just
  two numbers: how far we are through the option's life, and how deep in or out
  of the money we are right now.
- **Actions** — `0 = hold`, `1 = exercise`. That's the whole action space.
- **Transitions** — if you hold, the stock takes one binomial step forward; if
  you exercise, the episode is over.
- **Reward** — you get paid exactly once, when you stop: `max(K - spot, 0)`
  either at exercise or at expiry, and nothing while you're just holding.
- **Discount** — `gamma = exp(-r * dt)` per step, so a rupee later is worth
  slightly less than a rupee now.

### Why the state doesn't cheat

This is the part I want to be careful about, because it's easy to accidentally
build an agent that "sees the future" and then wonder why it looks suspiciously
good. In this environment the agent only ever gets the *current* time and
moneyness. The next price move is drawn inside `step_env` **after** the action
has already been chosen — so there's no way for the decision to lean on
information that hasn't happened yet, and nothing in the state carries the full
price history either. That mirrors the situation a real trader is in: you decide
whether to exercise based on where the price is today, not where it'll be
tomorrow.

## Honest notes

**The sample episodes are nudged toward holding.** In Part B I bias the demo
draws toward "hold" (85%) on purpose. Here's why: the option starts at the money,
so a plain 50/50 coin flip almost always lands on "exercise" at the very first
step, pays out zero, and ends the episode before anything interesting happens.
Five of those in a row tells you nothing. Biasing toward hold lets the sample
episodes actually wander into the situations worth seeing — holding for a while,
running to expiry, exercising while genuinely in the money. Importantly, the
**random policy in Part C is still a true 50/50** — the bias is only for the Part
B demo, not the actual comparison.

**Payoffs are reported undiscounted.** In Part C I compare the raw payoff at the
moment each policy stops, kept consistent across all four so it's an apples-to-
apples comparison.

**The Q-learning agent is deliberately a small, rough sketch** — a 20×30 grid of
state bins, 5000 training episodes, fixed exploration. It's enough to learn the
*character* of a good exercise policy, but not enough to nail it, and the
reflection below is upfront about that.

## Reflection: which policy behaves closest to American-put intuition?

Each of the three fixed baselines fails, but they fail in interestingly
different ways.

**Immediate-exercise** pulls the trigger at step 0 every single time. Since the
option starts at the money, that payoff is exactly zero — it effectively throws
the option in the bin before it has a chance to become worth anything.

**Always-hold-to-expiry** goes to the opposite extreme and never exercises early
at all. That means it quietly discards the entire early-exercise premium and
ends up pricing what is really a European put. It actually posts the *highest*
average payoff of the group here — but that's a bit of a trap, because it earns
that number by refusing to engage with the stopping problem at all, not by
solving it well.

**Random** ignores both moneyness and time, so it tends to bail out in the first
few steps for a near-zero payoff. It's the weakest of the lot, and the
exercise-timing table makes the reason obvious: almost all of its exercises are
crammed into the very first bin.

**Q-learning** is the only one that shows the right *shape*. Like an actual
American-put holder, it does exercise early sometimes — it understands that
stopping early can be the right call. But unlike random, its exercises are
spread out across the whole life of the option instead of piling up at step 0.
That spread is the tell-tale sign that its decisions actually depend on the state
(how much time is left, how deep in the money it is) rather than being blind.

Now for the honest part: Q-learning's average payoff still comes in *below*
always-hold. That's not a failure of the idea so much as a limitation of this
particular implementation. A tabular agent spread across 600 state bins, trained
for only 5000 episodes with a fixed exploration rate, simply doesn't get enough
signal from a reward that only shows up once per episode — so it ends up
exercising a bit too eagerly. The takeaway is that none of these policies is
truly optimal: always-hold wins on raw payoff precisely by ignoring early
exercise, while Q-learning is the only one *structurally capable* of learning the
real, state-dependent exercise boundary — which is exactly the reinforcement-
learning counterpart of the critical-price curve my Week 4 binomial model
produces. With more training and exploration that decays over time, I'd expect
the gap to close.
