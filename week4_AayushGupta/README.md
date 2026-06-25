# Week 4 Assignment – American Put Option Pricing using CRR Model

## Folder Structure

project_folder/

american_put.py → Core CRR implementation for pricing European and American put options

test_american_put.py → Sanity tests and convergence checks for validating the pricing model

week4_report.py → Computes option prices, early exercise premium, convergence table, price surface plot, and exercise boundary plot

figures/

  price_surface.png → Surface plot of American put option price against spot price and time to maturity

  exercise_boundary.png → Plot showing early exercise boundary for American put option

## Files Description

### 1. american_put.py

Implements the Cox-Ross-Rubinstein (CRR) binomial tree model.

Main function:

crr_put_price(S0, K, T, r, sigma, steps, american=True)

This function prices both European and American put options depending on the value of the parameter *american*.

### 2. test_american_put.py

Contains sanity checks for validating implementation.

Tests included:

• American put value is greater than or equal to European put value

• Put option value decreases as stock price increases

• Put option value increases as volatility increases

Also includes convergence check by evaluating option price for different tree depths.

### 3. week4_report.py

Generates final results for the assignment.

Outputs:

• European put option price

• American put option price

• Early exercise premium

• Convergence table

• Price surface plot

• Exercise boundary plot

## How to Run

Step 1:

Run test_american_put.py

This checks correctness of implementation and convergence behaviour.

Step 2:

Run week4_report.py

This generates numerical outputs and saves figures inside the figures folder.

## Dependencies

Python libraries required:

numpy

matplotlib

math

## Notes

The implementation uses the Cox-Ross-Rubinstein binomial tree model for pricing options.

American put pricing includes early exercise checks at every node during backward induction.
