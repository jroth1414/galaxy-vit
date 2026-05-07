"""Project loss functions, separated from the trainer for unit-testability.

The Dirichlet-Multinomial NLL is the load-bearing loss for the Phase 3
Dirichlet head; T3.4 lands ``dirichlet_multinomial_nll`` here, T3.5
adds the loss-parity test against Zoobot 2.0.
"""
