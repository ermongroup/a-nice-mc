## Utilities

Contains utility functions used for the algorithm. Here we explain the use of some of them:

### HMC

Includes a TensorFlow implementation of Hamiltonian Monte Carlo (HMC), one of the most popular modern MCMC algorithms for continuous distributions.

### NICE

Supports the construction of a invertible network (such as NICE) using a Keras-style fashion. NICE is the building block for A-NICE-MC.

### Evaluation

Contains methods to evaluate the performance of the MCMC algorithm through samples. Currently, the following evaluation methods are implemented:

- Effective sample size (ESS) - a standard measurement for MCMC algorithms. Higher is better
- Acceptance rate - useful for diagnosing the algorithm. In general we want the acceptance rate to be high, but not too high (25% to 75%).
- Gelman and Rubin's diagnostic - useful for evaluating the behavior of multiple chains. The closer to 1 the better (1.1-1.2 can be considered to be too high) See [this article](http://www.patricklam.org/uploads/3/8/2/6/3826399/convergence_print.pdf) for more details.

### Bootstrap

Contains code to support the bootstrapping prodecure, namely a replay buffer.