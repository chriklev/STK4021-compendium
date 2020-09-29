""" An implementation of the Gibbs sampler from the exampel on page 277 
"""
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

# Define true values of the parameters
theta = np.array([10, 4])
rho = 0.8

cov = np.array(
    [[1, rho],
     [rho, 1]]
)

# Draw (pseudo)random observation
# rand.seed(4021)
y = rand.multivariate_normal(theta, cov)

# From here we pretend like we dont know the true mean \theta of the distribution

n = 100
theta_sampels = np.empty((n, 2))

# First we draw inital values from our uniform prior distribution
lower, upper = 0, 1
theta_sampels[0] = rand.uniform(lower, upper, 2)

# Create samples
var = 1-rho**2
for t in range(1, n):
    theta_sampels[t, 0] = rand.normal(
        y[0] + rho*(theta_sampels[t-1, 1] - y[1]),
        var)
    theta_sampels[t, 1] = rand.normal(
        y[1] + rho*(theta_sampels[t-1, 0] - y[0]),
        var)

print("True thetas:             {}".format(theta))
print("Mean over Gibbs samples: {}".format(
    theta_sampels[int(n/10):].mean(axis=0)))

plt.plot(theta_sampels[:, 0], theta_sampels[:, 1])
plt.scatter(theta_sampels[:, 0], theta_sampels[:, 1], c="red")
plt.show()
