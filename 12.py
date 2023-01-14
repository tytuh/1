import streamlit as st
import numpy as np
from scipy.optimize import least_squares

st.title("Least Squares Optimization Example")

# Define the function to optimize
def fun(x, a, b, c):
    return a * np.exp(b * x) + c

# Define the residuals
def fun_residuals(x, y, a, b, c):
    return fun(x, a, b, c) - y

# Create some synthetic data
np.random.seed(42)
x_data = np.linspace(0, 4, 50)
y_data = fun(x_data, 2.5, 1.3, 0.5) + np.random.normal(scale=0.2, size=50)

# Define the initial guess for the parameters
x0 = [1, 1, 1]

# Optimize the parameters using least squares
res = least_squares(fun_residuals, x0, args=(x_data, y_data))

# Plot the data and the optimized function
import matplotlib.pyplot as plt
plt.scatter(x_data, y_data)
plt.plot(x_data, fun(x_data, *res.x))

st.pyplot()