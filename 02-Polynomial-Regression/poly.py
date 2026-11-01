# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

df = pd.read_csv("auto-mpg.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. PREPROCESS THE DATA
# --------------------------------------------------

# Remove missing values
df.dropna(inplace=True)

# Select input and output
X = df[["displacement"]]
y = df["mpg"]


# --------------------------------------------------
# 3. SPLIT THE DATASET
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. LINEAR REGRESSION
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_linear_pred = linear_model.predict(X_test)

linear_mse = mean_squared_error(y_test, y_linear_pred)
linear_r2 = r2_score(y_test, y_linear_pred)


print("\n========== LINEAR REGRESSION ==========")
print("MSE :", linear_mse)
print("R2 Score :", linear_r2)


# --------------------------------------------------
# 5. POLYNOMIAL REGRESSION
# --------------------------------------------------

degrees = [2, 3, 4, 5]

results = []

for degree in degrees:

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)

    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train linear regression on polynomial features
    poly_model = LinearRegression()

    poly_model.fit(X_train_poly, y_train)

    # Prediction
    y_poly_pred = poly_model.predict(X_test_poly)

    # Evaluation
    mse = mean_squared_error(y_test, y_poly_pred)
    r2 = r2_score(y_test, y_poly_pred)

    results.append([degree, mse, r2])

    print("\n========== POLYNOMIAL REGRESSION ==========")
    print("Degree :", degree)
    print("MSE :", mse)
    print("R2 Score :", r2)


# --------------------------------------------------
# 6. COMPARISON TABLE
# --------------------------------------------------

print("\n========== MODEL COMPARISON ==========")

print("Linear Regression")
print("MSE :", linear_mse)
print("R2 Score :", linear_r2)

for result in results:
    print(
        "Polynomial Degree", result[0],
        "MSE :", result[1],
        "R2 Score :", result[2]
    )


# --------------------------------------------------
# 7. VISUALIZE THE POLYNOMIAL FIT
# --------------------------------------------------

# Use degree 3 for visualization
degrees = [2, 3, 4, 5]

for degree in degrees:

	poly = PolynomialFeatures(degree=degree)

	X_poly = poly.fit_transform(X)

	poly_model = LinearRegression()

	poly_model.fit(X_poly, y)


	# Create smooth X values for plotting
	X_plot = np.linspace(
	    X["displacement"].min(),
	    X["displacement"].max(),
	    300
	).reshape(-1, 1)


	# Transform X_plot into polynomial features
	X_plot_poly = poly.transform(X_plot)

	# Predict MPG
	y_plot = poly_model.predict(X_plot_poly)


	# Plot actual data
	plt.scatter(
	    X,
	    y,
	    color="blue",
	    label="Actual Data"
	)

	# Plot polynomial curve
	plt.plot(
	    X_plot,
	    y_plot,
	    color="red",
	    linewidth=2,
	    label="Polynomial Regression"
	)

	plt.xlabel("Engine Displacement")
	plt.ylabel("Miles Per Gallon (MPG)")
	plt.title("Polynomial Regression: Displacement vs MPG")
	plt.legend()
	plt.grid()
	plt.show()
