import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
# README: in this code, we take in the raw data of battery pulse discharge from the excel file
# and fits the data to curves. These curves differ because they have different varible values, such 
# as differrent duty cycle. After the curve fit, we find mathmetical expression on these curves,
# and use linear regression to find the variable's influence on curve parameters.
"""
DC_ENABLE = False
PERIOD_ENABLE = True
CURRENT_ENABLE = False


if DC_ENABLE:
    """
    # data read-in starts here:
    """
    # Load the Excel file
    file_path = 'battery performance.xlsx'
    sheet_name = 'duty cycle'

    # Read the first sheet with 8 columns
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=range(8))

    # Function to remove infs and NaNs from data
    def clean_data(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        return x[mask], y[mask]

    # Extract and clean x and y data points for 4 curves
    x1, y1 = clean_data(data.iloc[:, 0], data.iloc[:, 1])
    x2, y2 = clean_data(data.iloc[:, 2], data.iloc[:, 3])
    x3, y3 = clean_data(data.iloc[:, 4], data.iloc[:, 5])
    x4, y4 = clean_data(data.iloc[:, 6], data.iloc[:, 7])

    """
    # curve fit starts here:

    """
    # Define the function to fit
    def func(x, a, b, c, d, e):
        return a * x + b + c * np.exp(d * x + e)

    # Objective function for minimize
    def objective(params, x, y):
        a, b, c, d, e = params
        y_pred = func(x, a, b, c, d, e)
        return np.sum((y - y_pred) ** 2)

    # Initial guesses for the parameters
    initial_params1 = [-0.001, 2.8, -0.001, 0.08, -6.2]
    initial_params2 = [-0.001, 2.82, -0.001, 0.08, -9]
    initial_params3 = [-0.001, 3, -0.001, 0.08, -11]
    initial_params4 = [-0.0001, 3, -0.001, 0.09, -12]

    # Fit each curve using minimize
    result1 = minimize(objective, initial_params1, args=(x1, y1))
    result2 = minimize(objective, initial_params2, args=(x2, y2))
    result3 = minimize(objective, initial_params3, args=(x3, y3))
    result4 = minimize(objective, initial_params4, args=(x4, y4))

    # Extract optimized parameters
    params1 = result1.x
    params2 = result2.x
    params3 = result3.x
    params4 = result4.x

    # Print results
    if params1 is not None:
        print(f"Parameters for Curve 1: a={params1[0]:.5f}, b={params1[1]:.3f}, c={params1[2]:.20f}, d={params1[3]:.3f}, e={params1[4]:.3f}")
    if params2 is not None:
        print(f"Parameters for Curve 2: a={params2[0]:.5f}, b={params2[1]:.3f}, c={params2[2]:.20f}, d={params2[3]:.3f}, e={params2[4]:.3f}")
    if params3 is not None:
        print(f"Parameters for Curve 3: a={params3[0]:.5f}, b={params3[1]:.3f}, c={params3[2]:.20f}, d={params3[3]:.3f}, e={params3[4]:.3f}")
    if params4 is not None:
        print(f"Parameters for Curve 4: a={params4[0]:.5f}, b={params4[1]:.3f}, c={params4[2]:.20f}, d={params4[3]:.3f}, e={params4[4]:.3f}")

    # Plot the raw and fitted curves
    plt.figure(figsize=(14, 10))

    # Curve 1
    plt.subplot(2, 2, 1)
    plt.scatter(x1, y1, label='Raw Data')
    if params1 is not None:
        plt.plot(x1, func(x1, *params1), label='Fitted Curve', color='red')
    plt.title('Curve 1')
    plt.legend()

    # Curve 2
    plt.subplot(2, 2, 2)
    plt.scatter(x2, y2, label='Raw Data')
    if params2 is not None:
        plt.plot(x2, func(x2, *params2), label='Fitted Curve', color='red')
    plt.title('Curve 2')
    plt.legend()

    # Curve 3
    plt.subplot(2, 2, 3)
    plt.scatter(x3, y3, label='Raw Data')
    if params3 is not None:
        plt.plot(x3, func(x3, *params3), label='Fitted Curve', color='red')
    plt.title('Curve 3')
    plt.legend()

    # Curve 4
    plt.subplot(2, 2, 4)
    plt.scatter(x4, y4, label='Raw Data')
    if params4 is not None:
        plt.plot(x4, func(x4, *params4), label='Fitted Curve', color='red')
    plt.title('Curve 4')
    plt.legend()

    plt.tight_layout()
    plt.show()

    """
    # linear regression on variable starts here:
    """
    """
    result shows that curve 4 is introducing noise (in other word, this curve cann be model the same as the rest curves)
    so we only use curve 1, 2, 3 to do linear regression.
    """
    # Define DC values for the four curves
    #DC_values = np.array([1/9, 1/14, 1/24, 1/74])  #four curve version
    DC_values = np.array([1/9, 1/14, 1/24 ])        #three curve version
    params = np.array([
        params1,  # From curve 1
        params2,  # From curve 2
        params3,  # From curve 3
        #params4   # From curve 4
    ])

    # Create a dataframe for DC and fitted parameters
    data = pd.DataFrame({
        'DC': DC_values,
        'a': params[:, 0],
        'b': params[:, 1],
        'c': params[:, 2],
        'd': params[:, 3],
        'e': params[:, 4]
    })

    print("Data for DC and parameters:")
    print(data)

    # Function to fit DC to parameters using polynomial regression
    def fit_parameter_to_DC(DC, parameter, degree=2):
        # Transform DC for polynomial regression
        poly = PolynomialFeatures(degree)
        DC_transformed = poly.fit_transform(DC.reshape(-1, 1))

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(DC_transformed, parameter)

        # Return the model and coefficients
        return model, model.coef_, model.intercept_

    # Fit and plot for each parameter
    parameters = ['a', 'b', 'c', 'd', 'e']
    plt.figure(figsize=(14, 10))

    for i, param in enumerate(parameters):
        # Fit the parameter to DC
        model, coeffs, intercept = fit_parameter_to_DC(data['DC'].values, data[param].values, degree=2)

        # Generate predictions
        DC_fit = np.linspace(min(DC_values), max(DC_values), 100)
        DC_fit_transformed = PolynomialFeatures(2).fit_transform(DC_fit.reshape(-1, 1))
        param_fit = model.predict(DC_fit_transformed)

        # Plot results
        plt.subplot(3, 2, i + 1)
        plt.scatter(data['DC'], data[param], label=f'Raw {param} Data')
        plt.plot(DC_fit, param_fit, label=f'Fitted Curve for {param}', color='red')
        plt.title(f'{param} vs DC')
        plt.xlabel('DC')
        plt.ylabel(param)
        plt.legend()

        # Print the fitted equation
        print(f"Fitted equation for {param}: {intercept:.3f} + {coeffs[1]:.3f}*DC + {coeffs[2]:.3f}*DC^2")

    plt.tight_layout()
    plt.show()

elif PERIOD_ENABLE:
    """
    # data read-in starts here:
    """
    # Load the Excel file
    file_path = 'battery performance.xlsx'
    sheet_name = 'period'

    # Read the first sheet with 8 columns
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=range(8))

    # Function to remove infs and NaNs from data
    def clean_data(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        return x[mask], y[mask]

    # Extract and clean x and y data points for 4 curves
    x1, y1 = clean_data(data.iloc[:, 0], data.iloc[:, 1])
    x2, y2 = clean_data(data.iloc[:, 2], data.iloc[:, 3])
    x3, y3 = clean_data(data.iloc[:, 4], data.iloc[:, 5])
    x4, y4 = clean_data(data.iloc[:, 6], data.iloc[:, 7])

    """
    # curve fit starts here:

    """
    # Define the function to fit
    def func(x, a, b, c, d, e):
        return a * x + b + c * np.exp(d * x + e)

    # Objective function for minimize
    def objective(params, x, y):
        a, b, c, d, e = params
        y_pred = func(x, a, b, c, d, e)
        return np.sum((y - y_pred) ** 2)

    # Initial guesses for the parameters
    initial_params1 = [-0.001, 2.8, -0.001, 0.08, -6.2]
    initial_params2 = [-0.0021, 2.62, -0.001, 0.065, -6]
    initial_params3 = [-0.0024, 2.59, -0.018, 0.068, -9]
    initial_params4 = [-0.0029, 2.55, -0.18, 0.09, -11.3]

    # Fit each curve using minimize
    result1 = minimize(objective, initial_params1, args=(x1, y1))
    result2 = minimize(objective, initial_params2, args=(x2, y2))
    result3 = minimize(objective, initial_params3, args=(x3, y3))
    result4 = minimize(objective, initial_params4, args=(x4, y4))

    # Extract optimized parameters
    params1 = result1.x
    params2 = result2.x
    params3 = result3.x
    params4 = result4.x

    # Print results
    if params1 is not None:
        print(f"Parameters for Curve 1: a={params1[0]:.5f}, b={params1[1]:.3f}, c={params1[2]:.20f}, d={params1[3]:.3f}, e={params1[4]:.3f}")
    if params2 is not None:
        print(f"Parameters for Curve 2: a={params2[0]:.5f}, b={params2[1]:.3f}, c={params2[2]:.20f}, d={params2[3]:.3f}, e={params2[4]:.3f}")
    if params3 is not None:
        print(f"Parameters for Curve 3: a={params3[0]:.5f}, b={params3[1]:.3f}, c={params3[2]:.20f}, d={params3[3]:.3f}, e={params3[4]:.3f}")
    if params4 is not None:
        print(f"Parameters for Curve 4: a={params4[0]:.5f}, b={params4[1]:.3f}, c={params4[2]:.20f}, d={params4[3]:.3f}, e={params4[4]:.3f}")

    # Plot the raw and fitted curves
    plt.figure(figsize=(14, 10))

    # Curve 1
    plt.subplot(2, 2, 1)
    plt.scatter(x1, y1, label='Raw Data')
    if params1 is not None:
        plt.plot(x1, func(x1, *params1), label='Fitted Curve', color='red')
    plt.title('Curve 1')
    plt.legend()

    # Curve 2
    plt.subplot(2, 2, 2)
    plt.scatter(x2, y2, label='Raw Data')
    if params2 is not None:
        plt.plot(x2, func(x2, *params2), label='Fitted Curve', color='red')
    plt.title('Curve 2')
    plt.legend()

    # Curve 3
    plt.subplot(2, 2, 3)
    plt.scatter(x3, y3, label='Raw Data')
    if params3 is not None:
        plt.plot(x3, func(x3, *params3), label='Fitted Curve', color='red')
    plt.title('Curve 3')
    plt.legend()

    # Curve 4
    plt.subplot(2, 2, 4)
    plt.scatter(x4, y4, label='Raw Data')
    if params4 is not None:
        plt.plot(x4, func(x4, *params4), label='Fitted Curve', color='red')
    plt.title('Curve 4')
    plt.legend()

    plt.tight_layout()
    plt.show()

    """
    # linear regression on variable starts here:
    """
    """
    result shows that curve 4 is introducing noise (in other word, this curve cann be model the same as the rest curves)
    so we only use curve 1, 2, 3 to do linear regression.
    """
    # Define DC values for the four curves
    #DC_values = np.array([1/9, 1/14, 1/24, 1/74])  #four curve version
    PERIOD_values = np.array([75,25,15,10])        #three curve version
    params = np.array([
        params1,  # From curve 1
        params2,  # From curve 2
        params3,  # From curve 3
        params4   # From curve 4
    ])

    # Create a dataframe for DC and fitted parameters
    data = pd.DataFrame({
        'PERIOD': PERIOD_values,
        'a': params[:, 0],
        'b': params[:, 1],
        'c': params[:, 2],
        'd': params[:, 3],
        'e': params[:, 4]
    })

    print("Data for PERIOD and parameters:")
    print(data)

    # Function to fit PERIOD to parameters using polynomial regression
    def fit_parameter_to_PERIOD(PERIOD, parameter, degree=2):
        # Transform PERIOD for polynomial regression
        poly = PolynomialFeatures(degree)
        PERIOD_transformed = poly.fit_transform(PERIOD.reshape(-1, 1))

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(PERIOD_transformed, parameter)

        # Return the model and coefficients
        return model, model.coef_, model.intercept_

    # Fit and plot for each parameter
    parameters = ['a', 'b', 'c', 'd', 'e']
    plt.figure(figsize=(14, 10))

    for i, param in enumerate(parameters):
        # Fit the parameter to PERIOD
        model, coeffs, intercept = fit_parameter_to_PERIOD(data['PERIOD'].values, data[param].values, degree=2)

        # Generate predictions
        PERIOD_fit = np.linspace(min(PERIOD_values), max(PERIOD_values), 100)
        PERIOD_fit_transformed = PolynomialFeatures(2).fit_transform(PERIOD_fit.reshape(-1, 1))
        param_fit = model.predict(PERIOD_fit_transformed)

        # Plot results
        plt.subplot(3, 2, i + 1)
        plt.scatter(data['PERIOD'], data[param], label=f'Raw {param} Data')
        plt.plot(PERIOD_fit, param_fit, label=f'Fitted Curve for {param}', color='red')
        plt.title(f'{param} vs PERIOD')
        plt.xlabel('PERIOD')
        plt.ylabel(param)
        plt.legend()

        # Print the fitted equation
        print(f"Fitted equation for {param}: {intercept:.3f} + {coeffs[1]:.3f}*PERIOD + {coeffs[2]:.3f}*PERIOD^2")

    plt.tight_layout()
    plt.show()

    
    
    
    
elif CURRENT_ENABLE:
    """
    # data read-in starts here:
    """
    # Load the Excel file
    file_path = 'battery performance.xlsx'
    sheet_name = 'drain current'

    # Read the first sheet with 8 columns
    data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=range(8))

    # Function to remove infs and NaNs from data
    def clean_data(x, y):
        mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
        return x[mask], y[mask]

    # Extract and clean x and y data points for 4 curves
    x1, y1 = clean_data(data.iloc[:, 0], data.iloc[:, 1])
    x2, y2 = clean_data(data.iloc[:, 2], data.iloc[:, 3])
    x3, y3 = clean_data(data.iloc[:, 4], data.iloc[:, 5])
    x4, y4 = clean_data(data.iloc[:, 6], data.iloc[:, 7])

    """
    # curve fit starts here:

    """
    # Define the function to fit
    def func(x, a, b, c, d, e):
        return a * x + b + c * np.exp(d * x + e)

    # Objective function for minimize
    def objective(params, x, y):
        a, b, c, d, e = params
        y_pred = func(x, a, b, c, d, e)
        return np.sum((y - y_pred) ** 2)

    # Initial guesses for the parameters
    
    initial_params1 = [-0.0001, 2.8, -0.0005, 0.01, -3]
    initial_params2 = [-0.002, 2.6, -0.004, 0.062, -7]
    initial_params3 = [-0.003, 2.5, -0.03, 0.068, -10]
    initial_params4 = [-0.004, 2.3, -1.2, 0.078, -10]
    
    # Fit each curve using minimize
    result1 = minimize(objective, x0=initial_params1,  args=(x1, y1))
    result2 = minimize(objective, x0=initial_params2, args=(x2, y2))
    result3 = minimize(objective, x0=initial_params3, args=(x3, y3))
    result4 = minimize(objective, x0=initial_params4, args=(x4, y4))

    # Extract optimized parameters
    params1 = result1.x
    params2 = result2.x
    params3 = result3.x
    params4 = result4.x

    # Print results
    if params1 is not None:
        print(f"Parameters for Curve 1: a={params1[0]:.5f}, b={params1[1]:.3f}, c={params1[2]:.20f}, d={params1[3]:.3f}, e={params1[4]:.3f}")
    if params2 is not None:
        print(f"Parameters for Curve 2: a={params2[0]:.5f}, b={params2[1]:.3f}, c={params2[2]:.20f}, d={params2[3]:.3f}, e={params2[4]:.3f}")
    if params3 is not None:
        print(f"Parameters for Curve 3: a={params3[0]:.5f}, b={params3[1]:.3f}, c={params3[2]:.20f}, d={params3[3]:.3f}, e={params3[4]:.3f}")
    if params4 is not None:
        print(f"Parameters for Curve 4: a={params4[0]:.5f}, b={params4[1]:.3f}, c={params4[2]:.20f}, d={params4[3]:.3f}, e={params4[4]:.3f}")

    # Plot the raw and fitted curves
    plt.figure(figsize=(14, 10))

    # Curve 1
    plt.subplot(2, 2, 1)
    plt.scatter(x1, y1, label='Raw Data')
    if params1 is not None:
        plt.plot(x1, func(x1, *params1), label='Fitted Curve', color='red')
    plt.title('Curve 1')
    plt.legend()

    # Curve 2
    plt.subplot(2, 2, 2)
    plt.scatter(x2, y2, label='Raw Data')
    if params2 is not None:
        plt.plot(x2, func(x2, *params2), label='Fitted Curve', color='red')
    plt.title('Curve 2')
    plt.legend()

    # Curve 3
    plt.subplot(2, 2, 3)
    plt.scatter(x3, y3, label='Raw Data')
    if params3 is not None:
        plt.plot(x3, func(x3, *params3), label='Fitted Curve', color='red')
    plt.title('Curve 3')
    plt.legend()

    # Curve 4
    plt.subplot(2, 2, 4)
    plt.scatter(x4, y4, label='Raw Data')
    if params4 is not None:
        plt.plot(x4, func(x4, *params4), label='Fitted Curve', color='red')
    plt.title('Curve 4')
    plt.legend()

    plt.tight_layout()
    plt.show()

    """
    # linear regression on variable starts here:
    """
    """
    result shows that curve 4 is introducing noise (in other word, this curve cann be model the same as the rest curves)
    so we only use curve 1, 2, 3 to do linear regression.
    """
    # Define CURRENT values for the four curves
    
    CURRENT_values = np.array([10, 30, 50, 80 ])        #three curve version
    params = np.array([
        params1,  # From curve 1
        params2,  # From curve 2
        params3,  # From curve 3
        params4   # From curve 4
    ])

    # Create a dataframe for DC and fitted parameters
    data = pd.DataFrame({
        'CURRENT': CURRENT_values,
        'a': params[:, 0],
        'b': params[:, 1],
        'c': params[:, 2],
        'd': params[:, 3],
        'e': params[:, 4]
    })

    print("Data for CURRENT and parameters:")
    print(data)

    # Function to fit DC to parameters using polynomial regression
    def fit_parameter_to_CURRENT(CURRENT, parameter, degree=2):
        # Transform DC for polynomial regression
        poly = PolynomialFeatures(degree)
        CURRENT_transformed = poly.fit_transform(CURRENT.reshape(-1, 1))

        # Fit a linear regression model
        model = LinearRegression()
        model.fit(CURRENT_transformed, parameter)

        # Return the model and coefficients
        return model, model.coef_, model.intercept_

    # Fit and plot for each parameter
    parameters = ['a', 'b', 'c', 'd', 'e']
    plt.figure(figsize=(14, 10))

    for i, param in enumerate(parameters):
        # Fit the parameter to DC
        model, coeffs, intercept = fit_parameter_to_CURRENT(data['CURRENT'].values, data[param].values, degree=2)

        # Generate predictions
        CURRENT_fit = np.linspace(min(CURRENT_values), max(CURRENT_values), 100)
        CURRENT_fit_transformed = PolynomialFeatures(2).fit_transform(CURRENT_fit.reshape(-1, 1))
        param_fit = model.predict(CURRENT_fit_transformed)

        # Plot results
        plt.subplot(3, 2, i + 1)
        plt.scatter(data['CURRENT'], data[param], label=f'Raw {param} Data')
        plt.plot(CURRENT_fit, param_fit, label=f'Fitted Curve for {param}', color='red')
        plt.title(f'{param} vs CURRENT')
        plt.xlabel('CURRENT')
        plt.ylabel(param)
        plt.legend()

        # Print the fitted equation
        print(f"Fitted equation for {param}: {intercept:.3f} + {coeffs[1]:.3f}*CURRENT + {coeffs[2]:.3f}*CURRENT^2")

    plt.tight_layout()
    plt.show()
   
