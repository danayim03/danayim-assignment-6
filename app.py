from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # Generate random datasets X and Y with normal error (mean=mu, variance=sigma^2)
    X = np.random.rand(N)
    Y = mu + np.sqrt(sigma2) * np.random.randn(N)

    # Fit linear regression model to X and Y
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Plot scatter of (X, Y) with fitted line
    plt.figure()
    plt.scatter(X, Y, color="blue", label="Data Points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Linear Fit: y = {intercept:.2f} + {slope:.2f}x")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to create histograms of slopes and intercepts
    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = mu + np.sqrt(sigma2) * np.random.randn(N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.6, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.6, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=2.5, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=2.5, label=f"Intercept: {intercept:.2f}")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=4, alpha=0.3)  # Shadow for slope
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=4, alpha=0.3)  # Shadow for intercept
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Compute proportions of slopes and intercepts exceeding baseline values
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        S = int(request.form["S"])
        plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

        return render_template("index.html", plot1=plot1, plot2=plot2, slope_extreme=slope_extreme, intercept_extreme=intercept_extreme)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
