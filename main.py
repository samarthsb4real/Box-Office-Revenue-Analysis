import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, norm
import random

# 1. Overview: Simulate movie dataset with genres, budget, and revenue
np.random.seed(42)
num_movies = 100

genres = ['Action', 'Drama', 'Comedy', 'Horror', 'Romance', 'Sci-Fi']
budgets = np.random.randint(10, 300, size=num_movies) * 1e6  # in millions
revenues = budgets * np.random.uniform(0.5, 3.0, size=num_movies)  # revenue in millions
weeks_top_10 = np.random.randint(1, 15, size=num_movies)

movie_data = pd.DataFrame({
    'Genre': np.random.choice(genres, size=num_movies),
    'Budget ($M)': budgets,
    'Revenue ($M)': revenues,
    'Weeks in Top 10': weeks_top_10
})

# 2. Define revenue categories (Low: < $50M, Medium: $50M-$150M, High: > $150M)
def categorize_revenue(revenue):
    if revenue < 50e6:
        return 'Low'
    elif revenue < 150e6:
        return 'Medium'
    else:
        return 'High'

movie_data['Revenue Category'] = movie_data['Revenue ($M)'].apply(categorize_revenue)

# 3. Probability of each revenue category
revenue_counts = movie_data['Revenue Category'].value_counts(normalize=True)
low_prob, medium_prob, high_prob = revenue_counts['Low'], revenue_counts['Medium'], revenue_counts['High']

# 4. Simulate probability distributions
# Binomial: Weeks in Top 10
n_trials = 10
p_success = 0.3  # arbitrary success probability
binom_dist = binom(n_trials, p_success)

# Poisson: Movies exceeding revenue threshold ($100M)
lambda_ = 100  # mean value for Poisson
poisson_dist = poisson(lambda_)

# Normal: Overall revenue distribution
mean_revenue = np.mean(movie_data['Revenue ($M)'])
std_revenue = np.std(movie_data['Revenue ($M)'])
normal_dist = norm(mean_revenue, std_revenue)

# 5. Plot distributions
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Binomial distribution plot
x_binom = np.arange(0, n_trials + 1)
axs[0].bar(x_binom, binom_dist.pmf(x_binom))
axs[0].set_title('Binomial Distribution: Weeks in Top 10')
axs[0].set_xlabel('Weeks in Top 10')
axs[0].set_ylabel('Probability')

# Poisson distribution plot
x_poisson = np.arange(0, lambda_ + 30)
axs[1].bar(x_poisson, poisson_dist.pmf(x_poisson))
axs[1].set_title('Poisson Distribution: Revenue > $100M')
axs[1].set_xlabel('Movies')
axs[1].set_ylabel('Probability')

# Normal distribution plot
x_norm = np.linspace(mean_revenue - 3*std_revenue, mean_revenue + 3*std_revenue, 100)
axs[2].plot(x_norm, normal_dist.pdf(x_norm))
axs[2].set_title('Normal Distribution: Revenue Spread')
axs[2].set_xlabel('Revenue ($M)')
axs[2].set_ylabel('Density')

plt.tight_layout()
plt.show()

movie_data.head(), revenue_counts
