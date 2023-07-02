import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the CSV file
data = pd.read_csv('data.csv')
values = data['Value'].values

# Calculate skewness and kurtosis
skewness = stats.skew(values)
kurtosis = stats.kurtosis(values)
# Print skewness and kurtosis values
print("Skewness:", skewness)
print("Kurtosis:", kurtosis)
# Plot the data distribution
plt.hist(values, bins=30, density=True, alpha=0.5, label='Data')

# Generate x-values for the PDF plot
x = np.linspace(np.min(values), np.max(values), 100)

# Calculate the PDF using a normal distribution with the same mean and standard deviation as the data
pdf = stats.norm.pdf(x, loc=np.mean(values), scale=np.std(values))

# Plot the PDF
plt.plot(x, pdf, 'r', label='PDF')

# Plot the lines representing skewness and kurtosis
plt.axvline(x=np.mean(values) + skewness * np.std(values), color='g', linestyle='--', label='Skewness')
plt.axvline(x=np.mean(values) + np.sqrt(kurtosis) * np.std(values), color='b', linestyle='--', label='Kurtosis')

plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Data Distribution with PDF')
plt.legend()
plt.show()


