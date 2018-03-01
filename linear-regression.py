import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import seaborn
import matplotlib.pyplot as plt


# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# Plot input data
plt.interactive(False)

plt.scatter(X_data, y_data)
#plt.show(block=True)
plt.show()
