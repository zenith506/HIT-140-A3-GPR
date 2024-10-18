import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the datasets using your local file paths
dataset1 = pd.read_csv(r'C:\Users\A Plus\Downloads\dataset1.csv')
dataset2 = pd.read_csv(r'C:\Users\A Plus\Downloads\dataset2.csv')
dataset3 = pd.read_csv(r'C:\Users\A Plus\Downloads\dataset3.csv')

# Merging dataset2 and dataset3 on the ID column for correlation analysis
merged_data = pd.merge(dataset2, dataset3, on='ID')

# Data visualization
plt.figure(figsize=(10, 6))
plt.hist([merged_data['C_we'], merged_data['C_wk']], bins=20, label=['Computers (Weekends)', 'Computers (Weekdays)'], alpha=0.7)
plt.title('Distribution of Computer Screen Time on Weekends vs Weekdays')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

correlation_matrix = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk'] + list(dataset3.columns[1:])].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation between Screen Time and Well-being Indicators')
plt.show()

# Defining features and targets
X = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]
y = merged_data[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']]

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Displaying the results
print(f'Mean Squared Error: {mse}')
coefficients = pd.DataFrame(model.coef_, columns=X.columns)
print(coefficients)
