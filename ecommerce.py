import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor

df = pd.read_csv(r"C:\Users\A S US\OneDrive\Desktop\Ecommerce\Ecommerce_Sales_Data_2024_2025.csv")

print(df.head(10))
print(df.tail(10))
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.nunique())
print(df.describe().loc[['min','max','mean']])
print(df.dtypes)
print(df.shape)
print(df.columns)
# convert order date to date time
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# 1 sales trend analysis
plt.figure(figsize=(12,6))
df.groupby('Order Date')['Sales'].sum().plot()
plt.title('Total Sales Trend over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# 2 sales by region
plt.figure(figsize=(12,6))
region_colors = {
    'South' : 'red',
    'East' : 'yellow',
    'West' : 'orange',
    'North' : 'blue'
}
colors = df['Region'].map(region_colors)

sns.barplot(x= 'Region', y='Sales', data = df, estimator='sum', ci=None, palette=region_colors)
plt.title('Total sales by region')
plt.xticks(rotation=45)
plt.show()

# 3 Top 10 cities by sales
top_cities = df.groupby('City')['Sales'].sum().nlargest(10).reset_index()
plt.figure(figsize=(12,6))
# create a list of colors
colors = sns.color_palette("tab10", 10) # tab10 gives 10 distinct colors
sns.barplot(x = 'City', y = 'Sales', data= top_cities, palette=colors)
plt.title('Top 10 cities by sales')
plt.xticks(rotation = 45)
plt.show()

# 4 Sales and profit by category
plt.figure(figsize=(12,6))
sns.barplot(x = 'Category', y = 'Sales', data= df, estimator='sum', ci=None,color='skyblue', label='Sales')
sns.barplot(x='Category', y='Profit', data=df, estimator='sum', ci=None, alpha=0.6, color='darkblue', label='Profit')
plt.title('Sales and profit by category')
plt.legend()
plt.show()

# 5 Subcategories wise sales distribution
import random
sub_categories = df['Sub-Category'].unique()
# generate random colors for each sub - category
random_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in sub_categories]
# Create a dictionary mapping sub-category -> color
color_dict = dict(zip(sub_categories, random_colors))
plt.figure(figsize=(15,6))
sns.barplot(x='Sub-Category', y= 'Sales', data= df, estimator='sum', ci=None, palette=color_dict)
plt.title('Sub category wise sales distribution')
plt.xticks(rotation=45)
plt.ylabel('Total Sales')
plt.show()

# 6 payment mode distribution
plt.figure(figsize=(8,8))
df['Payment Mode'].value_counts().plot.pie(autopct='%1.1f%%', startangle=120, colors= sns.color_palette('coolwarm'))
plt.title('Payment mode distribution')
plt.ylabel('')
plt.show()

# 7 Relationship between discount and profit
plt.figure(figsize=(12,6))
sns.scatterplot(x= 'Discount', y= 'Profit', data=df, color='darkred')
plt.title('Relation between discount and profit')
plt.show()

#8 Quantity vs sales
plt.figure(figsize=(12,6))
sns.scatterplot(x= 'Quantity', y= 'Sales', data=df, color='darkgreen')
plt.title('Quantity vs Sales')
plt.show()

# 9 top 10 most sold products
top_products = df.groupby('Product Name')['Quantity'].sum().nlargest(10).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(x='Quantity', y= 'Product Name', data=top_products, palette='Blues_r')
plt.title('Top 10 most Sold Products')
plt.show()

# 10 Profit Margin by Region
df['Profit Margin'] = (df['Profit']/df['Sales']) * 100
region_colors = {
    'South' : 'red',
    'East' : 'yellow',
    'West' : 'orange',
    'North' : 'blue'
}
colors = df['Region'].map(region_colors)
plt.figure(figsize=(8,5))
sns.barplot(x='Region', y= 'Profit Margin', data=df,estimator='mean', ci=None, palette=region_colors)
plt.title('Average Profit Margin by Region (%)')
plt.xticks(rotation=45)
plt.show()

df = df.dropna()
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
# Drop columns that are identifiers or not predictive
df = df.drop(['Order ID', 'Order Date', 'Customer Name'], axis = 1)
# separate target and features
target = 'Profit'
X = df.drop(columns=[target])
y= df[target]

label_enc = LabelEncoder()
for col in X.select_dtypes(include='object').columns:
    X[col] = label_enc.fit_transform(X[col])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR()
}
results ={}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = r2_score(y_test, preds) * 100 # convert to percentage
    results[name]=acc
accuracy_df = pd.DataFrame(list(results.items()),columns=["Model","Accuracy (%)"])
accuracy_df = accuracy_df.sort_values(by="Accuracy (%)", ascending = False)
print("Model Performance: ")
print(accuracy_df)

plt.figure(figsize=(10,6))
sns.barplot(x='Accuracy (%)', y = 'Model', data = accuracy_df, palette='coolwarm')
plt.title('Model Accuracy Comparison (R^2 in percentage)')
plt.xlabel('Accuracy (%)')
plt.ylabel('ML Model')
plt.xlim(0,100)
plt.show()






