import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
# import statsmodels.formula.api as smf

# Load the data
data = pd.read_csv('hosp_data.csv')


# Custom date parser
def custom_date_parser(date_str):
    try:
        return pd.to_datetime(date_str, format='%m/%d/%Y')
    except ValueError:
        return None


data['Date'] = data['Date'].apply(custom_date_parser)

# Convert time columns to datetime objects
data['Entry Time'] = pd.to_datetime(data['Entry Time'], format='%H:%M:%S')
data['Post-Consultation Time'] = pd.to_datetime(data['Post-Consultation Time'], format='%H:%M:%S')
data['Completion Time'] = pd.to_datetime(data['Completion Time'], format='%H:%M:%S')

# Set 'Date' as the index
data.set_index('Date', inplace=True)
print("----------------------------------")
print(data.head())
print("----------------------------------")
# Group by date and calculate average patients per day
average_patients_per_day = data.groupby('Date')['Patient ID'].count().mean()
print("----------------------------------")
print(f'Average Patients Seen Per Day: {average_patients_per_day:.2f}')

data['Day_of_Week'] = data.index.day_name()

# Count patients per day of the week
average_patients_by_day = data.groupby('Day_of_Week')['Patient ID'].count()
print("----------------------------------")
print(average_patients_by_day)

# Calculate the wait time in minutes
data['Wait Time (Minutes)'] = (data['Post-Consultation Time'] - data['Entry Time']).dt.total_seconds() / 60

# Fix column names: Remove potential leading and trailing spaces
data.columns = data.columns.str.strip()

# Calculate correlations
correlation_1 = data['Medication Revenue'].corr(data['Wait Time (Minutes)'])
print("----------------------------------")
print(correlation_1)

correlation_2 = data['Consultation Revenue'].corr(data['Wait Time (Minutes)'])
print("----------------------------------")
print(correlation_2)

# Scatter plot 1
plt.figure(figsize=(10, 6))
sns.regplot(x='Wait Time (Minutes)', y='Medication Revenue', data=data, scatter_kws={'alpha': 0.5},
            line_kws={'color': 'lightcoral'})
plt.title('Wait Times and Medication Revenue')
plt.xlabel('Wait Time (Minutes)')
plt.ylabel('Medication Revenue')
plt.grid(True)
plt.show()

# Scatter plot 2
plt.figure(figsize=(10, 6))
sns.regplot(x='Wait Time (Minutes)', y='Consultation Revenue', data=data, scatter_kws={'alpha': 0.5},
            line_kws={'color': 'darkred'})
plt.xlabel('Wait Time (Minutes)')
plt.ylabel('Consultation Revenue')
plt.title('Correlation Between Wait Time and Consultation Revenue')
plt.grid(True)
plt.show()

# Scatter plot 2.1
plt.scatter(data['Wait Time (Minutes)'], data['Consultation Revenue'], c=data['Consultation Revenue'],
            cmap='viridis', marker='o', edgecolor='k')
plt.xlabel('Wait Time (Minutes)')
plt.ylabel('Consultation Revenue')
plt.title('Correlation Between Wait Time and Consultation Revenue')
plt.grid(True)

# Regression line for Scatter plot 2.1
slope, intercept, _, _, _ = linregress(data['Wait Time (Minutes)'], data['Consultation Revenue'])
x_vals = np.array([min(data['Wait Time (Minutes)']), max(data['Wait Time (Minutes)'])])
y_vals = slope * x_vals + intercept
plt.plot(x_vals, y_vals, color='lightcoral', linewidth=2, label='Regression Line')
plt.legend()
cbar = plt.colorbar()
cbar.set_label('Consultation Revenue', rotation=90)
plt.show()

# Calculate daily wait time trends and stats
daily_wait_time = data['Wait Time (Minutes)'].resample('D').mean()
print(daily_wait_time.describe())

# Time series plot: Wait Time Trends Over Time
plt.figure(figsize=(12, 6))
plt.plot(daily_wait_time, marker='o', linestyle='-')
plt.title('Wait Time Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Average Wait Time (Minutes)')
plt.grid(True)
plt.show()

# Pie chart plot: Financial Class
# Calculate the counts of each financial class
financial_class_counts = data['Financial Class'].value_counts()

# Convert counts to percentages
financial_class_percentages = (financial_class_counts / financial_class_counts.sum()) * 100

colors = ['#f4f1de', '#e07a5f', '#748cab', '#81b29a', '#f2cc8f', '#ffb5a7']
colors2 = ['#f4f1de', '#e07a5f', '#748cab', '#81b29a', '#f2cc8f', '#ffb5a7']
colors3 = ['#797d62', '#9b9b7a', '#d9ae94', '#f1dca7', '#ffcb69', '#d08c60', '#997b66']
colors4 = ['#b8d4d4', '#d1e4e4', '#f9cdae', '#f5b281', '#e09b83', '#ddc993', '#ccd0b5']

# Explode parameter
explode = [0.1] * len(financial_class_percentages)

# Find the index of 'Medicare'
medicare_index = financial_class_percentages.index.get_loc('MEDICARE')
explode[medicare_index] = 0.3

# Plot the percentages in a pie chart
plt.figure(figsize=(10, 6))
financial_class_percentages.plot.pie(explode=explode, autopct='%1.1f%%', startangle=140, colors=colors, shadow=True)
plt.title('Patients per Financial Class (%)')
plt.ylabel('')
plt.show()

# Group by 'Financial Class' and calculate the average wait time
average_wait_per_financial_class = data.groupby('Financial Class')['Wait Time (Minutes)'].mean()

# Print the results for average wait times per financial class
print("\nAverage Wait Times per Financial Class:")
print("----------------------------------------")
print(average_wait_per_financial_class)
print("\n")

# Plot the results using a bar chart
plt.figure(figsize=(12, 6))
average_wait_per_financial_class.plot(kind='bar', color=colors, edgecolor='black')
plt.title('Average Wait Times per Financial Class')
plt.xlabel('Financial Class')
plt.ylabel('Average Wait Time (Minutes)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Average Wait time Per day of the week and Patients seen during the specific day
# Group by 'Day_of_Week' and calculate average wait time
average_wait_per_day = data.groupby('Day_of_Week')['Wait Time (Minutes)'].mean()

# Group by 'Day_of_Week' and count patient IDs
patient_count_per_day = data.groupby('Day_of_Week')['Patient ID'].count()

# Print the results
print("\nAverage Wait Times per Day of the Week:")
print("----------------------------------------")
print(average_wait_per_day)
print("\nPatient Count per Day of the Week:")
print("----------------------------------")
print(patient_count_per_day)
print("\n")

# Ensure the order is Monday through Sunday
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
average_wait_per_day = average_wait_per_day.reindex(ordered_days)
patient_count_per_day = patient_count_per_day.reindex(ordered_days)

# Plot the results using bar charts
# Plot the average wait times using the modified color list 'colors3'
plt.figure(figsize=(12, 6))
average_wait_per_day.plot(kind='bar', color=colors3, edgecolor='black')
plt.title('Average Wait Times per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Wait Time (Minutes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average wait times
plt.figure(figsize=(12, 6))
average_wait_per_day.plot(kind='bar', color=colors3, edgecolor='black')
plt.title('Average Wait Times per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Wait Time (Minutes)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Patient counts
plt.figure(figsize=(12, 6))
patient_count_per_day.plot(kind='bar', color=colors4, edgecolor='black')
plt.title('Patient Count per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting the combination of "Wait time trend" with the Patient Count
# Calculate daily wait time trends
daily_wait_time = data['Wait Time (Minutes)'].resample('D').mean()

# Calculate daily patient count
daily_patient_count = data['Patient ID'].resample('D').count()

# Create the main plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Time series line graph for average wait time
line, = ax1.plot(daily_wait_time.index, daily_wait_time, 'r-', marker='o',
                 label='Average Wait Time')  # Color changed to red
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Wait Time (Minutes)', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Wait Time Trends and Patient Count Over Time')

# Create a twin axis for the patient count
ax2 = ax1.twinx()

# Plot the patient count as a bar chart on ax2
bars = ax2.bar(daily_patient_count.index, daily_patient_count, alpha=0.6, label='Patient Count', width=0.5)
ax2.set_ylabel('Number of Patients', color='gray')
ax2.tick_params('y', colors='gray')

# Introduce the legends for both ax1 and ax2
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the plot
fig.tight_layout()
plt.show()

# Plotting the combination of "Wait time trend" with the Ave. Patient Per day
# Create the main plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot for average number of patients seen per day of the week
ax1.bar(patient_count_per_day.index, patient_count_per_day, alpha=0.6, color='gray', label='Average Patients Seen')
ax1.set_xlabel('Day of the Week')
ax1.set_ylabel('Average Number of Patients', color='gray')
ax1.tick_params('y', colors='gray')
ax1.set_title('Average Wait Times and Average Patients Seen per Day of the Week')

# Create a twin axis for the average wait time
ax2 = ax1.twinx()

# Line plot for average wait times per day of the week
ax2.plot(average_wait_per_day.index, average_wait_per_day, 'r-', marker='o', label='Average Wait Time')
ax2.set_ylabel('Average Wait Time (Minutes)', color='#f66f65')
ax2.tick_params('y', colors='r')

# Display legends outside the plot
ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.95), borderaxespad=0.)

# Adjust layout to accommodate the external legend and show the plot
fig.tight_layout()
plt.subplots_adjust(right=0.75)
plt.show()

# Deep analysis: Incorporation of the type of doctors assigned during the day into our analysis.
# Data Preprocessing
# Ensure there are no missing values in the 'Doctor_Type' column
data['Doctor Type'].fillna('Unknown', inplace=True)  # Replace NaNs with 'Unknown'

# Aggregate Data
grouped_data = data.groupby(['Day_of_Week', 'Doctor Type']).agg({
    'Wait Time (Minutes)': 'mean',
    'Patient ID': 'count'
}).reset_index()

# Rename columns for clarity
grouped_data.rename(columns={'Wait Time (Minutes)': 'Average Wait Time', 'Patient ID': 'Patient Count'}, inplace=True)

# Visual Analysis
# Using a heatmap to visualize average wait times
pivot_data = grouped_data.pivot(index="Day_of_Week", columns="Doctor Type", values="Average Wait Time")
ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_data = pivot_data.reindex(ordered_days)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_data, cmap="YlGnBu", annot=True, linewidths=.5)
plt.title('Average Wait Times by Day and Doctor Type')
plt.show()

# Define the order of days
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Print patient counts for each doctor type
for index, row in grouped_data.iterrows():
    print(f"On {row['Day_of_Week']}, Doctor Type: {row['Doctor Type']} saw {row['Patient Count']} patients.")

# Reorder the dataframe by days
grouped_data['Day_of_Week'] = pd.Categorical(grouped_data['Day_of_Week'], categories=days_order, ordered=True)
grouped_data = grouped_data.sort_values('Day_of_Week')

# Using FacetGrid to show patient counts for each doctor type
g = sns.FacetGrid(grouped_data, col="Doctor Type", col_wrap=4, height=4, sharey=False)
g = g.map(plt.bar, "Day_of_Week", "Patient Count", color='skyblue', edgecolor='black').set_titles("{col_name}").set_xticklabels(rotation=45)
plt.tight_layout()
plt.show()

# Colors for each doctor type
doctor_colors = {
    'ANCHOR': '#b8d4d4',
    'FLOATING': '#f9cdae',
    'LOCUM': '#f66f65'
}

# Using a Stacked Bar Chart to represent patient counts segmented by doctor type
plt.figure(figsize=(14, 8))
bottom_data = None
for doctor in grouped_data['Doctor Type'].unique():
    subset = grouped_data[grouped_data['Doctor Type'] == doctor].set_index('Day_of_Week')['Patient Count']
    subset = subset.reindex(ordered_days)
    plt.bar(subset.index, subset, bottom=bottom_data, label=doctor, color=doctor_colors[doctor])
    if bottom_data is None:
        bottom_data = subset
    else:
        bottom_data += subset
plt.title('Patient Counts per Day of the Week by Doctor Type')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.legend(title='Doctor Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Running Multiple Regression Analysis to determine factors that influence wait times
# Encoding categorical variables, while avoiding the dummy variable trap
hdata_dummies = pd.get_dummies(data[['Doctor Type', 'Day_of_Week']], drop_first=True)

# Prepare the data for regression
X = pd.concat([hdata_dummies, data['Medication Revenue'], data['Consultation Revenue']], axis=1)
y = data['Wait Time (Minutes)']

# Convert boolean columns to integer type
X[hdata_dummies.columns] = X[hdata_dummies.columns].astype(int)

# Displaying the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Predictors')
plt.show()

# Run the regression
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Feature Engineering for daily analysis
features_daily = pd.get_dummies(data[['Doctor Type', 'Day_of_Week', 'Medication Revenue', 'Consultation Revenue']])
target = data['Wait Time (Minutes)']

# Train-Test Split for daily analysis
X_train_daily, X_test_daily, y_train_daily, y_test_daily = train_test_split(features_daily, target, test_size=0.3, random_state=42)

# Build and Evaluate the Regression Model for daily analysis
model_daily = LinearRegression()
model_daily.fit(X_train_daily, y_train_daily)
y_pred_daily = model_daily.predict(X_test_daily)
mae_daily = mean_absolute_error(y_test_daily, y_pred_daily)
print("----------------------------------")
print(f"Daily Mean Absolute Error: {mae_daily}")

# Determine days that might be understaffed using a threshold
THRESHOLD = 30  # For example, if we want wait times to be less than 30 minutes
understaffed_days = (y_pred_daily > THRESHOLD).sum()
print("----------------------------------")
print(f"Number of days predicted to be understaffed: {understaffed_days}")

# Now, for the hourly analysis:
# Extract hour from our data's time columns (assuming 'Entry Time' represents the time patients come in)
data['Hour'] = data['Entry Time'].dt.hour

# Group by hour and date, then calculate the mean wait time
hourly_data = data.groupby([data.index.date, data['Hour']])['Wait Time (Minutes)'].mean().reset_index()

# Feature engineering for hourly data
hourly_data['Day_of_Week'] = pd.to_datetime(hourly_data['level_0']).dt.day_name()
features_hourly = pd.get_dummies(hourly_data[['Day_of_Week', 'Hour']])

# Train-Test Split for hourly analysis
X_train_hourly, X_test_hourly, y_train_hourly, y_test_hourly = train_test_split(features_hourly, hourly_data['Wait Time (Minutes)'], test_size=0.3, random_state=42)

# Build and Evaluate the Regression Model for hourly analysis
model_hourly = LinearRegression()
model_hourly.fit(X_train_hourly, y_train_hourly)
y_pred_hourly = model_hourly.predict(X_test_hourly)
mae_hourly = mean_absolute_error(y_test_hourly, y_pred_hourly)
print("----------------------------------")
print(f"Hourly Mean Absolute Error: {mae_hourly}")

# Determine hours that might be understaffed using the threshold
understaffed_hours = (y_pred_hourly > THRESHOLD).sum()
print("----------------------------------")
print(f"Number of hours predicted to be understaffed: {understaffed_hours}")