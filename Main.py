#import Libraries
import streamlit as st 
import pandas as pd 
import streamlit.components.v1 as stc
import matplotlib.pyplot as plt
import plotly.express as px 
import time
from streamlit_option_menu import option_menu
from numerize.numerize import numerize
import plotly.express as px 
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np


#page
st.set_page_config(page_title="Energy Consumption BI Analysis",page_icon="📈",layout="wide")

#Navigation bar
selected=option_menu

# load in energy data
energy_df = pd.read_csv('C:\\Users\\Haneesha\\OneDrive\\Desktop\\Myproject\\owid-energy-data (1).csv')

#kpi's
# Function to calculate KPIs dynamically
def calculate_kpis(start_year, end_year):
    # Filter data based on selected range of years
    filtered_df = energy_df[(energy_df['year'] >= start_year) & (energy_df['year'] <= end_year)]
    
    # Calculate KPIs
    total_countries = len(filtered_df['country'].unique())
    total_years = len(filtered_df['year'].unique())
    total_entries = len(filtered_df)
    
    return total_countries, total_years, total_entries
# Year range selection
start_year, end_year = st.slider('Select Year Range', min_value=energy_df['year'].min(), 
                                 max_value=energy_df['year'].max(), value=(energy_df['year'].min(), 
                                 energy_df['year'].max()))
# Calculate KPIs dynamically based on selected year range
total_countries, total_years, total_entries = calculate_kpis(start_year, end_year)

# Display KPIs
st.subheader('Key Performance Indicators (KPIs)')
st.markdown(f"- Total Countries: {total_countries}")
st.markdown(f"- Total Years: {total_years}")
st.markdown(f"- Total Entries: {total_entries}")

# get only columns relating to energy consumption
consumption_columns = [col for col in energy_df.columns if col.endswith('_consumption')]

# sort consumptions amounts to find the most used energy sources
total_consumption = energy_df[consumption_columns].sum().sort_values(ascending=False)

# retrive top 6 energy sources
top_consumptions = total_consumption.index[1:7].tolist()

# filter for relevant columns
selected_columns = ['iso_code', 'country', 'year', 'gdp', 'population'] + top_consumptions
top_energy_df = energy_df[selected_columns]

top_energy_df = top_energy_df[top_energy_df['year']>=1990]
top_energy_df = top_energy_df[top_energy_df['year']<=2018]

countries = ['United Kingdom', 'United States', 'Germany', 'France', 'India', 'Japan']
top_energy_df = top_energy_df.loc[top_energy_df['country'].isin(countries)]

viz_df = top_energy_df

# Function to plot the data as a line chart
def plot_data(selected_country, selected_consumption, start_year, end_year):
    # Filter data based on selected country and year range
    country_data = viz_df[(viz_df['country'] == selected_country) & 
                          (viz_df['year'] >= start_year) & 
                          (viz_df['year'] <= end_year)]

    # Plotting the trend
    plt.figure(figsize=(10, 6))  # Set the figure size here
    plt.plot(country_data['year'], country_data[selected_consumption], marker='o')
    plt.title(f'{selected_consumption} trend in {selected_country} ({start_year}-{end_year})')
    plt.xlabel('Year')
    plt.ylabel(selected_consumption)
    plt.grid(True)
    return plt

# Function to create and display a pie chart of consumption percentages
def plot_consumption_pie_chart(selected_country, start_year, end_year):
    # Filter data for the selected country and year range
    country_data = viz_df[(viz_df['country'] == selected_country) & 
                          (viz_df['year'] >= start_year) & 
                          (viz_df['year'] <= end_year)]

    # Calculating total consumption for each type
    total_by_type = country_data[['fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 
                                  'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']].sum()
    
    # Pie chart
    plt.figure(figsize=(10, 6))  # Ensure the figure size is the same as the line chart
    plt.pie(total_by_type, labels=total_by_type.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Consumption Breakdown in {selected_country} ({start_year}-{end_year})')
    return plt

# Streamlit app
st.title('World Energy Consumption Trends Dashboard')

# Year range selection
years = sorted(viz_df['year'].unique())
start_year, end_year = st.select_slider(
    'Select Year Range',
    options=years,
    value=(years[0], years[-1])
)

# Creating two columns for the country and consumption type selection
col1, col2 = st.columns(2)

# Country selection in the first column
with col1:
    country_list = viz_df['country'].unique()
    selected_country = st.selectbox('Select a Country', country_list)

# Consumption type selection in the second column
with col2:
    consumption_types = ['fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 
                         'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']
    selected_consumption = st.selectbox('Select Consumption Type', consumption_types)

# Creating two columns for the pie chart and the line chart
col1, col2 = st.columns(2)

# Display the pie chart in the first column
with col1:
    pie_chart = plot_consumption_pie_chart(selected_country, start_year, end_year)
    st.pyplot(pie_chart)

# Display the line chart in the second column
with col2:
    line_chart = plot_data(selected_country, selected_consumption, start_year, end_year)
    st.pyplot(line_chart)
    
    
# Filter the data for the selected country and year range
selected_country_data = energy_df[(energy_df['country'] == selected_country) & 
                                  (energy_df['year'] >= start_year) & 
                                  (energy_df['year'] <= end_year)]


# Create a density plot using Plotly Express
fig = px.density_heatmap(selected_country_data, x='year', y=selected_consumption,
                         title=f'Density Plot of {selected_consumption} in {selected_country}',
                         marginal_x="histogram", marginal_y="histogram")

# Display the density plot
st.plotly_chart(fig)

#stacked area chart

# Select relevant columns
selected_columns = ['year', 'fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']
energy_df = energy_df[selected_columns]

# Filter out rows with missing values
energy_df.dropna(inplace=True)

# Melt the dataframe to convert it to long format suitable for plotting
melted_df = energy_df.melt(id_vars=['year'], var_name='energy_source', value_name='consumption')

# Create a stacked area chart using Plotly Express
fig = px.area(melted_df, x='year', y='consumption', color='energy_source',
              title='Composition of Energy Consumption Over Time',
              labels={'consumption': 'Energy Consumption', 'year': 'Year'},
              hover_name='energy_source',
              width=800, height=500)

# Add a legend and enable hover information
fig.update_layout(legend_title='Energy Source', hovermode='x')

# Display the chart using Streamlit
st.plotly_chart(fig)


#Bar chart 

import pandas as pd
import streamlit as st
import plotly.express as px

# Load the energy data
energy_df = pd.read_csv('C:\\Users\\Haneesha\\OneDrive\\Desktop\\Myproject\\owid-energy-data (1).csv')

# Display the column names to verify which columns are available
st.write(energy_df.columns)

# Since the column names might vary in the dataset, let's manually specify the relevant columns for our visualization
selected_columns = ['gdp','iso_code','population','country', 'year', 'fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']

# Filter the DataFrame based on the selected columns
energy_df = energy_df[selected_columns]

# Streamlit app
st.title('Compare Energy Consumption Between Countries')

# Multi-select for selecting countries
selected_countries = st.multiselect('Select Countries', energy_df['country'].unique())

# Filter data for selected countries
filtered_df = energy_df[energy_df['country'].isin(selected_countries)]

# Group by country and sum the energy consumption for each type
grouped_df = filtered_df.groupby('country').sum().reset_index()

# Melt the dataframe for stacked bar chart
melted_df = grouped_df.melt(id_vars=['country'], var_name='Energy Source', value_name='Consumption')

# Plot the stacked bar chart using Plotly Express
fig = px.bar(melted_df, x='country', y='Consumption', color='Energy Source', barmode='stack', title='Energy Consumption Comparison')
st.plotly_chart(fig, use_container_width=True)

#ScatterPlot
# Select relevant columns
selected_columns = ['iso_code', 'country', 'year', 'gdp', 'population', 'fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']
viz_df = energy_df[selected_columns]

# Filter the data based on user input
def filter_data(selected_country, start_year, end_year):
    filtered_data = viz_df[(viz_df['country'] == selected_country) & 
                           (viz_df['year'] >= start_year) & 
                           (viz_df['year'] <= end_year)]
    return filtered_data

# Streamlit app
st.title('GDP vs. Energy Consumption Scatter Plot')

# Country selection
selected_country = st.selectbox('Select a Country', sorted(viz_df['country'].unique()))

# Filter the data
filtered_data = filter_data(selected_country, start_year, end_year)

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['gdp'], filtered_data['fossil_fuel_consumption'], label='Fossil Fuel Consumption')
plt.scatter(filtered_data['gdp'], filtered_data['renewables_consumption'], label='Renewables Consumption')
plt.xlabel('GDP')
plt.ylabel('Energy Consumption')
plt.title(f'GDP vs. Energy Consumption for {selected_country} ({start_year}-{end_year})')
plt.legend()
plt.grid(True)

# Show plot in Streamlit
st.pyplot(plt)

#geospatial heatmap
# Streamlit app
st.title('World Energy Consumption Trends')


# Create a geospatial heatmap
fig = px.choropleth(
    viz_df, 
    locations='iso_code',
    color='population', 
    hover_name='country',
    animation_frame='year',
    range_color=[0, viz_df['population'].max()],  
    title=f'Population Heatmap ({start_year}-{end_year})'
)

# Display the geospatial heatmap
st.plotly_chart(fig)

#correlation matrix
consumption_columns = [col for col in energy_df.columns if col.endswith('_consumption')]
selected_columns = ['country', 'year'] + consumption_columns
filtered_df = energy_df[selected_columns]
# Remove non-numeric columns
numeric_df = filtered_df.select_dtypes(include=['number'])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

# Streamlit app
st.title('Correlation Heatmap of Energy Consumption')

# Create heatmap
fig = px.imshow(correlation_matrix,
                labels=dict(x='Energy Sources', y='Energy Sources', color='Correlation'),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='viridis')

# Update layout
fig.update_layout(width=800, height=600,
                  title='Correlation Matrix Heatmap',
                  xaxis_title='Energy Sources',
                  yaxis_title='Energy Sources')

# Display heatmap
st.plotly_chart(fig)



# Select relevant columns
selected_columns = ['country', 'year', 'fossil_fuel_consumption', 'oil_consumption', 'coal_consumption', 'gas_consumption', 'low_carbon_consumption', 'renewables_consumption']
energy_df = energy_df[selected_columns]

# Drop rows with missing values
energy_df.dropna(inplace=True)

# Use LabelEncoder to encode country names as numeric values
label_encoder = LabelEncoder()
energy_df['country_code'] = label_encoder.fit_transform(energy_df['country'])

# Define features and target variable
X = energy_df[['country_code', 'year']]  # Features: country code and year
y = energy_df['fossil_fuel_consumption']  # Target variable: fossil fuel consumption (you can change this to any other consumption type)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction function
def predict_consumption(selected_country):
    # Check if the selected country exists in the dataset
    if selected_country in energy_df['country'].unique():
        # Convert the selected country to its corresponding code
        selected_country_code = label_encoder.transform([selected_country])[0]

        # Predict consumption rates for the next year
        next_year = energy_df['year'].max() + 1
        predictions = model.predict([[selected_country_code, next_year]])  # Use the country code instead of the country name
        predicted_consumption = np.round(predictions[0], 2)

        return predicted_consumption
    else:
        return None

# Streamlit app
st.title('Energy Consumption Prediction')

# Country selection
selected_country = st.selectbox('Select a Country', sorted(energy_df['country'].unique()))

# Predict consumption rate
predicted_consumption = predict_consumption(selected_country)

# Display prediction result in a table
if predicted_consumption is not None:
    st.table(pd.DataFrame({'Country': [selected_country], 'Predicted Fossil Fuel Consumption for Next Year': [predicted_consumption]}))
else:
    st.error(f"Selected country '{selected_country}' not found in the dataset.")
