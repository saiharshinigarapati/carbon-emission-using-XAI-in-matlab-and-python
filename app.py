import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model_filename = '/Users/saiharshinigarapati/Desktop/sample1/venv/model.pkl'
loaded_model = joblib.load(model_filename)

# Create a Streamlit app
st.title("Carbon Impact Visualization")

# Sidebar with parameter inputs
st.sidebar.header("Parameters")
param1 = st.sidebar.slider("Non Metanic HydroCarbons concentration", min_value=0.00, max_value=100.00, value=50.00, step=0.01)
param2 = st.sidebar.slider("NOx concentration", min_value=100.0, max_value=3000.0, value=400.00, step=0.01)
param3 = st.sidebar.slider("NO2 concentration", min_value=0.0, max_value=500.0, value=150.00, step=0.01)
param4 = st.sidebar.slider("indium oxide", min_value=-10.0, max_value=100.0, value=50.00, step=0.01)
param5 = st.sidebar.slider("Temperature", min_value=-50.0, max_value=150.0, value=25.00, step=0.01)
param6 = st.sidebar.slider("Relative Humidity", min_value=0.0, max_value=5.0, value=2.50, step=0.01)
param7 = st.sidebar.slider("Absolute humidity", min_value=0.0, max_value=5.0, value=2.50, step=0.01)


# Normalize parameter values to the range [0, 1]
param1_normalized = (param1 - 0) / (100 - 0)
param2_normalized = (param2 - 100) / (3000 - 100)
param3_normalized = (param3 - 0) / (500 - 0)
param4_normalized = (param4 - (-10)) / (100 - (-10))
param5_normalized = (param5 - (-50)) / (150 - (-50))
param6_normalized = (param6 - 0) / (5 - 0)
param7_normalized = (param7 - 0) / (5 - 0)


# Sidebar with radio buttons for seasons
st.sidebar.header("Season")
season_option = st.sidebar.radio("Select Season", ["Autumn", "Spring", "Summer", "Winter"])

# Initialize all season parameters to 0
param8, param9, param10, param11 = 0, 0, 0, 0

# Set the selected season parameter to 1
if season_option == "Autumn":
    param8 = 1
elif season_option == "Spring":
    param9 = 1
elif season_option == "Summer":
    param10 = 1
elif season_option == "Winter":
    param11 = 1

# Function to make predictions using the loaded model
def predict_parameter4(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11):
    new_data = np.array([[param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11]])
    param12 = loaded_model.predict(new_data)
    return param12[0]

# Predict Parameter 4 using user input
predicted_param4 = predict_parameter4(param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11)

# Determine the carbon value threshold and set background color accordingly
carbon_threshold = 500 # Adjust this threshold as needed
if predicted_param4 > carbon_threshold:
    background_color = '#FF5722'  # Red background for high carbon
else:
    background_color = '#1C1C1C'  # Dark background for low carbon

# Set the background color of the Streamlit app
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {background_color};
        color: #FFFFFF;  /* White text on all backgrounds */
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a 3D scatter plot
st.header("3D Scatter Plot")

# Create a DataFrame with the user's parameters and the predicted parameter
scatter_data = pd.DataFrame({
    'Non Metanic HydroCarbons concentration': [param1],
    'NOx concentration': [param2],
    'Carbon (Predicted)': [predicted_param4]
})

# Create a 3D scatter plot using the selected columns
scatter_fig = px.scatter_3d(scatter_data, x='Non Metanic HydroCarbons concentration', y='NOx concentration', z='Carbon (Predicted)', title='3D Scatter Plot')

st.plotly_chart(scatter_fig)
# Create data for the histograms
param_names = [
    "Non Metanic HydroCarbons concentration",
    "NOx concentration",
    "NO2 concentration",
    "indium oxide",
    "Temperature",
    "Relative Humidity",
    "Absolute humidity",
    "Predicted Carbon Value"
]

hist_data = [
    param1,
    param2,
    param3,
    param4,
    param5,
    param6,
    param7,
    predicted_param4
]

# Create histograms
st.header("Histograms for Parameters and Predicted Value")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(param_names, hist_data)
ax.set_xlabel("Parameter Name")
ax.set_ylabel("Value")
ax.set_title("Histograms for Parameters and Predicted Value")
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
st.pyplot(fig)


# Display the predicted carbon value
st.sidebar.header("Carbon Impact")
st.sidebar.text(f"Predicted Carbon Value: {predicted_param4:.2f}")



# Create a DataFrame with normalized parameter values and corresponding gas names
param_normalized_values = {
    'Parameter 1': param1_normalized,
    'Parameter 2': param2_normalized,
    'Parameter 3': param3_normalized,
    'Parameter 4': param4_normalized,
    'Parameter 5': param5_normalized,
    'Parameter 6': param6_normalized,
    'Parameter 7': param7_normalized,
}
param_gas_names = {
    'Parameter 1': 'Non Metanic HydroCarbons concentration',
    'Parameter 2': 'NOx concentration',
    'Parameter 3': 'NO2 concentration',
    'Parameter 4': 'indium oxide',
    'Parameter 5': 'Temperature',
    'Parameter 6': 'Relative Humidity',
    'Parameter 7': 'Absolute humidity',
    'Parameter 8':'carbon',
}
param_df = pd.DataFrame.from_dict(param_normalized_values, orient='index', columns=['Normalized Value'])
param_df['Gas Name'] = param_df.index.map(param_gas_names)

# Create a scatter plot with circle markers and customized settings
fig = px.scatter(param_df, x='Normalized Value', y='Normalized Value', size='Normalized Value', color='Gas Name',
                 title="Dynamic Circle Mark Plot for All Parameters (Normalized)",
                 labels={'Normalized Value': 'Normalized Parameter Value'},
                color_discrete_map={
                     'Non Metanic HydroCarbons concentration': 'blue',
                     'NOx concentration': 'green',
                     'NO2 concentration': 'red',
                     'indium oxide': 'purple',
                     'Temperature': 'orange',
                     'Relative Humidity': 'pink',
                     'Absolute humidity': 'brown',
                     'carbon':'yellow',
                 })
                 # Customize the plot layout (remove axis labels, grid, and parameter names on bubbles)
fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

                 # Set a custom legend title
fig.update_layout(legend_title_text="Gas Names")

# Display the dynamic circle mark plot
st.plotly_chart(fig)
      
# Run the Streamlit app
if __name__ == '__main__':
    st.sidebar.text("To view this app in a browser, run it with the following command:")
    st.sidebar.code(f"streamlit run {__file__}")
