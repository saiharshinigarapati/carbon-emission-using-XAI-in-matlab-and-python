import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
# Define a function to go to the "Country-wide Solutions" page
def go_to_country_solutions():
    st.session_state.page = "Country-wide Solutions"
# Page 1: Home Page
def home_page():
    st.title("Echo Pulse Home Page")
    st.write("Welcome to Echo Pulse! This is the home page.")
    
    if st.button("Go to Carbon Impact Visualization"):
        st.session_state.page = "Carbon Impact Visualization"
    if st.button("Go to Country-wide Solutions"):
        go_to_country_solutions()
    
def go_to_country_solutions_page():
    # Set your OpenAI API key
    api_key = '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'

    # Initialize the OpenAI API client
    openai.api_key = api_key

    # Define the sentiment analysis function using VADER
    def analyze_sentiment_vader(text):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        if sentiment_scores['compound'] >= 0.05:
            return "Positive"
        elif sentiment_scores['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Load the CO2 emissions dataset
    data_path = 'CO2_Emissions_1960-2018.csv'
    df = pd.read_csv(data_path)

    # Get the list of available countries in the dataset
    available_countries = df['Country Name'].unique()

    # Streamlit UI
    st.title("CO2 Emissions Analysis")

    # Dropdown for selecting options
    selected_option = st.selectbox("Select an option:", ["Worldwide CO2 Emission Graph", "Country-wide CO2 Emission Graph with Effecting Features", "Country-wide Solutions to Reduce CO2 Emissions"])

    if selected_option == "Worldwide CO2 Emission Graph":
        st.header("Worldwide CO2 Emission Graph")

        # Transpose the dataset
        data = df.set_index('Country Name').T
        data.index = pd.to_datetime(data.index).year
        data.dropna(axis=1, inplace=True)

        # Create a line chart using Plotly Express
        fig = px.line(data, x=data.index, y=["World"], markers=True)
        fig.update_traces(line_color="red")
        fig.update_yaxes(title_text='CO2 emissions (metric tons per capita)')
        fig.update_xaxes(title_text='Year', rangeslider_visible=False)
        fig.update_layout(legend=dict(title="Country"), showlegend=True,
                        title="CO2 Emissions Worldwide")

        # Display the chart
        st.plotly_chart(fig)

    elif selected_option == "Country-wide CO2 Emission Graph with Effecting Features":
        st.header("Country-wide CO2 Emission Graph with Effecting Features")

        # Create a dropdown widget for selecting a country
        selected_country = st.selectbox("Select a Country:", available_countries)

        # Filter the data for the selected country
        country_data = df[df['Country Name'] == selected_country]

        # Extract the years and CO2 emissions values
        years = country_data.columns[1:].astype(int)
        emissions = country_data.iloc[:, 1:].values.ravel()

        # Find years where CO2 emissions decreased compared to the previous year
        reduction_years = [year for year in years[1:] if emissions[year - years[0]] < emissions[year - years[0] - 1]]

        # Calculate the reduction magnitude for each year (difference in emissions)
        reduction_magnitudes = [emissions[year - years[0] - 1] - emissions[year - years[0]] for year in reduction_years]

        # Sort the reduction years by magnitude (from largest to smallest)
        sorted_years_and_magnitudes = sorted(zip(reduction_years, reduction_magnitudes), key=lambda x: x[1], reverse=True)

        # Get the top 5 years with the most significant reductions
        top_5_reduction_years = sorted_years_and_magnitudes[:5]

        # Create a line plot with reduction years marked (only top 5)
        plt.figure(figsize=(12, 6))
        plt.plot(years, emissions, marker='o', linestyle='-', color='b', label='CO2 Emissions')
        plt.scatter([year for year, _ in top_5_reduction_years], [emissions[year - years[0]] for year, _ in top_5_reduction_years],
                    color='r', marker='o', label='Top 5 Reduction Years')
        plt.vlines([year for year, _ in top_5_reduction_years], ymin=0, ymax=max(emissions), color='r', linestyle='--', label='Top 5 Reduction Years')
        plt.title(f'CO2 Emissions Over Time in {selected_country} with Top 5 Reduction Years Highlighted')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (metric tons per capita)')
        plt.legend()
        plt.grid(True)

        # Print the top 5 reduction years and their magnitudes
        st.pyplot(plt)

        st.subheader(f"Top 5 Reduction Years in {selected_country}:")
        for year, magnitude in top_5_reduction_years:
            st.write(f"Year: {year}, Reduction Magnitude: {magnitude:.2f} metric tons per capita")

        # Generate a prompt for the OpenAI API
        prompt = f"Provide the NER to identify specific entities, such as government policies, technologies, or initiatives related to CO2 reduction in {selected_country} in {year}."

        # Initialize an empty list to store the reasons
        reasons_list = []

        # Generate reasons for the given year using the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,  # Limit the response to 50 tokens (adjust as needed)
        )

        # Append the generated reason to the list
        reasons_list.append(response.choices[0].text.strip())

        # Perform sentiment analysis on the generated reason
        sentiment = analyze_sentiment_vader(reasons_list[0])

        # Print the reason with sentiment analysis
        st.subheader(f"Reason for Reduction (Sentiment: {sentiment}):")
        st.write(reasons_list[0])

    elif selected_option == "Country-wide Solutions to Reduce CO2 Emissions":
        st.header("Country-wide Solutions to Reduce CO2 Emissions")

        # Create a dropdown widget for selecting a country
        selected_country = st.selectbox("Select a Country:", available_countries)

        # Generate a prompt for the OpenAI API
        prompt = f"Provide the list of future implementations by government policies, technologies, or initiatives that can be taken by {selected_country} in the future to further reduce CO2 emissions."

        # Initialize an empty list to store the reasons
        reasons_list = []

        # Generate reasons for the selected country using the OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,  # Increase the max tokens to capture longer responses
        )

        # Append the generated reason to the list
        reasons_list.append(response.choices[0].text.strip())

        # Extract and display the top 5 future implementations
        future_implementations = reasons_list[0].split('\n')
        valid_future_implementations = [impl for impl in future_implementations if impl.strip()]

        st.subheader(f"Future Implementations for {selected_country}:")
        for i, reason in enumerate(valid_future_implementations[:5], start=1):
            sentiment = analyze_sentiment_vader(reason)
            st.write(f"{i}. {reason} (Sentiment: {sentiment})")
        if st.button("Back to Home - Visualization"):
            st.session_state.page = "Home"

        pass    



# Page 2: Carbon Impact Visualization
def carbon_impact_visualization():
    st.title("Carbon Impact Visualization")
    # Load the trained model
    model_filename = 'model.pkl'
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
    if st.button("Back to Home - Visualization"):
         st.session_state.page = "Home"

    pass
# Function to show Home Page
def show_home_page():
    app_container.empty()
    home_page()
# Function to show Carbon Impact Visualization Page
def show_carbon_impact_visualization():
    app_container.empty()
    carbon_impact_visualization()

# Create a Streamlit app
st.set_page_config(page_title="Echo Pulse", page_icon="🌍")
app_container = st.empty()

# Function to handle page switching
def show_page():
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Carbon Impact Visualization":
        carbon_impact_visualization()
    elif st.session_state.page == "Country-wide Solutions":
        go_to_country_solutions_page()  # This function will switch the page



# Clear the previous content before displaying a new page
st.empty()
show_page()
 
if __name__ == "__main__":
    pass
