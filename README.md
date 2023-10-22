# Eco Pulse - Carbon Emission Analysis and Prediction

## Overview

Eco Pulse is a web application that provides tools for analyzing, visualizing, and predicting carbon emissions using data from air quality sensors. This project includes a Streamlit-based web interface and preprocessing code in MATLAB. The goal of the project is to raise awareness of environmental issues and provide insights into carbon emissions.



## Features

- Data preprocessing in MATLAB to clean and prepare the air quality dataset.
- Data analysis and visualization using Streamlit and Plotly Express.
- Carbon impact prediction using a Random Forest regression model.
- Sentiment analysis for reasons behind carbon emissions reductions (via the OpenAI API).
- Recommendations for country-wide solutions to reduce carbon emissions (via the OpenAI API).

## Requirements

- Python 3.7+
- MATLAB (for preprocessing, if required)
- Streamlit
- Plotly Express
- Pandas
- NumPy
- Joblib
- Seaborn
- Matplotlib

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/eco-pulse.git
Navigate to the project directory:


cd eco-pulse
Install the required Python packages:



pip install -r requirements.txt
Usage
Data Preprocessing in MATLAB (if needed):

Place your air quality data in the specified path.
Run the MATLAB script for data preprocessing.


Streamlit Web Application:

Run the Streamlit web application:


streamlit run app.py


Interact with the Web Application:

Open your web browser and go to the provided URL.
Explore various features and visualizations.
Use the web interface to make predictions and access recommendations.
Configuration


To access the OpenAI API for sentiment analysis and recommendations, you need to set your API key in the app.py file (in the go_to_country_solutions_page function and other relevant sections).

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Special thanks to the Streamlit and Plotly Express communities for their powerful tools and resources.
OpenAI for providing the API used for sentiment analysis and recommendations.
