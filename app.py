from flask import Flask, render_template, request
from apify_client import ApifyClient
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
from pandas import json_normalize
import json
import pickle
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# Initialize the Flask application
app = Flask(__name__)

# Initialize the ApifyClient with your API token
client = ApifyClient("apify_api_EzWCjdvY8XaRkpwZGu5kKTObIOdcb10BbN4g")

# Load the pre-trained model
model = pickle.load(open("model_f.pkl", "rb"))


@app.route("/")
def index():
    return render_template("k.html")

@app.route("/get_instagram_data", methods=["POST"])
def get_instagram_data():
    # Retrieve the username from the form submission
    username = request.form["username"]
    
    # Prepare the actor input
    run_input = {
        "username": [username],
        "resultsLimit": 1,
    }

    # Run the actor and wait for it to finish
    run2 = client.actor("alexey/instagram-reel-scraper").call(run_input=run_input)

    # Fetch and print actor results from the run's dataset (if there are any)
    data = []
    for item in client.dataset(run2["defaultDatasetId"]).iterate_items():
        data.append(item)

    # Convert JSON data to a pandas dataframe and flatten the response
    df_data = json_normalize(data)
    #df_igtv = json_normalize(df_data.to_dict(orient="records"), record_path="latestIgtvVideos")
    df_igt = json_normalize(df_data.to_dict(orient="records"), record_path="latestPosts")
    df_concat = pd.concat([df_data, df_igt], axis=1)
    

    # Select features for prediction
    features = df_concat[['commentsCount','likesCount','followsCount','followersCount']]
    print(features)
    

    # drop duplicate columns
    features = features.loc[:,~features.columns.duplicated()]

    
    
    # check for missing values in the first row
    if features.iloc[0].isna().any():
# fill missing values with corresponding values from the second row
        #features.iloc[0, features.iloc[0].isnull()] = features.iloc[1, features.iloc[0].isnull()]

        features.iloc[0, features.iloc[0].isnull().values] = features.iloc[1, features.iloc[0].isnull().values].values

        # save only the first row to a new dataframe

    features = features.iloc[[0]]
    print(features)


    # # Impute missing values
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imputed_features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

    # Scale the features
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(imputed_features), columns=imputed_features.columns)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_features)

     # Calculate engagement rate
    engagement_rate = (features['commentsCount'] + features['likesCount']) / features['followersCount']

      # Create bar plot using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Engagement Rate'], [engagement_rate])
    ax.set_ylim([0, 10])
    ax.set_ylabel('Engagement Rate')
    ax.set_title('Instagram Engagement Rate')
    # Convert plot to base64-encoded string
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()

    

    return render_template("k.html", username=username, prediction = prediction,engagement_rate=engagement_rate[0],
                           plot_data=plot_data)



if __name__ == "__main__":
    app.run(debug=True)

