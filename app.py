import streamlit as st
import pandas as pd
from urllib.parse import urlparse
import model_loader
import numpy as np
from collections import Counter
from math import log2
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load models and scaler
rf_model, nb_model, lr_model, xgb_model, adaboost_model, knn_model, nn_model, scaler = model_loader.load_models()

def calculate_features(domain):
    features = {}

    # Debugging: Print the input domain
    print(f"Input domain: {domain}")

    # Entropy
    p, lns = Counter(domain), float(len(domain))
    entropy = -sum(count / lns * log2(count / lns) for count in p.values())
    features['entropy_domain'] = entropy

    # Other features
    features['qty_dot_domain'] = domain.count('.')
    features['qty_underline_domain'] = domain.count('_')
    features['qty_slash_domain'] = domain.count('/')
    features['qty_questionmark_domain'] = domain.count('?')
    features['qty_equal_domain'] = domain.count('=')
    features['qty_at_domain'] = domain.count('@')
    features['qty_and_domain'] = domain.count('&')
    features['qty_tilde_domain'] = domain.count('~')
    features['qty_percent_domain'] = domain.count('%')
    features['qty_digits_url'] = sum(c.isdigit() for c in domain)
    features['www_present'] = 1 if 'www.' in domain else 0
    features['ratio_digits_letters_domain'] = features['qty_digits_url'] / (len(domain) - features['qty_digits_url']) if (len(domain) - features['qty_digits_url']) > 0 else 0
    features['tld_len'] = len(domain.split('.')[-1]) if '.' in domain else len(domain)
    features['numeric_subdomain_count'] = sum(part.isdigit() for part in domain.split('.'))
    features['domainLen'] = len(domain)
    features['haveDash'] = 1 if '-' in domain else 0
    features['urlLen'] = len(urlparse(domain).path)

    # Debugging: Print each calculated feature
    for key, value in features.items():
        print(f'{key}: {value}')

    # Order features to match training order
    ordered_features = [
        'urlLen', 'qty_questionmark_domain',
        'qty_equal_domain', 'qty_slash_domain', 'qty_dot_domain',
        'haveDash', 'domainLen', 'qty_underline_domain',
        'qty_percent_domain', 'qty_tilde_domain',
        'qty_digits_url', 'www_present',
        'ratio_digits_letters_domain', 'entropy_domain', 'tld_len', 'numeric_subdomain_count'
    ]

    ordered_feature_values = {feature: features.get(feature, 0) for feature in ordered_features}

    return ordered_feature_values

def main():
    st.title('Phishing Detector')

    domain = st.text_input('Enter a domain:', '')

    if domain:

        features = calculate_features(domain)

        # Convert features to DataFrame
        df = pd.DataFrame([features])

        # Standardize the data using the loaded scaler
        df_scaled = scaler.transform(df[list(features.keys())])
        df_scaled = pd.DataFrame(df_scaled, columns=list(features.keys()))

        # Print predictions for each model
        st.write('## Predictions')

        # Create a list to store predictions
        results = []

        # RandomForest
        rf_prediction = rf_model.predict(df_scaled)
        rf_prediction_proba = rf_model.predict_proba(df_scaled)
        results.append({
            'Model': 'RandomForest',
            'Prediction': 'Legitimate' if rf_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': rf_prediction_proba[0][0],
            'Probability - Phishing': rf_prediction_proba[0][1]
        })

        # Naive Bayes
        nb_prediction = nb_model.predict(df_scaled)
        nb_prediction_proba = nb_model.predict_proba(df_scaled)
        results.append({
            'Model': 'Naive Bayes',
            'Prediction': 'Legitimate' if nb_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': nb_prediction_proba[0][0],
            'Probability - Phishing': nb_prediction_proba[0][1]
        })

        # Logistic Regression
        lr_prediction = lr_model.predict(df_scaled)
        lr_prediction_proba = lr_model.predict_proba(df_scaled)
        results.append({
            'Model': 'Logistic Regression',
            'Prediction': 'Legitimate' if lr_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': lr_prediction_proba[0][0],
            'Probability - Phishing': lr_prediction_proba[0][1]
        })

        # XGBoost
        xgb_prediction = xgb_model.predict(df_scaled)
        xgb_prediction_proba = xgb_model.predict_proba(df_scaled)
        results.append({
            'Model': 'XGBoost',
            'Prediction': 'Legitimate' if xgb_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': xgb_prediction_proba[0][0],
            'Probability - Phishing': xgb_prediction_proba[0][1]
        })

        # AdaBoost
        adaboost_prediction = adaboost_model.predict(df_scaled)
        adaboost_prediction_proba = adaboost_model.predict_proba(df_scaled)
        results.append({
            'Model': 'AdaBoost',
            'Prediction': 'Legitimate' if adaboost_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': adaboost_prediction_proba[0][0],
            'Probability - Phishing': adaboost_prediction_proba[0][1]
        })

        # KNN
        knn_prediction = knn_model.predict(df_scaled)
        knn_prediction_proba = knn_model.predict_proba(df_scaled)
        results.append({
            'Model': 'KNN',
            'Prediction': 'Legitimate' if knn_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': knn_prediction_proba[0][0],
            'Probability - Phishing': knn_prediction_proba[0][1]
        })

        # Neural Network
        nn_prediction_prob = nn_model.predict(df_scaled)
        nn_prediction = (nn_prediction_prob > 0.5).astype(int)
        nn_prediction_proba = nn_prediction_prob[0][0] if nn_prediction[0][0] == 0 else 1 - nn_prediction_prob[0][0]

        results.append({
            'Model': 'Neural Network',
            'Prediction': 'Legitimate' if nn_prediction[0] == 0 else 'Phishing',
            'Probability - Legitimate': nn_prediction_proba,
            'Probability - Phishing': 1 - nn_prediction_proba
        })

        # Convert results list to DataFrame and handle N/A
        results_df = pd.DataFrame(results)
        results_df = results_df.fillna('N/A')

        # Display results in a table
        st.table(results_df)

if __name__ == '__main__':
    main()
