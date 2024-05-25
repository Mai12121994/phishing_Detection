import pickle
import tensorflow as tf
import numpy as np

# Load models using pickle
def load_models():
    with open('models/random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)

    with open('models/naive_bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)

    with open('models/logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)

    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)


    with open('models/adaboost_model.pkl', 'rb') as f:
        adaboost_model = pickle.load(f)


    with open('models/knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)

    # Load the Neural Network model using Keras
    nn_model = tf.keras.models.load_model('models/neural_network_model.h5')

    # Load the StandardScaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return rf_model, nb_model, lr_model, xgb_model, adaboost_model, knn_model, nn_model, scaler


