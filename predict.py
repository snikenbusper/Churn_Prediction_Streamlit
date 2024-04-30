import pickle
import pandas as pd
import numpy as np

def load_from_pkl(path):
    return pickle.load(open(path, 'rb'))

def scale(input, scaler):
    return scaler.transform(input)

def model_predict(input, model):
    return model.predict(input)

def main():
    scaler = load_from_pkl("scaler.pkl")
    model = load_from_pkl("xgb.pkl")

    test_scale = [[550, 40, 4, 0, 1, 65000]]  #"CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"

    data = {"CreditScore":[0.5], 'Gender': [1], 'Age':[-0.4], 'Tenure':[1.2], 'Balance':[0.9],
            'NumOfProducts': [0.8], 'HasCrCard':[1], 'IsActiveMember': [1],
            'EstimatedSalary':[0.6], 'France':[1], 'Germany':[0], 'Spain' : [0]}
    
    test_input = pd.DataFrame.from_dict(data)

    print("Scaler result : ", scale(test_scale, scaler))
    print("Model result : ", model_predict(test_input, model))
if __name__ == "__main__":
    main()
