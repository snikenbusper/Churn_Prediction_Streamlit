import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle


class Data():
    def __init__(self, dir):
        self.dir = dir
        self.dataset = None
        self.train = None
        self.test = None
        self.output_col = None
        self.y = None
        self.x = None
        self.scaler = None

    def load(self):
        self.dataset = pd.read_csv(self.dir)

    def col_median(self, col):
        return self.dataset[col].median()

    def remove_col(self, col):
        self.dataset = self.dataset.drop(col, axis=1)

    def fill_na(self, col, fill):
        self.dataset[col].fillna(fill)

    def set_output_col(self, col):
        self.output_col = col
        self.y = self.dataset[self.output_col]
        self.x = self.dataset.drop([self.output_col], axis=1)


    def one_hot_encode(self, col):
        self.dataset = self.dataset.join(pd.get_dummies(self.dataset[col]).astype("int")).drop(col, axis=1)
    

    def label_encode(self, col, mapping):
        self.dataset[col] = self.dataset[col].map(mapping)

    def train_test_split(self,test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size)
        self.train = [x_train, y_train]
        self.test = [x_test, y_test]

    def scale_numeric(self, cols):
        self.scaler = StandardScaler()
        self.scaler.fit(self.train[0][cols])
        
        self.train[0].loc[:, cols] = self.scaler.transform(self.train[0][cols])
        self.test[0].loc[:, cols] = self.scaler.transform(self.test[0][cols])

    def save_scaler(self, path):
        pickle.dump(self.scaler, open(path, 'wb'))


    def get_train(self):
        return self.train[0], self.train[1]

    def get_test(self):
        return self.test[0], self.test[1]

    
        
class Model():
    def __init__(self, df):
        self.df = df
        self.model = None
        self.preds = None

    def init_model(self, eta=0.3, n_estimators=100, max_depth=None):
        self.model = XGBClassifier(eta=eta, n_estimators=n_estimators, max_depth = max_depth)
    
    def train(self):
        self.model.fit(*self.df.get_train())

    def predict(self):
        self.preds = self.model.predict(self.df.get_test()[0])

    def class_report(self):
        return classification_report(self.df.get_test()[1], self.preds)

    def roc_auc(self):
        return roc_auc_score(self.df.get_test()[1], self.preds)

    def grid_search(self, params):
        self.model = GridSearchCV(XGBClassifier(), params)

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))




data = Data("data_C.csv")
data.load()
data.remove_col("Unnamed: 0")
data.remove_col("id")
data.remove_col("CustomerId")
data.remove_col("Surname")
data.fill_na("CreditScore", data.col_median("CreditScore"))
data.one_hot_encode("Geography")
data.label_encode("Gender", {"Male":0, "Female":1})


data.set_output_col("churn")
data.train_test_split(0.2)
data.scale_numeric(["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"])


model = Model(data)


print("Initial Model")
model.init_model()
model.train()
model.predict()
print("Classification Report")
print(model.class_report())
print("ROC AUC : ", model.roc_auc())


print("Tune Parameter")
model.grid_search({"eta" : [0.1,0.9], "max_depth":[None,5], "n_estimators" : [25, 100]})
model.train()
model.predict()
print("Classification Report")
print(model.class_report())
print("ROC AUC : ", model.roc_auc())


model.save_model("xgb.pkl")
data.save_scaler("scaler.pkl")
