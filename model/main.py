import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
import pickle


def clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32','id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1 , 'B':0})
    return data

def create_model(data):
    # creating dependent and independent variable
    x = data.drop(['diagnosis'], axis =1)
    y = data['diagnosis']

    # scaling the data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # splitting the data
    x_train,x_test,y_train,y_test = train_test_split(x , y ,test_size=0.2 , random_state=42)

    # create Model
    model = LogisticRegression()
    model.fit(x_train,y_train)

    # test model
    y_pred = model.predict(x_test)
    print("Accuracy score : ",accuracy_score(y_test,y_pred))
    print("Classification report :\n ",classification_report(y_test,y_pred))
    return model , scaler

def main():
    data = clean_data()
    model , scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

if __name__== '__main__':
    main()
# print(df.head())