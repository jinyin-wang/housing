# use all previous month data to predict 201505 according to different zip code(e.g. 98001)
# show predicted price and original price
import pickle
import pandas
import numpy as np
if __name__ == '__main__':

    df = pandas.read_csv('predict.csv', low_memory=False)
    df = df.loc[df['zipcode'] == 98001]


    ftrs = ["date_new", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",  "floors",  "waterfront",  "view", "condition",
             "grade", "sqft_basement",  "yr_built"]
    features = df[ftrs]
    ytest = df["price"]
    Xtest= np.array(features)
    print(Xtest)


    filename = "98001.sav"
    regressor = pickle.load(open(filename,'rb'))
    print("predict price")
    print(regressor.predict(Xtest))
    print("original price")
    print(ytest)
