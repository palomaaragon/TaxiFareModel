from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                                  ('stdscaler', StandardScaler())])
        print(dist_pipe)
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        print(time_pipe)

        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                          remainder="drop")
        print(preproc_pipe)
        self.pipe =  Pipeline([('preproc', preproc_pipe),
                                     ('linear_model', LinearRegression())])
        print(self.pipe)

        return self

    def run(self):
        """set and train the pipeline"""
        return self.pipe.fit(self.X,self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        return self.pipeline.score(X_test, y_test)




if __name__ == "__main__":
    from data import get_data
    df = get_data()
    print(len(df))
    from data import clean_data
    df = clean_data(df)
    print(len(df))
    y = df.pop("fare_amount")
    X = df
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train = Trainer(X_train, y_train)
    train.set_pipeline()
    print(train.pipe)
    # train.run()
    # print(train.evaluate(X_test,y_test))
