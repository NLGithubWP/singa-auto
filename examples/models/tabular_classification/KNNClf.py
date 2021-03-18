import numpy as np
import pandas as pd
import json
import pickle
import base64

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from singa_auto.model import TabularClfModel, IntegerKnob, CategoricalKnob, FloatKnob, logger
from singa_auto.model.dev import test_model_class
from singa_auto.constants import ModelDependency


class KNNClf(TabularClfModel):
    '''
    Implements K-Nearest Neighbors Classifier using Pima Indian Diabetes dataset.
    '''

    @staticmethod
    def get_knob_config():
        return {
            'n_neighbors': IntegerKnob(2, 3),
            'p': IntegerKnob(2, 3),
            'metric': CategoricalKnob(['minkowski', 'euclidean']),
        }

    def __init__(self, **knobs):
        self._knobs = knobs
        self.__dict__.update(knobs)
        self._clf = self._build_classifier(self._knobs.get("n_neighbors"),
                                           self._knobs.get("p"),
                                           self._knobs.get("metric"))

    def train(self, dataset_path, features=None, target=None, **kwargs):
        # Record features & target
        self._features = features
        self._target = target

        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        # Encode categorical features
        X = self._encoding_categorical_type(X)

        self._clf.fit(X, y)

        # Compute train accuracy
        score = self._clf.score(X, y)
        logger.log('Train accuracy: {}'.format(score))

    def evaluate(self, dataset_path, **kwargs):
        # Load CSV file as pandas dataframe
        csv_path = dataset_path
        data = pd.read_csv(csv_path)

        # Extract X & y from dataframe
        (X, y) = self._extract_xy(data)

        # Encode categorical features
        X = self._encoding_categorical_type(X)

        accuracy = self._clf.score(X, y)
        return accuracy

    def predict(self, queries):
        queries = [pd.DataFrame(query, index=[0]) for query in queries]
        probs = [self._clf.predict_proba(self._features_mapping(query)).tolist()[0] for query in queries]
        return probs

    def destroy(self):
        pass

    def dump_parameters(self):
        params = {}

        # Put model parameters
        clf_bytes = pickle.dumps(self._clf)
        clf_base64 = base64.b64encode(clf_bytes).decode('utf-8')
        params['clf_base64'] = clf_base64
        params['encoding_dict'] = json.dumps(self._encoding_dict)
        params['features'] = json.dumps(self._features)
        params['target'] = self._target

        return params

    def load_parameters(self, params):
        # Load model parameters
        assert 'clf_base64' in params
        clf_base64 = params['clf_base64']
        clf_bytes = base64.b64decode(clf_base64.encode('utf-8'))

        self._clf = pickle.loads(clf_bytes)
        self._encoding_dict = json.loads(params['encoding_dict'])
        self._features = json.loads(params['features'])
        self._target = params['target']


    def _extract_xy(self, data):
        features = self._features
        target = self._target

        if features is None:
            X = data.iloc[:, :-1]
        else:
            X = data[features]

        if target is None:
            y = data.iloc[:, -1]
        else:
            y = data[target]

        return (X, y)

    def _encoding_categorical_type(self, cols):
        # Apply label encoding for those categorical columns
        cat_cols = list(
            filter(lambda x: cols[x].dtype == 'object', cols.columns))
        encoded_cols = pd.DataFrame({col: cols[col].astype('category').cat.codes \
            if cols[col].dtype == 'object' else cols[col] for col in cols}, index=cols.index)

        # Recover the missing elements 
        encoded_cols = encoded_cols.replace(to_replace=-1, value=np.nan)

        # Generate the dict that maps categorical features to numerical
        encoding_dict = {col: {cat: n for n, cat in enumerate(cols[col].astype('category'). \
            cat.categories)} for col in cat_cols}
        self._encoding_dict = encoding_dict

        return encoded_cols

    def _features_mapping(self, df):
        # Encode the categorical features with pre saved encoding dict
        cat_cols = list(filter(lambda x: df[x].dtype == 'object', df.columns))
        df_temp = df.copy()
        for col in cat_cols:
            df_temp[col] = df[col].map(self._encoding_dict[col])
        df = df_temp
        return df
        

    def _build_classifier(self, n_neighbors, p, metric):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, metric=metric)
        return clf


if __name__ == '__main__':
    test_model_class(model_file_path=__file__,
                     model_class='KNNClf',
                     task='TABULAR_CLASSIFICATION',
                     dependencies={ModelDependency.SCIKIT_LEARN: '0.20.0'},
                     train_dataset_path='data/diabetes_train.csv',
                     val_dataset_path='data/diabetes_val.csv',
                     train_args={
                         'features': [
                             'Pregnancies', 'Glucose', 'BloodPressure',
                             'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction','BMI', 'Age'],
                         'target': 'Outcome'
                     },
                     queries={
                         'Pregnancies': 3,
                         'Glucose': 130,
                         'BloodPressure': 92,
                         'SkinThickness': 30,
                         'Insulin': 90,
                         'DiabetesPedigreeFunction': 1,
                         'BMI': 30.4,
                         'Age': 40
                     })