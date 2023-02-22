import numpy as np
# General imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from decouple import config

from deepchecks.tabular.datasets.classification import iris
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular import Dataset

def train_report():
    # Load Data
    iris_df = iris.load_data(data_format='Dataframe', as_train_test=False)
    label_col = 'target'
    df_train, df_test = train_test_split(iris_df, stratify=iris_df[label_col], random_state=0)

    # Train Model
    rf_clf = RandomForestClassifier(random_state=0)
    rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col])

    # We explicitly state that this dataset has no categorical features, otherwise they will be automatically inferred
    # If the dataset has categorical features, the best practice is to pass a list with their names

    ds_train = Dataset(df_train, label=label_col, cat_features=[])
    ds_test =  Dataset(df_test,  label=label_col, cat_features=[])

    suite = full_suite()
    suite_result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)
    suite_result.save_as_html(config("REPORT_NAME"))

if __name__ == "__main__":
    report = train_report()