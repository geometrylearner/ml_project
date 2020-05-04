import os
import numpy as np

from sklearn import metrics
from sklearn import ensemble
import pandas as pd
from sklearn import preprocessing
import joblib

from . import dispatcher

TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL") # The value of MODEL comes from the shebang script run.sh where MODEL=$1 that is the first variable after run.sh in shell;
TRAINING_DATA = os.environ.get("TRAINING_DATA")




def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values

    predictions = None



    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoders.pkl"))
        for c in encoders:
            print(c)
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5
    # training

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['id', 'target'] )

    return sub

if __name__ == "__main__":
    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"models/{MODEL}.csv", index=False)