import os
from sklearn import ensemble
import pandas as pd
from sklearn import preprocessing

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD")) # The int type must be applied, otherwise the .get([FOLD]) is null;
                                   # Very notice;

# recall the k-fold value is 0:4
FOLD_MAPPING = { # It could be complement of the FOLD
    0: [1, 2, 3, 4], # folds saving 0 (The complement of FOLD)
    1: [0, 2, 3, 4], # folds saving 1
    2: [0, 1, 3, 4], # folds saving 2
    3: [0, 1, 2, 4], # folds saving 3
    4: [0, 1, 2, 3]  # folds saving 4
}


if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))] # separate train data and validation data from k-fold dataset;
    valid_df = df[df.kfold==FOLD] # train_df - valid_df = df[kfold] 

    ytrain = train_df.target.values # separate y varaible from dataframes
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1) # separate x variable from dataframes, it better be xtrain;
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1) # xvalid

    valid_df = valid_df[train_df.columns] # Redundancy or coding habbits of Abhishek Thakur?

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist()) # generate the label list(classe_) in type of numpy and converting to python list;
        
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())

        label_encoders.append((c, lbl)) # since the labelEncoder applied columnwise, each column c owns diff encoder.
    
    # training
    clf = ensemble.RandomForestClassifier(n_jobs=-1, verbose=2) # model parameters setting;
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    print(preds)