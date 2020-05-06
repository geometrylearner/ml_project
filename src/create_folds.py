# The validation is running to rival the over-fitting for any training algrorithmn, then it's absolutly neccessary;
# The cross-validation is also neccessary, since only one partition of dataset into training set and validation set will
# failed in many cases;

# This script is used to create cross-validation dataset as a pathology of data augmentation;
'''
The approach to cross validation here is addting fold index to the dataframe while the dataframes being as validation dataset;
There are five folds in which every has 1/5 as its validation dataset, for every dataframe there exists exactly one fold in which
it become validation data in the 1/5 dataframes in total; so adding a columns of fold for featuring;
So, the initiation of k-fold feature should be -1 or other num diff [0:4];
'''

"""
The algorithmn of sklearn.model_selection.StraitifiedKFold(n_splits = k) is:
create k num cross-validation folds, 
in each fold, the len of validation is $1/k\cdot dim$ of the dim of raw data;
The location of validations in raw data is random if each fold; 
"""


import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df["kfold"] = -1 # evlaued kfold column as -1 as initiation;

    df = df.sample(frac=1).reset_index(drop=True) # shuffle df by sample(frac=1)
    # Only index droped dataframe can apply the Stratified K-Folds cross-validator;

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
    # StratifiedKFold return train index and validation index, that is to divide the dataset into two parts in every fold;
    # for 5 times. Let the dimension of dataset be m, chose 5 element in $A_m^2$ is sufficient;
    count = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
        # train_index and validation_index
        # fold is the count of k-fold
        # split X y together into 5 folds for cross-validation;
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, "kfold"] = fold # add the kfold feature to distinguish which fold the datas belong;
        # the data frame has nothing to change but add the kfold feature to label for validation; 
        count += 1
    print("count=", count)
    df.to_csv("../input/train_folds.csv", index=False)