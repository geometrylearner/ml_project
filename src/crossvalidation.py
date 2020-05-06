from sklearn import model_selection
import pandas as pd

"""
PROBLEM(MODEL) TYPES:
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""


class CrossValidation:
    def __init__(self,
                 df,
                 target_cols,
                 shuffle,
                 problem_type="binary_classification",
                 multilabel_delimiter=",", # this default option must be applied otherwise error;
                 num_folds=5,
                 random_state=42
                 ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter
        
        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True) # shuffle df by sample(frac=1)

        self.dataframe["kfold"] = -1

    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number targets for this problem type")
            target = self.target_cols[0] # take the element as string;
            unique_values = self.dataframe[target].nunique()

            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                target = self.target_cols[0]
                kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

                for fold, (_ , val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)): # apply "_" to substitute the unused variable;
                    self.dataframe.loc[val_idx, "kfold"] = fold
        
        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and problem_type == "single_col_regression":
                raise Exception("Invalid number targets for this problem type")
            if self.num_targets < 2 and problem_type == "multi_col_regression":
                raise Exception("Invalid number targets for this problem type")
            target = self.target_cols[0]
            kf = model_selection.KFold(n_splits=self.num_folds)
            
            for fold, (_ , val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)): # apply "_" to substitute the unused variable;
                    self.dataframe.loc[val_idx, "kfold"] = fold
        
        elif self.problem_type.startswith("holdout_"): # embed cross-validation info into the name of problem type by string "_X" directly;
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0 # first 20 percent as validation dataset;
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1 # last 80 percent as training dataset;

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter))) # number of splitted value in target
            kf = model_selection.KFold(n_splits=self.num_folds) 
            
            for fold, (_, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)): # it's a trick to take the num of spltted target value as y parameter here
                                                                                        # the y= is to randomize the fold separation virtually;
                self.dataframe.loc[val_idx, "kfold"] = fold

        else:
            raise Exception("Problem type not understood")

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("../input/train_multilabel.csv")
    cv = CrossValidation(df, 
                        target_cols = ["attribute_ids"], 
                        shuffle=True,
                        problem_type="multilabel_classification",
                        multilabel_delimiter=" "
                        )
    #df = pd.read_csv("../input/train_reg.csv")
    #cv = CrossValidation(df, target_cols=["SalePrice"])
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts()) # counts of unique values, that is kfold=0,1,2,3,4 for 60000 times