"""
- label encoding
- one hot encoding
- binarization
"""
from sklearn import preprocessing

class CategoricalFeatures:
    """
    df: pandas dataframe
    categorical features: list of columns names ["ord_1", "nom_0", ......]
    encoding_type: label, binary, ohe
    """


    def __init__(self, df, categorical_features, encoding_type, handle_na=True):
        self.df = df
        self.handle_na = handle_na
        #self.output_df = self.df.copy(deep=True)
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.label_encoders = dict() # point the encoder by a dictionary;
        self.binary_encoders = dict()

        # converting all the data into strings and fill nan by some imaginary number(str)
        if handle_na: # stringed and fillna for df;
            for c in self.cat_feats: #
                self.df.loc[:, c] = df.loc[:, c].astype(str).fillna("-99999999")
        self.output_df = self.df.copy(deep=True)

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats])

    def _label_encoding(self): # interial function just return inward this class
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl # label encoder;
        return self.df

    def _label_binarization(self): # this eat much memories, speparate the train set to reduce the consumption of meories per sec;
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl # binarizer encoder
        return self.output_df

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, dataframe): # dataframe is to apply the fitted encoder to other object
        if self.handle_na:
            self.output_df = self.df.copy(deep=True)

            for c in self.cat_feats:
               dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-99999999")
        
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self._one_hot(dataframe[cat_feats].values)

        else:
            raise Exception("Encoding type not understood")
        
if __name__ == "__main__":
    from sklearn import linear_model
    import pandas as pd
    df = pd.read_csv("../input/train_cat.csv")#.head(500)
    df_test = pd.read_csv("../input/test_cat.csv")#.head(500)
    sample = pd.read_csv("../input/sample_submission.csv")

    # for encoding_type="binary"
    #train_idx = df["id"].values # by .head() it will show the these two id differed
    #test_idx = df_test["id"].values

    train_len = len(df)
    #test_len = len(df_test)

    df_test["target"] = -1

    # full_data_encoding:
    full_data = pd.concat([df, df_test]) # concat together then provided universal encoding for both sets;
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(df=full_data, # by TAB type author prefer to cat_feats this macroed name;
                                    categorical_features=cols,
                                    encoding_type="ohe",
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform() # full_data_set first;
    #test_transformed = cat_feats.transform(df_test) # test set the next, which is redundant here;
    
    # Separate the full_data_set back into train set and test set:
    train_df = full_data_transformed[:train_len, :]
    test_df = full_data_transformed[train_len:, :]
    
    #print(train_transformed.head())

    # for encoding_type="binary"
    #train_df = full_data_transformed[full_data_transformed["id"].isin(train_idx)].reset_index(drop=True)
    #test_df = full_data_transformed[full_data_transformed["id"].isin(test_idx)].reset_index(drop=True)

    print(train_df.shape) # dataframe does not count the row of columns name as a row but linux will
    print(test_df.shape)  # so if you check the file.csv by wc -l ../input/train_cat.csv or test_cat.csv the output added one row there;
    
    X = full_data_transformed[:train_len, :]
    X_test = full_data_transformed[train_len:, :]

    clf = linear_model.LogisticRegression()
    clf.fit(X, df.target.values)

    preds = clf.predict_proba(X_test)[:, 1]

    sample.loc[:, "target"] = preds
    sample.to_csv("submission.csv", index=False)