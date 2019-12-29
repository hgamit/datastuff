import tqdm
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


N_SPLITS = 5
TRAIN_CSV_PATH = 'filtered_df.csv'
CLS_CSV_PATH = 'class-descriptions-boxable.csv'

class_names = pd.read_csv('class-descriptions-boxable.csv', header=None, names=["code", "name"])
class_names.drop(["name"], axis=1, inplace=True)
LABEL_MAP = dict(zip(class_names.code, class_names.index))

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_train['LabelIndex'] = df_train.LabelName.map(lambda v: LABEL_MAP[v])


X = []
y = []
image_ids = []

df_group = df_train.groupby('ImageID')
for i, (key, df) in tqdm.tqdm(enumerate(df_group), total=len(df_group)):
    X.append([i])
    ml = np.zeros(601)
    df = df.dropna()
    ml[np.array(df.LabelIndex)-1] = 1
    y.append(ml)
    image_ids.append(key)


random_state = 1234
mskf = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=random_state)

df_train['Fold'] = 0
df_train = df_train.set_index('ImageID')
for f, (train_index, test_index) in enumerate(mskf.split(X, y)):
    for i in tqdm.tqdm(test_index):
        df_train.loc[image_ids[i], 'Fold'] = f

df_train = df_train.reset_index()
df_train.to_csv(f'data/train.ver0.csv', index=False)
