import pandas as pd
import numpy as np
import shutil, os
from multiprocessing import Pool
num_partitions = 10 #number of partitions to split dataframe
num_cores = os.cpu_count() #number of cores on your 
train_df_2015 = pd.read_csv('C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/trainlabels/trainLabels15.csv')
train_df_2015 = train_df_2015.rename(index=str, columns={"image": "id_code", "level": "diagnosis"})
df_test_2015 = pd.read_csv('C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/trainlabels/testLabels15.csv')
df_test_2015 = df_test_2015.rename(index=str, columns={"image": "id_code", "level": "diagnosis"})
train_df_2015 = train_df_2015.append(df_test_2015, ignore_index=True)
label_0 = "C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/label_0"
label_1 = "C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/label_1"
label_2 = "C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/label_2"
label_3 = "C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/label_3"
label_4 = "C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/label_4"

def process(id_code):
    path="C:/Users/hmnsh/repos/resized-2015-2019-blindness-detection-images/gooddy/"+id_code+".jpg"
    label = int(train_df_2015.loc[train_df_2015.id_code == id_code]["diagnosis"])
    print("Run", path)
    if(not(os.path.isfile(path))):
        print("File not exist", id_code)
    else:
        if label == 0:
            shutil.move(path, label_0)
        elif label ==1:
            shutil.move(path, label_1)
        elif label ==2:
            shutil.move(path, label_2)
        elif label ==3:
            shutil.move(path, label_3)
        elif label ==4:
            shutil.move(path, label_4)
    return id_code

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def process_img(df):
    df["id_code"]   = df["id_code"].apply(lambda x: process(x))
    return df