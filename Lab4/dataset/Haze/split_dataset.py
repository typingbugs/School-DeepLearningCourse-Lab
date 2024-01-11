import os
import pandas as pd
import random

train_list = set()

img_list = [i for i in os.listdir("raw/haze") if i.endswith(".jpg")]
random.shuffle(img_list)
for img in img_list[ : int(len(img_list) * 0.8)]:
    train_list.add(img)
img_list.sort()
data = list()
for img in img_list:
    data.append([img, 1 if img in train_list else 0])

pd.DataFrame(data=data, columns=["Image", "Train"]).to_csv("./split.csv", index=False)