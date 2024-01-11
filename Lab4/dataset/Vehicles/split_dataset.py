import os
import random
import pandas as pd

train_list = list()
test_list = list()

root_dir = "raw"
class_index = 0
for vehicle in os.listdir(root_dir):
    img_list = [i for i in os.listdir(os.path.join(root_dir, vehicle)) if i.endswith(".jpg")]
    random.shuffle(img_list)
    split_num = int(len(img_list) * 0.8)
    for img in img_list[0 : split_num]:
        train_list.append([os.path.join(root_dir, vehicle, img), class_index])
    for img in img_list[split_num : ]:
        test_list.append([os.path.join(root_dir, vehicle, img), class_index])
    class_index += 1

train_list.sort()
test_list.sort()

pd.DataFrame(data=train_list, columns=["Vehicle", "Label"]).to_csv("./train.csv", index=False)
pd.DataFrame(data=test_list, columns=["Vehicle", "Label"]).to_csv("./test.csv", index=False)


