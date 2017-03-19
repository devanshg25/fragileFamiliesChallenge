import numpy as np
import pandas as pd

bg = open('background_filled.csv', 'r')
train = open('train_labels.csv', 'r')
bg_train = open('bg_train.csv', 'w')
bg_test = open('bg_test.csv', 'w')

train_ids = []
count = 0
for line in train:
    if count == 0:
        count = count + 1 
        continue
    l = line.split(',')
    train_ids.append(int(l[0]))
    count = count + 1

count = 0
for line in bg:
    if count == 0:
        count = count + 1
        continue
    l = line.split(',')
    bg_id = int(l[-1])
    if bg_id in train_ids:
        bg_train.write(line+'\n')
    else:
        bg_test.write(line+'\n')
    count = count + 1

bg_train.close()
bg_test.close()


