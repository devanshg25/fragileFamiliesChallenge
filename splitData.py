import numpy as np
import pandas as pd

bg = open('background_sorted.csv', 'r')
train = open('train_labels_copy.csv', 'r')
bg_train = open('bg_train.csv', 'w')
bg_test = open('bg_test.csv', 'w')

train_ids = []
na_rows = []
count = 0
for line in train:
    l = line.split(',')
    flag = 1
    for x in l:
        if str(x) == "NA":
           flag = 0
           break
    if flag == 1:
        train_ids.append(int(l[0]))
    else:
        na_rows.append(count)
    count = count + 1

print len(train_ids)
print len(na_rows)
np.save('na_rows.npy', na_rows)

# print train_ids

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


