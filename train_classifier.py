import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

data =   np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = model.score(x_test,y_test)
print(score)
print(y_predict)

print('{}% of samples have been classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
