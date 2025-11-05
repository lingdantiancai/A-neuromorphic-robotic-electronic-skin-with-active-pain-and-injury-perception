from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os,re
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


features = []
labels = []

def convert_location(type,value):
    if type == 'AM':
        if value == 2.0:
            return 1
        if value == 2.7:
            return 2    
        if value == 3.9:        
            return 3
        if value == 6.2:
            return 4
    if type == 'PW':
        if value == 1.2:
            return 1
        if value == 1.8:
            return 2
        if value == 2.4:
            return 3
        if value == 3.0:
            return 4
    if type == 'PS':
        if value == 1:
            return 1
        if value == 220:
            return 2
        if value == 680:
            return 3
        if value == 1000:
            return 4
    if type == 'FQ':
        if value == 1.5:
            return 1
        if value == 2.0:
            return 2
        if value == 3.0:
            return 3

rootDir = 'training_data_PCB'
for dirName, subdirList, fileList in os.walk(rootDir):
    print(f'Found directory: {dirName}')
    for filename in fileList:
        if filename.endswith('.txt'):  # Check if the file has a .txt extension
            with open(f'./training_data_PCB/{filename}', 'r') as f:
                for line in f:
                    data = line.strip().split(',')
                    features.append([float(data[2][4:])])
                    match = re.search(r"AM_([0-9.]+)k", filename)
                    labels.append(convert_location('AM',float(match.group(1))))


# plt.plot(labels, features)
plt.scatter(labels, features)
plt.show()
# 划分训练集和测试集
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

# 创建 SVM 分类器
clf = svm.SVC()

# 训练分类器
clf.fit(features_train, labels_train)

# 进行预测
predictions = clf.predict(features_test)

# 计算准确率
accuracy = accuracy_score(labels_test, predictions)

matrix = confusion_matrix(labels_test, predictions)
matrix_percentage = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]*100


# 打印准确率
print(f"准确率: {accuracy}")
print (matrix_percentage)
dump(clf, 'AM_svm_model_PCB-test.joblib')