import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#讀取資料集
pd_data = pd.read_csv('weatherAUS.csv')
pd_data.shape
#刪除缺失值
pd_data=pd_data.dropna(how='any')
print(pd_data.shape)
#刪除不相關欄位
drop_columns_list = ['WindGustDir', 'WindDir9am', 'WindDir3pm','Date','Location']
pd_data = pd_data.drop(drop_columns_list, axis=1)
print(pd_data.shape)
pd_data.head()
#change yes/no to 1/0
pd_data['RainToday'].replace({'No':0,'Yes':1},inplace=True)
pd_data['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
pd_data.head()
#Task: Split the data into train and test
train_y = pd_data['RainTomorrow'].head(55000)
test_y= pd_data['RainTomorrow'].tail(1420)
train_x = pd_data.head(55000).drop(['RainTomorrow'], axis=1)
test_x= pd_data.tail(1420).drop(['RainTomorrow'], axis=1)
print(train_y.head())
print(train_x.head())

#Secision Tree
dtree=tree.DecisionTreeClassifier(max_depth=3)
dtree=dtree.fit(train_x,train_y)
dot_data = tree.export_graphviz(dtree, 
                filled=True, 
                feature_names=list(train_x),
                class_names=['No rain','rain'],
                special_characters=True)
graph = graphviz.Source(dot_data)
graph
#不同資料與結果的關聯性
dtree.feature_importances_
#把訓練好的模型套用到測試數據
predict_y = dtree.predict(test_x)
predict_y
#計算訓練數據與測試數據的正確率

acc_log = dtree.score(train_x, train_y)
print('training accuracy: %.5f' % acc_log)
x=accuracy_score(test_y, predict_y)
print('test accuracy: %.5f' % x)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=1)
print('max_depth=3 auc: %.5f' % metrics.auc(fpr, tpr))
#ROC
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(metrics.auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()
plt.show()

tree_train_acc=[]   #訓練模型套用到訓練數據的正確率
tree_test_acc=[]    #訓練模型套用到測試數據的正確率
tree_depth=[]       #不同的max_depth

for i in range (2,20):
    dtree=tree.DecisionTreeClassifier(max_depth=i)
    dtree=dtree.fit(train_x,train_y)
    acc_log = dtree.score(train_x, train_y)
    print('max_depth=%d ' % i,'training accuracy: %.5f' % acc_log)
    
    predict_y = dtree.predict(test_x)    
    X=accuracy_score(test_y, predict_y)
    print('\t\ttest accuracy: %.5f' % X)
    
    tree_train_acc.append(acc_log)
    tree_test_acc.append(X)
    tree_depth.append(i)

plt.plot(tree_depth,tree_train_acc,'b', label="training accuracy")
plt.plot(tree_depth,tree_test_acc,'r', label="test accuracy")
plt.ylabel('accuracy (%)')
plt.xlabel('max depth ')
plt.xticks(np.arange(2, 20, 1))
plt.legend()
plt.show()
best_depth = tree_depth[tree_test_acc.index(max(tree_test_acc))]
print ("max depth: ", best_depth)
print ("best test accuracy: %.5f"% max(tree_test_acc))
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=1)
print('max_depth=7 auc: %.5f' % metrics.auc(fpr, tpr))
# 交叉驗證
scores = cross_val_score(dtree,train_x,train_y,cv=5,scoring='accuracy')

# 計算平均值與標準差
print('average of Cross validation: %.5f'%scores.mean())
print('standard deviation of Cross validation: %.5f'%scores.std(ddof=1))
print("-----------------------------------------------------------------------")
#Logistic Regression
print("Logistic Regression")
logreg = LogisticRegression()
logreg = logreg.fit(train_x, train_y)
predict_y = logreg.predict(test_x)
acc_log = logreg.score(train_x, train_y)
print('training accuracy: %.5f' % acc_log)
predict_y =logreg.predict(test_x)
X=accuracy_score(test_y, predict_y)
print('test accuracy: %.5f' % X)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=1)
print('auc: %.5f' % metrics.auc(fpr, tpr))
print("-----------------------------------------------------------------------")
#KNN
print("KNN")
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
acc_knn = knn.score(train_x, train_y)
print('training accuracy: %.5f' % acc_knn)
predict_y =knn.predict(test_x)
X=accuracy_score(test_y, predict_y)
print('test accuracy: %.5f' % X)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=1)
print('auc: %.5f' % metrics.auc(fpr, tpr))
print("-----------------------------------------------------------------------")
# Gaussian Naive Bayes
print("Gaussian Naive Bayes")
gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
predict_y = gaussian.predict(test_x)
acc_gaussian = gaussian.score(train_x, train_y)
print('training accuracy: %.5f' % acc_gaussian)
predict_y =gaussian.predict(test_x)
X=accuracy_score(test_y, predict_y)
print('test accuracy: %.5f' % X)
#AUC
fpr, tpr, thresholds = metrics.roc_curve(test_y, predict_y, pos_label=1)
print('auc: %.5f' % metrics.auc(fpr, tpr))
print("-----------------------------------------------------------------------")
#MLP
print("MLP")
model = Sequential()
model.add(Dense(100, input_dim=train_x.shape[1], activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
print("開始訓練")
history = model.fit(train_x, train_y, validation_split=0.3, epochs=20, batch_size=100)
loss, accuracy = model.evaluate(train_x, train_y)
print("training accuracy: {:.5f}".format(accuracy))
loss, accuracy = model.evaluate(test_x, test_y)
print("test accuracy: {:.5f}".format(accuracy))
print("-----------------------------------------------------------------------")
#SVM
print("SVM")
model = svm.SVC()
model.fit(train_x, train_y)
print("training accuracy: {:.5f}".format(model.score(train_x, train_y)))
print("test accuracy: {:.5f}".format(model.score(test_x, test_y)))
print("-----------------------------------------------------------------------")
#RandonForest
randomForestModel = RandomForestClassifier(n_estimators=100, criterion = 'gini', random_state=42)
randomForestModel.fit(train_x, train_y)
predicted = randomForestModel.predict(train_x)
print('訓練集: ',randomForestModel.score(train_x,train_y))
print('測試集: ',randomForestModel.score(test_x,test_y))