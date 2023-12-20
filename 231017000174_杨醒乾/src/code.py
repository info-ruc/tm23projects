import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def plot_pred_act(act, pred, model, reg_line=True, label=''):
    xy_max = np.max([np.max(act), np.max(pred)])  # 定义输出图像的横纵坐标最大值
    plot = plt.figure(figsize=(6, 6))  # 6 * 6大小的图像
    plt.plot(act, pred, 'o', ms=9, mec='k', mfc='silver', alpha=0.4)  # 设置输出图片的样式（横纵坐标  银色原点 实线  alpha设计图片基色）
    plt.plot([0, xy_max], [0, xy_max], 'k--', label='ideal')  # 同上
    if reg_line:
        polyfit = np.polyfit(act, pred, deg=1)
    reg_ys = np.poly1d(polyfit)(np.unique(act))
    plt.plot(np.unique(act), reg_ys, alpha=0.8, label='linear fit')  # linear fit  线性模型
    plt.axis('scaled')  # 保证每个刻度相同
    plt.xlabel(f'Actual {label}')  # 横坐标
    plt.ylabel(f'Predicted {label}')  # 纵坐标
    plt.title(f'{model}, r2: {r2_score(act, pred):0.4f}')  # 图片名称 分别为 模型名 与 r2数值  #保留小数点后4位
    plt.legend(loc='upper left')  # 图片的位置
    return plot

df = pd.read_csv('combine.csv')
X=df.drop("Eg",1)
df['Eg'] = df['Eg'].astype('float64')
y = np.asarray(df['Eg'])

# 梯度增强回归
from sklearn.ensemble import GradientBoostingRegressor
X_train1, X_test1, y_train1, y_test1 = train_test_split( X, y, test_size=0.1, random_state=9)

gbr = GridSearchCV (GradientBoostingRegressor (),{
    'n_estimators': [2000], 'max_depth': [2], 'min_samples_split': [2], 'learning_rate': [0.1],
    'loss': ['ls'], 'random_state':[72]}, cv=5)

X_train1.drop(columns=['formula'],axis=1,inplace=True)
X_test1.drop(columns=['formula'],axis=1,inplace=True)

gbr.fit(X_train1, y_train1)
y_predicted1 = gbr.predict(X_test1)
gbr_score = gbr.score(X_train1,y_train1)
gbr_score1 = gbr.score(X_test1,y_test1)
plot = plot_pred_act(y_test1, y_predicted1, 'GBR Model', reg_line=True, label='$ (eV/atom)')

# 核岭回归
from sklearn.kernel_ridge import KernelRidge
X_train2, X_test2, y_train2, y_test2 = train_test_split( X, y, test_size=0.1, random_state=9)
krr = GridSearchCV (KernelRidge (),{ 'alpha':[0.001],'kernel':['linear']}, cv=5)
X_train2.drop(columns=['formula'],axis=1,inplace=True)
X_test2.drop(columns=['formula'],axis=1,inplace=True)
krr.fit(X_train2, y_train2)
y_predicted2 = krr.predict(X_test2)
krr_score = krr.score(X_train2,y_train2)
krr_score1 = krr.score(X_test2,y_test2)
plot = plot_pred_act(y_test2, y_predicted2, 'KRR Model', reg_line=True, label='$ (eV/atom)')

# 支持向量机
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X_train3, X_test3, y_train3, y_test3 = train_test_split( X, y, test_size=0.1, random_state=10)
steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid= {'SVM__C':[100], 'SVM__gamma':['auto'], 'SVM__kernel': ['rbf'],
                                           'SVM__epsilon':[0.001]}, cv=5)

X_train3.drop(columns=['formula'],axis=1,inplace=True)
X_test3.drop(columns=['formula'],axis=1,inplace=True)
grid.fit(X_train3, y_train3)
svr_score = grid.score(X_train3,y_train3)
svr_score1 = grid.score(X_test3,y_test3)
y_predicted3 = grid.predict(X_test3)

plot = plot_pred_act(y_test3, y_predicted3, 'SVR Model', reg_line=True, label='$ (eV/atom)')

print('GBR Model| R2 sq on train set: %.4f'% gbr_score)
print('GBR Model| R2 sq on test set: %.4f'% gbr_score1)
print("GBR Model| MSE on test set: %.4f"% mean_squared_error(y_test1, y_predicted1))
print("GBR Model| MAE on test set: %.4f"% mean_absolute_error(y_test1, y_predicted1))
print ("---------------------------------")

print('KRR Model| R2 sq on train set: %.4f'% krr_score)
print('KRR Model| R2 sq on test set: %.4f'% krr_score1)
print('KRR Model| MSE on test set: %.4f'% mean_squared_error(y_test2, y_predicted2))
print('KRR Model| MAE on test set: %.4f'% mean_absolute_error(y_test2, y_predicted2))
print ("---------------------------------")
print('SVR Model| R2 sq on train set: %.4f'% svr_score)
print('SVR Model| R2 sq on test set: %.4f'% svr_score1)
print('SVR Model| MSE on test set: %.4f'% mean_squared_error(y_test3, y_predicted3))
print('SVR Model| MAE on test set: %.4f'% mean_absolute_error(y_test3, y_predicted3))
print ("---------------------------------")
plt.show()