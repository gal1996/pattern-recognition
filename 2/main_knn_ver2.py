#独学でdeeplearningを勉強しており、そこで使用しているmnistデータを用いる
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score



def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

#xには各ピクセルのデータが1*784の形で1万個入っており、tにはその正解ラベルが入っている.
x, t = get_data()
#xの最後の列にtの値を入れる
x = np.insert(x, 784, t, axis=1)

#クラス分けする配列の初期化
class0 = np.empty((1,785), dtype=np.int)
class1 = np.empty((1,785), dtype=np.int)
class2 = np.empty((1,785), dtype=np.int)
class3 = np.empty((1,785), dtype=np.int)
class4 = np.empty((1,785), dtype=np.int)
class5 = np.empty((1,785), dtype=np.int)
class6 = np.empty((1,785), dtype=np.int)
class7 = np.empty((1,785), dtype=np.int)
class8 = np.empty((1,785), dtype=np.int)
class9 = np.empty((1,785), dtype=np.int)
#クラスに分ける
for i in range(x.shape[0]):
    if(x[i][x.shape[1]-1] == 0):
        class0 = np.append(class0, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 1):
        class1 = np.append(class1, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 2):
        class2 = np.append(class2, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 3):
        class3 = np.append(class3, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 4):
        class4 = np.append(class4, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 5):
        class5 = np.append(class5, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 6):
        class6 = np.append(class6, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 7):
        class7 = np.append(class7, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 8):
        class8 = np.append(class8, [x[i]], axis=0)
    elif(x[i][x.shape[1]-1] == 9):
        class9 = np.append(class9, [x[i]], axis=0)
#1行目のデータは初期化に用いた余計な値なので削除する
class0 = np.delete(class0, 0, axis=0)
class1 = np.delete(class1, 0, axis=0)
class2 = np.delete(class2, 0, axis=0)
class3 = np.delete(class3, 0, axis=0)
class4 = np.delete(class4, 0, axis=0)
class5 = np.delete(class5, 0, axis=0)
class6 = np.delete(class6, 0, axis=0)
class7 = np.delete(class7, 0, axis=0)
class8 = np.delete(class8, 0, axis=0)
class9 = np.delete(class9, 0, axis=0)

#テストデータの形成
mnist_test = np.empty((1,785))

#それぞれのクラスから100個ずつ適当にデータを取ってきて、mnist_testに代入する
#まず抽出する行のインデクスをランダムに生成
id0 = np.random.choice(class0.shape[0], 100, replace=False)
id1 = np.random.choice(class1.shape[0], 100, replace=False)
id2 = np.random.choice(class2.shape[0], 100, replace=False)
id3 = np.random.choice(class3.shape[0], 100, replace=False)
id4 = np.random.choice(class4.shape[0], 100, replace=False)
id5 = np.random.choice(class5.shape[0], 100, replace=False)
id6 = np.random.choice(class6.shape[0], 100, replace=False)
id7 = np.random.choice(class7.shape[0], 100, replace=False)
id8 = np.random.choice(class8.shape[0], 100, replace=False)
id9 = np.random.choice(class9.shape[0], 100, replace=False)

mnist_test = np.append(mnist_test, class0[id0,:], axis=0)
mnist_test = np.append(mnist_test, class1[id1,:], axis=0)
mnist_test = np.append(mnist_test, class2[id2,:], axis=0)
mnist_test = np.append(mnist_test, class3[id3,:], axis=0)
mnist_test = np.append(mnist_test, class4[id4,:], axis=0)
mnist_test = np.append(mnist_test, class5[id5,:], axis=0)
mnist_test = np.append(mnist_test, class6[id6,:], axis=0)
mnist_test = np.append(mnist_test, class7[id7,:], axis=0)
mnist_test = np.append(mnist_test, class8[id8,:], axis=0)
mnist_test = np.append(mnist_test, class9[id9,:], axis=0)
mnist_test = np.delete(mnist_test, 0, axis=0)

#==============
#ホールドアウト法
#==============
#テストデータを250個、テストデータを750個として推定を行う

accuracy = np.empty((1,20))
#knn法の実装
for i in range(1,11):
    accuracy_k = np.empty(1)
    for j in range(20):
        id_hold_test = np.random.choice(1000, 250, replace=False)
        #学習データのインデクスの生成
        id_hold_train = np.array([i for i in range(1000) if i not in id_hold_test])
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(mnist_test[id_hold_train, 0:783], mnist_test[id_hold_train, 784])
        pred =knn.predict(mnist_test[id_hold_test, 0:783])
        #予測精度の計測
        accuracy_k = np.append(accuracy_k, metrics.accuracy_score(mnist_test[id_hold_test, 784], pred))
    #初期化に用いた要素の削除
    accuracy_k = np.delete(accuracy_k, 0)
    accuracy = np.append(accuracy, [accuracy_k], axis=0)
accuracy = np.delete(accuracy, 0, axis=0)
accuracy_mean = np.mean(accuracy, axis=1)
accuracy_max = float("inf")
#最大値を与えるkの値を探す
for i in range(10):
    if(accuracy_max > accuracy_mean[i]):
        accuracy_max = accuracy_mean[i]
        max_k = i
print(max_k + 1)
print(accuracy_max)

#==========
#一つ抜き法
#==========
accuracy_one_cluster = np.empty(1)
accuracy_one_mean  = np.empty(1)
for i in range(20):
    accuracy = np.empty(1)
    accuracy_one = np.empty(1)
    #knn法の実装
    for i in range(0,999):
        id_one_test = i
        id_one_train = np.array([k for k in range(1000) if k not in [id_one_test]])
        accuracy_one = np.empty((1,20))
        #kの値
        k = 3
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(mnist_test[id_one_train, 0:783], mnist_test[id_one_train, 784])
        pred = knn.predict([mnist_test[id_one_test, 0:783]])
        #print(pred)
        #予測精度の計測
        accuracy_k = metrics.accuracy_score([mnist_test[id_one_test,784]], [pred])
        #初期化に用いた要素の削除
        accuracy = np.append(accuracy, accuracy_k)
    accuracy = np.delete(accuracy, 0)
    accuracy_one  = np.mean(accuracy)
    accuracy_one_cluster = np.append(accuracy_one_cluster, accuracy_one)
#20回の平均を出す
accuracy_one_cluster = np.delete(accuracy_one_cluster, 0)
accuracy_one_mean = np.mean(accuracy_one_cluster)
print(accuracy_one_mean)
