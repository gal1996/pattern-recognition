import pandas as pd
import numpy as np
import matplotlib as mpl

#データの読み込み
d_pima_tr = pd.read_csv("pimatr.csv", index_col=0)
pima_tr = np.array(d_pima_tr)
#yes, noで分割する
yn = 0
nn = 0
pima_tr_n = np.zeros((1,8))
pima_tr_p = np.zeros((1,8))
#糖尿病である人を１、そうでない人を０とする
for i in range(200):
    if(pima_tr[i][7] == "Yes"):
        pima_tr_p = np.append(pima_tr_p, pima_tr[i])
        pima_tr[i][7] = 1
        yn += 1
    else:
        pima_tr_n = np.append(pima_tr_n, pima_tr[i])
        pima_tr[i][7] = 0
        nn += 1
#データの形成
pima_tr_n = np.reshape(pima_tr_n, (nn+1, 8))
pima_tr_p = np.reshape(pima_tr_p, (yn+1, 8))
pima_tr_n = np.delete(pima_tr_n, 0, 0)
pima_tr_p = np.delete(pima_tr_p, 0, 0)
pima_tr_n[:,7] = 0
pima_tr_p[:,7] = 1



pima_tr_n = pima_tr_n.astype(float)
pima_tr_p = pima_tr_p.astype(float)
pima_tr = pima_tr.astype(float)

#=============================================================
#ベイズ識別規則の作成(i = "yes"(poisitve), j = "no"(negative))
#=============================================================

#共分散行列の生成
pima_tr_p = pima_tr_p.T
pima_tr_n = pima_tr_n.T
cov_p = np.cov(pima_tr_p[0:7][:])
cov_n = np.cov(pima_tr_n[0:7][:])

#テストデータの形成
pima_tr = pima_tr.T
x = pima_tr
xt = pima_tr.T

#行列Sの生成
covi_p = np.linalg.inv(cov_p)
covi_n = np.linalg.inv(cov_n)
S = covi_p - covi_n

#ベクトルcの生成
mu_i = np.mean(pima_tr_p[0:7], axis=1)
mu_j = np.mean(pima_tr_n[0:7], axis=1)
mu_it = mu_i.T
mu_jt = mu_j.T
c = np.dot(mu_jt, covi_n) - np.dot(mu_it, covi_p)

#Fの生成
det_cov_p = np.linalg.det(cov_p)
det_cov_n = np.linalg.det(cov_n)
P = pima_tr.shape[1]
Pi = pima_tr_p.shape[1]
Pj = pima_tr_n.shape[1]
#print(Pi)
PCi = Pi/P
PCj = Pj/P
F = np.dot(np.dot(mu_it, covi_p), mu_i) - np.dot(np.dot(mu_jt, covi_n), mu_j) + np.log(det_cov_p / det_cov_n) - \
    2 * np.log(PCi/PCj)

classP = np.zeros((8,1))
classN = np.zeros((8,1))
class_line = np.zeros((1,8))
y = 0
n = 0
e = 0
#ベイス識別関数の作成
for i in range(pima_tr.shape[1]):
    #識別関数の値
    f = np.dot(np.dot(xt[i,0:7], S), x[0:7,i]) + 2 * np.dot(c.T, x[0:7,i]) + F
    if(f < 0): #陽性である（糖尿病である）
        classP = np.append(classP, x[:,i])
        y += 1
    elif(f > 0): #陰性である(糖尿病でない)
        classN = np.append(classN, x[:,i])
        n += 1
    elif(f == 0):
        class_line = np.append(class_line, x[:,i])
        e += 1
#得られたデータの整形
classP = np.reshape(classP, (y+1, 8))
classN = np.reshape(classN, (n+1, 8))
classP = np.delete(classP, 0, 0)
classN = np.delete(classN, 0, 0)
class_line = np.reshape(class_line, (e+1, 8))
class_line = np.delete(class_line, 0, 0)

#============================
#混同行列の作成
#============================
TP = 0
FP = 0
TN = 0
FN = 0
for i in range(classP.shape[0]):
    #陽性であって、陽性クラスに正しく分類された人
    if(classP[i,7] == 1.0e+00):
        TP += 1
    #陰性であって間違って陽性クラスに分類された人
    elif(classP[i,7] == 0.0e+00):
        FP += 1
for i in range(classN.shape[0]):
    #陽性であって間違って陰性クラスに分類された人
    if(classN[i,7] == 1.0e+00):
        FN += 1
    #陰性であって正しく陰性に分類された人
    elif(classN[i,7] == 0.0e+00):
        TN += 1
mix_matrix = np.array([[TP, FN],
                       [FP, TN]])

#=================
#各性能評価値の計算
#=================
N = FP + TN
P = TP + FN
#偽陽性率
FPr = FP / F
#真陽性率
TPr = TP / P
#適合率
precision = TP / (TP + FP)
#正確度
accuracy = (TP + TN) / (P + N)
#再現率
recall = TP / P
#F値
F_measure = 2/(1/precision + 1/recall)
print(accuracy)
