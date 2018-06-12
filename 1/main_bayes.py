import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Bayes_identification_rules as bi
import linear_identification_rules as li

#テストデータの読み込み
d_pima_te = pd.read_csv("pimate.csv", index_col=0)
pima_te = np.array(d_pima_te)
#yes, noで分割する
yn = 0
nn = 0
pima_te_n = np.zeros((1,8))
pima_te_p = np.zeros((1,8))
#糖尿病である人を１、そうでない人を０とする

for i in range(pima_te.shape[0]):
    if(pima_te[i][7] == "Yes"):
        pima_te_p = np.append(pima_te_p, pima_te[i])
        pima_te[i][7] = 1
        yn += 1
    else:
        pima_te_n = np.append(pima_te_n, pima_te[i])
        pima_te[i][7] = 0
        nn += 1
#データの形成
pima_te_p = np.reshape(pima_te_p, (yn+1, 8))
pima_te_n = np.reshape(pima_te_n, (nn+1, 8))
pima_te_n = np.delete(pima_te_n, 0, 0)
pima_te_p = np.delete(pima_te_p, 0, 0)
pima_te_n[:,7] = 0
pima_te_p[:,7] = 1
pima_te_n = pima_te_n.astype(float)
pima_te_p = pima_te_p.astype(float)
pima_te = pima_te.astype(float)

#**************************
#ベイズの識別規則の実装
#**************************

mix_matrix = bi.bayes(pima_te, pima_te_p, pima_te_n, 0)
print(mix_matrix)

#=============
#評価指標の計算
#=============
TP = mix_matrix[0][0]
FN = mix_matrix[0][1]
FP = mix_matrix[1][0]
TN = mix_matrix[1][1]
P = TP + FN
N = FP + TN

#偽陽性率
FPr = FP / N
#真陽性率
TPr = TP / P
#適合率
precision = TP/(TP + FP)
#再現率
recall = TP / P
#正確度
accuracy = (TP + TN) / (P + N)
#F値
F_m = 2/(1/precision + 1/recall)

#================
#問題3のプログラム
#================
FPR = np.zeros((1,1))
TPR = np.zeros((1,1))
for i in range(-5000, 5000, 1):
    mix_matrix = bi.bayes(pima_te, pima_te_p, pima_te_n, i/50)
    #偽陽性率と真陽性率の計算
    TP = mix_matrix[0][0]
    FN = mix_matrix[0][1]
    FP = mix_matrix[1][0]
    TN = mix_matrix[1][1]
    P = TP + FN
    N = FP + TN
    fpr = FP / N
    tpr = TP / P
    FPR = np.append(FPR, fpr)
    TPR = np.append(TPR, tpr)
FPR = np.delete(FPR, (0,0))
TPR = np.delete(TPR, (0,0))
#====================
#データのプロット
#====================
plt.title("pima-indiean ROCcurve")
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("FPr")
plt.ylabel("TPr")
plt.plot(FPR, TPR)
plt.show()

#=============
#AUCの計算
#=============
AUC = 0
for i in range(FPR.shape[0] - 1):
    AUC += TPR[i] * (FPR[i+1] - FPR[i])
print(AUC)
