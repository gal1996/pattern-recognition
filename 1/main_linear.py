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

#*******************
#線形識別関数の実装
#*******************
mix_matrix_linear = li.linear(pima_te, pima_te_p, pima_te_n, 0)
print(mix_matrix_linear)

#==============
#評価基準の計算
#==============

TP = mix_matrix_linear[0][0]
FN = mix_matrix_linear[0][1]
FP = mix_matrix_linear[1][0]
TN = mix_matrix_linear[1][1]
P = TP + FN
N = FP + TN

#偽陽性率
FPr_l = FP / N
#真陽性率
TPr_l = TP / P
#適合率
precision_l = TP/(TP + FP)
#再現率
recall_l = TP / P
#正確度
accuracy_l = (TP + TN) / (P + N)
#F値
F_m_l = 2/(1/precision_l + 1/recall_l)
#================
#問題3のプログラム
#================
FPR_l = np.zeros((1,1))
TPR_l = np.zeros((1,1))
for i in range(-5000, 5000, 1):
    mix_matrix_l = li.linear(pima_te, pima_te_p, pima_te_n, i/50)
    #偽陽性率と真陽性率の計算
    TP = mix_matrix_l[0][0]
    FN = mix_matrix_l[0][1]
    FP = mix_matrix_l[1][0]
    TN = mix_matrix_l[1][1]
    P = TP + FN
    N = FP + TN
    fpr = FP / N
    tpr = TP / P
    FPR_l = np.append(FPR_l, fpr)
    TPR_l = np.append(TPR_l, tpr)
FPR_l = np.delete(FPR_l, (0,0))
TPR_l = np.delete(TPR_l, (0,0))
#====================
#データのプロット
#====================
plt.title("pima-indiean ROCcurve(linear)")
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel("FPr")
plt.ylabel("TPr")
plt.plot(FPR_l, TPR_l)
plt.show()

#=============
#AUCの計算
#=============
AUC_l = 0
for i in range(FPR_l.shape[0] - 1):
    AUC_l += TPR_l[i] * (FPR_l[i+1] - FPR_l[i])
print(AUC_l)
