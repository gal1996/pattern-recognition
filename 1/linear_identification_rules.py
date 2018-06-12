#******************************************************************
#線形識別関数は、ベイズの識別規則の共分散行列が0であると仮定することで得られる
#******************************************************************
import pandas as pd
import numpy as np
import matplotlib as mpl

def linear(x, x_p, x_n, p):
    #共分散行列の生成
    x_p = x_p.T
    x_n = x_n.T
    cov_p = np.cov(x_p[0:7][:])
    cov_n = np.cov(x_n[0:7][:])

    #テストデータの形成
    x = x.T
    x = x
    xt = x.T

    #行列Sの生成
    covi_p = np.linalg.inv(cov_p)
    covi_n = np.linalg.inv(cov_n)
    S = covi_p - covi_n

    #ベクトルcの生成
    mu_i = np.mean(x_p[0:7], axis=1)
    mu_j = np.mean(x_n[0:7], axis=1)
    mu_it = mu_i.T
    mu_jt = mu_j.T
    c = np.dot(mu_jt, covi_n) - np.dot(mu_it, covi_p)

    #Fの生成
    det_cov_p = np.linalg.det(cov_p)
    det_cov_n = np.linalg.det(cov_n)
    P = x.shape[1]
    Pi = x_p.shape[1]
    Pj = x_n.shape[1]
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
    for i in range(x.shape[1]):
        #識別関数の値
        f = 2 * np.dot(c.T, x[0:7,i]) + F
        if(f < p): #陽性である（糖尿病である）
            classP = np.append(classP, x[:,i])
            y += 1
        elif(f > p): #陰性である(糖尿病でない)
            classN = np.append(classN, x[:,i])
            n += 1
        elif(f == p):
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
    return mix_matrix
