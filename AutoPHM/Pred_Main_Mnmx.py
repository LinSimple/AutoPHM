# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 13:32:43 2021

@author: SESA580088
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import csv
import datetime
import json 
import copy
import threading
import time
import schedule
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from scipy import stats as st
import scipy.signal as sn
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from tkintertable import TableCanvas

feature_name = ["Date", "Mean", "Std", "Skewness", "Kurtosis", "Peak2Peak",
                "RMS", "CrestFactor", "ShapeFactor", "ImpulseFactor", "MarginFactor", 
                "Energy", "SKMean", "SKStd", "SKSkewness", "SKKurtosis"]

###保存路径
global savepath

###执行次数
global ActTime

###生成参数
global SourceData#数据源
global PlotTimeData#时间域数据
global x_date#数据源时间序列
global p_date#预测时间序列
global PlotFreqData#STFT后频域数据
global NonSmothedFeature#未平滑特征
global SmothedFeature#平滑后特征
global TrainSmothedFeature#训练集特征
global SortedFeaValue#特征价值排序
global PCAData#PCA
global SelectedPCA#选择的PCA
global PreHealInd#预测值
global HealthIndicator#预测曲线值

###输入参数
global CsvFilePath#csv文件路径
global SampleFrequency#采样频率
global WC#STFT 窗函数
global MovingStep#平滑步数
global TrainSampleDays#训练样本天数
global FeatureImportance#特征重要性
global PCARate#PCA成分方差比率
global LifeThreshold#剩余寿命指标
global NumOfPreDays#预测天数

###图片对象
global img1, photo1
global img2, photo2
global img3, photo3
global img4, photo4
global img5, photo5

#表对象
global table

###随机生成大小为number的颜色数组
def RandomColor(number):
    color = []
    intnum = [str(x) for x in np.arange(10)]
    alphabet = [chr(x) for x in (np.arange(6) + ord('A'))]
    colorArr = np.hstack((intnum, alphabet))
    for j in range(number):
        color_single = '#'
        for i in range(6):
            index = np.random.randint(len(colorArr))
            color_single += colorArr[index]
            #Out[]: '#EDAB33'
        color.append(color_single)
    return color

###普峭度计算
def Pkurtosis(v, fs, wc):
    f, t, Z = sn.stft(v, fs, nperseg=wc)#短时傅里叶变换
    P = np.square(np.abs(Z))
    M4 = np.mean(np.square(P), 1)
    M2 = np.mean(P, 1)
    K = Z.shape[1]
    result = []
       
    if K < 2:
        SK = np.divide(M4, np.square(M2)) - 2
    else:
        SK = (K+1)/(K-1)*np.divide(M4, np.square(M2)) - 2
    for i in range(len(f)):
        if ((f[i] <= fs/wc) or (f[i] >= (fs/2 - fs/wc))):
            SK[i] = 0
    result.append(SK)
    result.append(f)
    return result

###计算时域和频域主要特征
def Feature(time_y, fre_y):
    feature = []
    #时域
    feature.append(np.mean(time_y))#Mean 均值
    feature.append(np.std(time_y))#STD 标准差
    feature.append(st.skew(time_y))#Skewness 偏度
    feature.append(st.kurtosis(time_y))#Kurtosis 峰度
    feature.append(np.max(time_y) - np.min(time_y))#P2P 峰峰值
    feature.append(np.sqrt(np.mean(np.square(time_y))))#RMS 均方根
    feature.append(np.max(time_y)/(np.sqrt(np.mean(np.square(time_y)))))#Crest Factor峰值因子
    feature.append(np.sqrt(np.mean(np.square(time_y)))/(np.mean(np.abs(time_y))))#Shape Factor波形因子
    feature.append(np.max(time_y)/(np.mean(np.abs(time_y))))#Impulse Factor脉冲因子
    feature.append(np.max(time_y)/np.square(np.mean(np.abs(time_y))))#Margin Factor 裕度因子
    feature.append(np.sum(np.square(time_y)))#energy 能量
    #频域
    feature.append(np.mean(fre_y))
    feature.append(np.std(fre_y))
    feature.append(st.skew(fre_y))
    feature.append(st.kurtosis(fre_y))    
    return feature
    
###移动均值滤波器
def MoveAverage(A, a):
    A = A[::-1]
    B = []
    for i in range(len(A)):
        j = i+a
        if (j <= len(A)):
            x = A[i:j]
        else:
            x = A[i:]
        B.append(np.mean(x))
    return B[::-1]

###获取有效数据
def GetValidData():
    global CsvFilePath
    try:    
        CsvFilePath = E1.get()
        if not os.path.exists(CsvFilePath):
            raise
        with open(os.getcwd() + "\\sourcepath.json", "r") as read_json:
            sourcepara = json.load(read_json)
    except:
        messagebox.showerror('Error', 'Can not Get Data, Please Check filepath or files')
    else: 
        spl = []
        sourcepath = sourcepara["sourcepath"]
        for file in os.listdir(sourcepath):#获取该路径下所有文件(csv)
            spl.append(sourcepath + "\\" + file)
        spl = sorted(spl, reverse=True)
        i=0
        for path in spl:
            if (i < -88):
                break
            data = pd.read_csv(path)
            if (data['Value'].max() > 1):
                if (i >= -88):
                    tname = CsvFilePath + "\\" +(datetime.datetime.now()+datetime.timedelta(days=i)).strftime("%Y%m%d") + '.csv'
                    shutil.copyfile(path, tname)
                    i=i-1                

def BackUpCsv():
    global CsvFilePath
    backuppath = os.getcwd() + "\\backup_csvdata" +"\\" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")        
    if not os.path.exists(backuppath):
        os.makedirs(backuppath)
    for file in os.listdir(CsvFilePath):
        shutil.move(CsvFilePath + "\\" + file, backuppath+ "\\" + file)
        
###获取数据
def GetSourceData():
    global x_date
    global CsvFilePath
    global SourceData
    global PlotTimeData
    global img1, photo1
    try:
        CsvFilePath = E1.get()
        if not os.path.exists(CsvFilePath):
            raise 
    except:
        messagebox.showerror('Error', 'Can not Get Data, Please Check filepath or files')
    else:
        for widget in fig1.winfo_children():
            widget.destroy()
        L = tk.Label(fig1, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
                
        CsvFileList = []
        date = []
        df = pd.DataFrame()
        
        ###获取数据
        for file in os.listdir(CsvFilePath):#获取该路径下所有文件(csv)
            CsvFileList.append(CsvFilePath + "\\" + file)
        CsvFileList = sorted(CsvFileList)
        for csvp in CsvFileList:
            data = pd.read_csv(csvp)#读csv文件
            d = csvp.split("\\")[-1][2:8]
            #print(d)
            date.append(d)
            #if (data['Value'].max() > 1):
            data.rename(columns={'Value' : d}, inplace=True)#每个csv文件数据都是一个dataframe，按顺序重命名0,1,2...
            df = pd.concat([df,data[d]], axis = 1)#按列拼接
        x_date = date#原始数据日期
        E2.delete(0, tk.END)#删除Entry内容
        E2.insert("insert",datetime.datetime.strptime(x_date[0], "%y%m%d").strftime("%Y-%m-%d"))#添加Entry内容
        E3.delete(0, tk.END)#删除Entry内容
        E3.insert("insert",datetime.datetime.strptime(x_date[-1], "%y%m%d").strftime("%Y-%m-%d"))#添加Entry内容
        E4.delete(0, tk.END)#删除Entry内容
        E4.insert("insert",str(df.shape[1]))#添加Entry内容
        SourceData = df
        
        ###生成简单展示图片
        Ran_Col = RandomColor(SourceData.shape[1])#获取随机颜色数组
        DataPlot = pd.DataFrame()#dataframe格式转为一列
        DataPlot_ = list()#plot数据集
        sec = range(SourceData.shape[1]+1)
        cmap = matplotlib.colors.ListedColormap(Ran_Col)#定义颜色映射
        norm = matplotlib.colors.BoundaryNorm(sec, cmap.N)#定义颜色映射区间
        for i in range(SourceData.shape[1]):#按照列遍历
            DataPlot = pd.concat([DataPlot, SourceData.iloc[:,i]], axis = 0)#转换dataframe形状
        interval = SourceData.shape[0] / 1000
        for i in range(0, SourceData.shape[0] * SourceData.shape[1], int(interval)):#每隔x个点取一个
            DataPlot_.append(DataPlot.iat[i,0])#保存为list格式
        PlotTimeData = DataPlot_
        x = np.arange(0, SourceData.shape[1], 0.001)#x大小为10000xn
        y = np.array(DataPlot_)#list转array
        points = np.array([x, y]).T.reshape(-1, 1, 2)#x和y合并成[[x1,y1],[x2,y2]...]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)#拼接，不懂这步是否需要，看别人都带了这步
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(x)#区间x映射颜色区间
        plt.figure(figsize=(3, 3))
        plt.gca().add_collection(lc)#绘图
        plt.xticks(list(sec)[::20], list(x_date)[::20], fontsize=9)
        plt.yticks(fontsize=9)
        plt.title(u"Time domain diagram of vibration signal", fontsize=9)
        plt.margins(0, 0)
        plt.savefig("img/TimeDomain.png", transparent=True, dpi=100)
        plt.close()
        
        for widget in fig1.winfo_children():
            widget.destroy()
        ###图片展示
        img1 = Image.open("img/TimeDomain.png")
        photo1 = ImageTk.PhotoImage(img1.resize((298, 270))) 
        tk.Label(fig1, image=photo1).grid()
        
        #messagebox.showinfo('Reminder', 'GetSourceData Completed!')
        
###获取数据源的线程
def ThreadForGetSourceData():
    t1 = threading.Thread(target=GetSourceData)
    t1.setDaemon(True)
    t1.start()

###特征提取
def GetFea():
    global x_date
    global SourceData
    global PlotFreqData
    global NonSmothedFeature
    global img2, photo2
    global table
    global SampleFrequency#采样频率
    global WC#STFT 窗函数  
    feature = []
    SK = []
    ###特征提取
    try:
        SampleFrequency = int(E5.get())
        WC = int(E6.get())
        if SourceData.empty:
            raise
    except:
        messagebox.showerror('Error', 'Can not Get Needed Para, Please Check Input')
    else:
        for widget in fig2.winfo_children():
            widget.destroy()
        for widget in fig3.winfo_children():
            widget.destroy()
        L = tk.Label(fig2, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
        L = tk.Label(fig3, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
        
        
        t_y = [[] for i in range(SourceData.shape[1])]        
        for i in range(SourceData.shape[1]):
            for j in range(SourceData.shape[0]):
                t_y[i].append(SourceData.iat[j, i])#dataframe格式转数组
            sk = Pkurtosis(t_y[i], SampleFrequency, WC)#计算普峭度
            SK.append(sk)#绘制谱峭度图
            fea = Feature(t_y[i], sk[0])#计算特征
            feature.append(fea)#特征   
        PlotFreqData = SK#获取STFT
        NonSmothedFeature = feature#获取未平滑的feature,行为日期，列为特征
           
        ###生成简单展示图片
        Ran_Col = RandomColor(SourceData.shape[1])
        fig = plt.figure(figsize=(3, 3))
        ax = Axes3D(fig)
        for i in range(SourceData.shape[1]):  
            y = [(i+1) for j in range(len(SK[i][0]))]
            ax.plot(SK[i][1], y, SK[i][0], Ran_Col[i])
        plt.yticks(list(range(1, SourceData.shape[1]+1))[::20], list(x_date)[::20], fontsize=7)
        plt.xticks(fontsize=9)
        plt.title(u"Frequency domain diagram of vibration signal", fontsize=9)
        plt.savefig("img/FreqDomain.png", transparent=True, dpi=100)
        plt.close()
        
        for widget in fig2.winfo_children():
            widget.destroy()
        for widget in fig3.winfo_children():
            widget.destroy()
            
        ###图片展示
        img2 = Image.open("img/FreqDomain.png")
        photo2 = ImageTk.PhotoImage(img2.resize((298, 270))) 
        tk.Label(fig2, image=photo2).grid()
        
        ###特征表展示
        table = TableCanvas(fig3)
        table.show()
        for i in range(6, 17):        
            table.addColumn('%d'%i)#增加列 
        for i in range(len(NonSmothedFeature) - 10):
            table.addRow()#增加行 
        for i in range(1, 17):
            table.model.columnlabels[str(i)] = feature_name[i-1]#修改列名
        for i in range(len(NonSmothedFeature)):
            table.model.setValueAt(datetime.datetime.strptime(x_date[i], "%y%m%d").strftime("%Y-%m-%d"),i,0)
        for i in range(len(NonSmothedFeature)):
            for j in range(15):
                table.model.setValueAt(str(round(NonSmothedFeature[i][j],2)),i,j+1)
        table.redraw()
        
        #messagebox.showinfo('Reminder', 'GetFea Completed!')

###获取数据特征的线程
def ThreadForGetFea():
    t2 = threading.Thread(target=GetFea)
    t2.setDaemon(True)
    t2.start()

###特征排序
def SortFea():
    global NonSmothedFeature
    global SmothedFeature
    global TrainSmothedFeature
    global SourceData
    global SortedFeaValue
    global img3, photo3
    global MovingStep#平滑步数
    global TrainSampleDays#训练样本天数
    
    try:
        MovingStep = int(E7.get())
        TrainSampleDays = int(E8.get())
        if (TrainSampleDays > SourceData.shape[1] or TrainSampleDays < 10):
            raise
        if len(NonSmothedFeature) == 0:
            raise
    except:
        messagebox.showerror('Error', 'Can not Get Needed Para, Please Check Input')
    else:
        for widget in fig4.winfo_children():
            widget.destroy()
        L = tk.Label(fig4, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
        
        nonsmtfea = copy.deepcopy(NonSmothedFeature)#行为日期，列为特征
        fea_fitter = []
        nonsmtfea = np.transpose(nonsmtfea)#转置, 行为特征，列为日期      
        for i in range(len(nonsmtfea)):
            fea_fitter.append(MoveAverage(nonsmtfea[i], MovingStep))#滑动平均滤波 
        SmothedFeature = fea_fitter#行为特征，列为日期
        
        smtfea = copy.deepcopy(SmothedFeature)
        smtfea = np.transpose(smtfea)#转置 行为日期，列为特征
        trainsmtfea = smtfea[0:TrainSampleDays]#取前sampledays天数据
        trainsmtfea = np.transpose(trainsmtfea)#转置回来, 行为特征，列为日期
        TrainSmothedFeature = trainsmtfea
        fea_name = feature_name[1:]#15个特征名
        fea_value = {}
        for i in range(len(trainsmtfea)):
            posi = 0
            nega = 0
            for j in range(len(trainsmtfea[i])-1):
                if (trainsmtfea[i][j+1] > trainsmtfea[i][j]):
                    posi = posi + 1
                if (trainsmtfea[i][j+1] < trainsmtfea[i][j]):
                    nega = nega + 1
            value = np.abs(posi - nega) / (len(trainsmtfea[i])-1)
            fea_value[fea_name[i]] = value
        SortedFeaValue = dict(sorted(fea_value.items(),  key=lambda d: d[1], reverse=True))
        
        ###生成简单展示图片
        plt.figure(figsize=(3, 3))
        plt.bar(SortedFeaValue.keys(), SortedFeaValue.values())
        plt.xticks(rotation=70, fontsize=6)
        plt.title(u"Feature Importance", fontsize=9)
        plt.margins(0, 0)
        plt.savefig("img/SortedFeature.png", transparent=True, dpi=100)
        plt.close()
        
        for widget in fig4.winfo_children():
            widget.destroy()
            
        ###图片展示
        img3 = Image.open("img/SortedFeature.png")
        photo3 = ImageTk.PhotoImage(img3.resize((298, 270))) 
        tk.Label(fig4, image=photo3).grid()
        
        #messagebox.showinfo('Reminder', 'SortFea Completed!')

###排序数据特征的线程
def ThreadForSortFea():
    t3 = threading.Thread(target=SortFea)
    t3.setDaemon(True)
    t3.start()

###PCA
def PriCA():
    global FeatureImportance
    global SortedFeaValue
    global SmothedFeature
    global TrainSmothedFeature
    global PCAData
    global PCARate
    global x_date
    global img4, photo4
    
    try:
        FeatureImportance = float(E9.get())
        if len(SmothedFeature)==0 or len(TrainSmothedFeature)==0:
            raise
        if (FeatureImportance >= 1 or FeatureImportance <= 0):
            raise
        else:
            a = list(SortedFeaValue.values())
            b = []
            for i in a:
                if i not in b:
                    b.append(i)
            sltfea = { k: v for k, v in SortedFeaValue.items() if v>= b[1] }#特征提取
    except:
        messagebox.showerror('Error', 'Can not Get Needed Para, Please Check Input')
    else:
        
        for widget in fig5.winfo_children():
            widget.destroy()
        L = tk.Label(fig5, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
        
        allfeatureselected = []
        trainfeatureselected = []
        fea_name = feature_name[1:]#15个特征名
        selectedfeaname = list(sltfea.keys())
        for i in range(len(selectedfeaname)):
            fnindex = fea_name.index(selectedfeaname[i])
            allfeatureselected.append(list(SmothedFeature[fnindex]))#原始数据
            trainfeatureselected.append(list(TrainSmothedFeature[fnindex]))#训练数据
        
        ###归一化
        trainfeanl = []#归一化后的trainfeatureselected数组
        allfeanl = []#归一化后的featureselected数组
        for i in range(len(trainfeatureselected)):
            trainfeamean = np.mean(trainfeatureselected[i])
            trainfeastd = np.std(trainfeatureselected[i])
            trainfeanormalized = np.divide((trainfeatureselected[i] - trainfeamean), trainfeastd)#训练数据归一化
            allfeanormalized = np.divide((allfeatureselected[i] - trainfeamean), trainfeastd)#原始数据归一化
            trainfeanl.append(list(trainfeanormalized))
            allfeanl.append(list(allfeanormalized))
            
        trainfeanl = np.transpose(trainfeanl)#转置
        ###PCA
        pca = PCA(n_components=1)#'mle')
        pca.fit(trainfeanl)
        feavector = pca.components_#特征向量
        fearate = pca.explained_variance_ratio_
        
        pcarate = []
        pcadata = []
        #x = range(1, len(allfeanl[0])+1)
        for i in range(feavector.shape[0]):
            f = np.dot(feavector[i],allfeanl)
            pcadata.append(f)
            lab = 'PCA' + str((i+1)) + '-' + str(round(fearate[i], 2))
            pcarate.append(lab)
        
        PCAData = pcadata
        PCARate = pcarate
        com1["value"] = (pcarate)
        com1.current(int(para['pcafeature']))
        ###生成简单展示图片
        plt.figure(figsize=(3, 3))
        x = range(1, len(allfeanl[0])+1)
        for i in range(feavector.shape[0]):
            plt.plot(x, pcadata[i], label= pcarate[i], marker='o')        
        plt.legend(framealpha=1, frameon=True, fontsize=6, loc='lower left')
        plt.xticks(list(x)[::20], list(x_date)[::20], fontsize=9)
        plt.title(u"PCA", fontsize=9)
        plt.margins(0, 0)
        plt.savefig("img/PCAFeature.png", transparent=True, dpi=100)
        plt.close()
        
        for widget in fig5.winfo_children():
            widget.destroy()
            
        ###图片展示
        img4 = Image.open("img/PCAFeature.png")
        photo4 = ImageTk.PhotoImage(img4.resize((298, 270))) 
        tk.Label(fig5, image=photo4).grid()
        
        #messagebox.showinfo('Reminder', 'PCA Completed!')

###PCA线程
def ThreadForPCA():
    t4 = threading.Thread(target=PriCA)
    t4.setDaemon(True)
    t4.start()      

################
##三次指数平滑预测算法
################
    
###一次指数平滑预测
def ExpSmoothing1(alpha, data):
    s_single=[]
    s_single.append(data[0])
    for i in range(1, len(data)):
        s_single.append(alpha * data[i] + (1 - alpha) * s_single[i-1])
    return s_single

###三次指数平滑预测
def ExpSmoothing3(alpha, data):
    s_single = ExpSmoothing1(alpha, data)
    s_double = ExpSmoothing1(alpha, s_single)
    s_triple = ExpSmoothing1(alpha, s_double)
    
    a_triple = [0 for i in range(len(data))]
    b_triple = [0 for i in range(len(data))]
    c_triple = [0 for i in range(len(data))]
    F_triple = [0 for i in range(len(data))]
    
    for i in range(len(data)):
        a = 3 * s_single[i] - 3 * s_double[i] + s_triple[i]
        b = (alpha / (2 * ((1 - alpha) ** 2))) * ((6 - 5 * alpha) * s_single[i] - 2 * ((5 - 4 * alpha) * s_double[i]) + (4 - 3 * alpha) * s_triple[i])
        c = ((alpha ** 2) / (2 * ((1 - alpha) ** 2))) * (s_single[i] - 2 * s_double[i] + s_triple[i])
        F = a + b + c
        a_triple[i] = a
        b_triple[i] = b
        c_triple[i] = c
        F_triple[i] = F#包含预测一天的预测值
    return a_triple, b_triple, c_triple, F_triple

###误差分析
def ErrorAnalysis(F, data):
    AE_num = 0
    for i in range(1,len(data)):
        _AE = abs(F[i-1] - data[i])        
        AE_num += _AE        
    MAE = AE_num / (len(data)-1)
    return MAE

###寻找最优平滑a, b, c, F
def FindBestPre(data):
    alpha_all = [0.01 * i for i in range(1,100)]
    best_alpha = 0
    best_a = []
    best_b = []
    best_c = []
    best_F = []
    min_MAE = float('Inf') #  无穷大
    for i in range(len(alpha_all)):
        alpha = alpha_all[i]
        a_triple, b_triple, c_triple, F_triple = ExpSmoothing3(alpha, data)#三次平滑
        MAE_triple= ErrorAnalysis(F_triple, data)#误差计算
        if MAE_triple <= min_MAE:
            min_MAE = MAE_triple
            best_alpha = alpha
            best_a = a_triple
            best_b = b_triple
            best_c = c_triple
            best_F = F_triple
        else:
            pass
    #print(ba)
    return best_alpha, best_a, best_b, best_c, best_F

###剩余寿命预测
def GetRUL():
    global PCAData
    global SelectedPCA
    global PCARate
    global LifeThreshold
    global NumOfPreDays
    global x_date
    global p_date
    global PreHealInd#x天预测值
    global HealthIndicator#预测曲线值
    global img5, photo5
    
    try:
        if len(PCARate)==0 or len(PCAData) == 0:
            raise
        i = PCARate.index(com1.get())
        LifeThreshold = float(E10.get())
        NumOfPreDays = int(E12.get())
    except:
        messagebox.showerror('Error', 'Can not Get Needed Para, Please Check Input')
    else:
        for widget in fig6.winfo_children():
            widget.destroy()
        L = tk.Label(fig6, text = '数据处理中...', font = ('Arial', 16), fg='green')
        L.grid()
        
        Health = (-PCAData[i]) - (-PCAData[i][0])
        SelectedPCA = Health

        datelist = copy.deepcopy(x_date)
        for i in range(NumOfPreDays):#预测x天
            datelist.append((datetime.datetime.strptime(datelist[-1], "%y%m%d") 
            + datetime.timedelta(days=1)).strftime("%y%m%d"))
        p_date = datelist
        alpha, a, b, c, F = FindBestPre(Health)
        E11.delete(0, tk.END)
        E11.insert("insert",str(alpha))
        p = []
        for day in range(2,NumOfPreDays+2):
            HIpre = a[-1]+b[-1]*day + c[-1]*(day**2)
            p.append(round(HIpre, 2))
            F.append(HIpre)
        PreHealInd = p
        HealthIndicator = F
        
        upper = [(i*1.1) for i in HealthIndicator]#信任上限
        lower = [(i*0.9) for i in HealthIndicator]#信任下限
        x = np.arange(1, len(HealthIndicator)+1)
        
        plt.figure(3, figsize=(3, 3))
        plt.axhline(LifeThreshold, c='y', label='Threshold')#阈值
        plt.plot(x[:len(Health)], Health[:len(Health)], c='b', label='Health Indicator')#健康因子
        plt.plot(x, HealthIndicator, c='g', label='ExponentialModel', marker='o')#三次指数平滑模型
        plt.plot(x, upper, c='r', alpha=0.5, linestyle='--', label='Confidence Interval')#信任区间上限
        plt.plot(x, lower, c='r', alpha=0.5, linestyle='--')#信任区间下限
        plt.fill_between(x, y1=upper, y2=lower, alpha=0.2, color="grey")
        plt.legend(framealpha=1, frameon=True, fontsize=6, loc='upper left')
        plt.xticks(list(x)[::20], list(p_date)[::20], fontsize=9)
        plt.yticks(fontsize=9)
        plt.ylim((-10, 40))
        plt.title(u"Predictors of Machine Health Indicators", fontsize=9)
        plt.margins(0, 0)
        plt.savefig("img/RUL.png", transparent=True, dpi=100)
        plt.close()
        
        for widget in fig6.winfo_children():
            widget.destroy()
            
        ###图片展示
        img5 = Image.open("img/RUL.png")
        photo5 = ImageTk.PhotoImage(img5.resize((298, 270))) 
        tk.Label(fig6, image=photo5).grid()
        
        #messagebox.showinfo('Reminder', 'GetRUL Completed!')
        
###RUL线程
def ThreadForGetRUL():
    t5 = threading.Thread(target=GetRUL)
    t5.setDaemon(True)
    t5.start()

###打开文件
def openfile():
    inputfile = filedialog.askopenfilename()
    print(inputfile)
    
###打开文件夹
def opendir():
    inputdir = filedialog.askdirectory()
    path = inputdir.replace('/','\\')#处理路径
    E1.delete(0, tk.END)#删除Entry内容
    E1.insert("insert",path)#添加Entry内容

###savefile线程
def ThreadForSaveFile():
    t6 = threading.Thread(target=savefile)
    t6.setDaemon(True)
    t6.start()

def savefile():
    global PlotTimeData
    global PlotFreqData
    global NonSmothedFeature
    global SortedFeaValue
    global PCAData
    global PCARate
    global PreHealInd#x天预测值
    global HealthIndicator#预测曲线
    global SelectedPCA
    global savepath
    
    try:
        if len(PlotTimeData) == 0:
            raise
        if len(PlotFreqData) == 0:
            raise
        if len(NonSmothedFeature) == 0:
            raise
        if not SortedFeaValue:
            raise
        if len(PCAData) == 0 or len(PCARate) == 0:
            raise
        if len(PreHealInd) == 0 or len(HealthIndicator) == 0 or len(SelectedPCA) == 0:
            raise
    except:
        messagebox.showerror('Error', 'Save Failed, Please Check')
    else:
        datename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")        
        savepath = os.getcwd() + "\\result" +"\\" + datename
        os.makedirs(savepath)
        PlotTimeDomain()
        PlotFreqDomain()
        CreateCsv()
        PlotSortedFea()
        PlotPCA()
        PlotRUL()
        messagebox.showinfo('Reminder', 'Savefile Completed!')
                   
###生成时域图片
def PlotTimeDomain():
    global SourceData
    global PlotTimeData
    global x_date
    global savepath
    
    Ran_Col = RandomColor(SourceData.shape[1])#获取随机颜色数组
    sec = range(SourceData.shape[1]+1)
    cmap = matplotlib.colors.ListedColormap(Ran_Col)#定义颜色映射
    norm = matplotlib.colors.BoundaryNorm(sec, cmap.N)#定义颜色映射区间
    x = np.arange(0, SourceData.shape[1], 0.001)#x
    y = np.array(PlotTimeData)#list转array
    points = np.array([x, y]).T.reshape(-1, 1, 2)#x和y合并成[[x1,y1],[x2,y2]...]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)#拼接，不懂这步是否需要，看别人都带了这步
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)#区间x映射颜色区间
    
    plt.figure()
    plt.gca().add_collection(lc)#绘图
    plt.xticks(list(sec)[::5], list(x_date)[::5], rotation=45, fontsize=8)
    #plt.xticks(sec, x_date, rotation=45, fontsize=8)
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min()+1, y.max()+1)
    plt.xlabel(u"Time(Day)")
    plt.title(u"Time domain diagram of vibration signal")
    plt.savefig(savepath + "/TimeDomainData.png",  dpi=500)
    plt.close()

###生成频域图片
def PlotFreqDomain():
    global PlotFreqData
    global SourceData
    global x_date
    global savepath
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(SourceData.shape[1]):  
        y = [(i+1) for j in range(len(PlotFreqData[i][0]))]
        ax.plot(PlotFreqData[i][1], y, PlotFreqData[i][0])
    
    plt.yticks(list(range(1, SourceData.shape[1]+1))[::5], list(x_date)[::5], fontsize=8)
    plt.xlabel(u"Frequancy(Hz)")
    plt.ylabel(u"Time(Day)")
    plt.title(u"Frequency domain diagram of vibration signal")
    plt.savefig(savepath + "/FreqDomainData.png", dpi=500)
    plt.close()

###保存特征表
def CreateCsv():
    global NonSmothedFeature
    global x_date
    global savepath
    
    featurelist = copy.deepcopy(NonSmothedFeature)
    with open(savepath + "/featuretable.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(feature_name)
        for i in range(len(featurelist)):
            flag = datetime.datetime.strptime(x_date[i], "%y%m%d").strftime("%Y-%m-%d")
            featurelist[i].insert(0, flag)
            writer.writerow(featurelist[i])
            
###保存特征排序图
def PlotSortedFea():
    global SortedFeaValue
    global savepath
    
    plt.figure()
    plt.bar(SortedFeaValue.keys(), SortedFeaValue.values())
    plt.xticks(rotation=45, fontsize=8)
    plt.title(u"Feature Importance")
    plt.savefig(savepath + "/SortedFeature.png", dpi=500)
    plt.close()

###保存PCA结果图
def PlotPCA():
    global PCAData
    global PCARate
    global x_date
    global savepath
    
    plt.figure()
    x = range(1, len(PCAData[0])+1)
    for i in range(len(PCAData)):
        plt.plot(x, PCAData[i], label= PCARate[i], marker='o')        
    plt.legend(framealpha=1, frameon=True, fontsize=8, loc='upper right')
    plt.xticks(list(x)[::5], list(x_date)[::5], rotation=45, fontsize=8)
    #plt.xticks(x, x_date, rotation=45, fontsize=8)
    plt.title(u"PCA")
    plt.savefig(savepath + "/PCAFeature.png", dpi=500)
    plt.close()

###保存RUL预测图和预测结果
def PlotRUL():
    global PreHealInd#x天预测值
    global HealthIndicator#预测曲线
    global LifeThreshold
    global x_date
    global p_date
    global SelectedPCA
    global savepath
    
    upper = [(i*1.1) for i in HealthIndicator]#信任上限
    lower = [(i*0.9) for i in HealthIndicator]#信任下限
    x = np.arange(1, len(HealthIndicator)+1)
    
    ###预测图
    plt.figure()
    plt.axhline(LifeThreshold, c='y', label='Threshold')#阈值
    plt.plot(x[:len(SelectedPCA)], SelectedPCA, c='b', label='True Values of Health Indicator')#健康因子
    #plt.plot(x, HealthIndicator, c='g', label='ExponentialModel', marker='o')#三次指数平滑模型
    plt.plot(x, HealthIndicator, c='g', label='Predictors of Health Indicator', marker='o')
    plt.plot(x, upper, c='r', alpha=0.5, linestyle='--', label='Confidence Interval')#信任区间上限
    plt.plot(x, lower, c='r', alpha=0.5, linestyle='--')#信任区间下限
    plt.fill_between(x, y1=upper, y2=lower, alpha=0.2, color="grey")
    plt.legend(framealpha=1, frameon=True, fontsize=8, loc='upper left')
    #plt.xticks(x, p_date, rotation=45, fontsize=8)
    plt.xticks(list(x)[::5], list(p_date)[::5], rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.title(u"Predictors of Machine Health Indicators")
    plt.savefig(savepath + "/RUL.png", dpi=500)
    plt.close()
    
    ###预测结果
    '''
    hl = copy.deepcopy(PreHealInd)
    n = datetime.datetime.now().strftime("%Y-%m-%d")
    d = datetime.datetime.strptime(x_date[-1], "%y%m%d").strftime("%Y-%m-%d")
    filename = n + ".csv"
    np.savetxt("predictedhealind/" + filename, hl, fmt='%.2f', delimiter = ',')
    with open("predictedhealind/" + filename, "a+", newline='') as f:
        a = csv.writer(f)
        a.writerow([d])
    '''
    
###保存参数
def SavePara():
    global PCARate
    try:
        wpara = {}
        wpara['csvfilepath'] = str(E1.get())
        wpara['samplefrequeny'] = int(E5.get())
        wpara['wc'] = int(E6.get())
        wpara['movingstep'] = int(E7.get())
        wpara['trainsampledays'] = int(E8.get())
        wpara['featureimportance'] = float(E9.get())
        wpara['pcafeature'] = int(PCARate.index(com1.get()))
        wpara['lifethreshold'] = float(E10.get())
        wpara['numofpredays'] = int(E12.get())
    except:
        messagebox.showerror('Error', 'Save failed, Please Check')
    else:
        with open(os.getcwd() + "\\para.json", "w") as write_json:
            json.dump(wpara, write_json, indent=4, separators=(',', ': '))
        messagebox.showinfo('Reminder', 'SavePara Completed!')

###SavePara线程
def ThreadForSavePara():
    t7 = threading.Thread(target=SavePara)
    t7.setDaemon(True)
    t7.start()

def WtToCSV(d):
    global LifeThreshold
    global HealthIndicator
    global SelectedPCA
    
    datax = []
    datax.append(LifeThreshold)
    if (d<=88):
        datax.append(SelectedPCA[d])
    else:
        datax.append(0)
    datax.append(HealthIndicator[d])
        
    np.savetxt("predictedhealind/ma.csv", datax, fmt='%.2f', delimiter = ',')
    
    d = d + 1
    timer = threading.Timer(900, WtToCSV, (d,))
    if (d > 95):
        timer.cancel()
    timer.start()

def AutoWTCSV():
    schedule.every().day.at("23:53").do(WtToCSV, 0)
    while True:    
        schedule.run_pending()
        time.sleep(1)

def Process():
    GetValidData()
    GetSourceData()
    GetFea()
    SortFea()
    PriCA()
    GetRUL()
    BackUpCsv()
    
def AutoProcess():
    schedule.every().saturday.at("23:48").do(Process)
    while True:    
        schedule.run_pending()
        time.sleep(1)

###自动处理
def AutoStart():
    Process()
    
    t9 = threading.Thread(target=AutoWTCSV)
    t9.setDaemon(True)
    t9.start()
    
    t10 = threading.Thread(target=AutoProcess)
    t10.setDaemon(True)
    t10.start()

###AutoStart线程
def ThreadForAutoStart():
    t8 = threading.Thread(target=AutoStart)
    t8.setDaemon(True)
    t8.start()
        
###退出
def exitapp():
    if messagebox.askokcancel("exit", "Sure to exit?"):
        MyWindow.destroy()

def DelayAuto():
    time.sleep(180)
    AutoStart()

t11 = threading.Thread(target=DelayAuto)
t11.setDaemon(True)
t11.start()

###UI界面
MyWindow = tk.Tk()
###参数获取
with open(os.getcwd() + "\\para.json", "r") as read_json:
    para = json.load(read_json)

MyWindow.title('WPF-MNMX Predictive Maintenance System')
max_width, max_height = MyWindow.maxsize()
align_center = "1346x660+%d+%d" %((max_width-1346)/2, (max_height-660)/2)
MyWindow.resizable(0,0)
MyWindow.geometry(align_center)
MyWindow.iconbitmap("img/theme.ico")

###菜单栏
menubar = tk.Menu()

filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='New File', accelerator="Ctrl+N", command='')
filemenu.add_command(label='Open File', accelerator="Ctrl+O", command = opendir)
filemenu.add_command(label='Save File', accelerator="Ctrl+S", command = ThreadForSaveFile)
filemenu.add_command(label='Save As', accelerator="Shift+Ctrl+N", command='')
filemenu.add_separator()
filemenu.add_command(label='Quit', accelerator="Alt+F4", command=exitapp)
menubar.add_cascade(label='file', menu=filemenu)

editormenu = tk.Menu(menubar, tearoff=0)
editormenu.add_command(label='Undo', accelerator="Ctrl+Z", command='')
editormenu.add_command(label='Redo', accelerator="Ctrl+Y", command='')
editormenu.add_separator()
editormenu.add_command(label='Copy', accelerator="Ctrl+C", command='')
editormenu.add_command(label='Paste', accelerator="Ctrl+V", command='')
editormenu.add_command(label='Cut', accelerator="Ctrl+X", command='')
editormenu.add_separator()
editormenu.add_command(label='Find', accelerator="Ctrl+F", command='')
editormenu.add_separator()
editormenu.add_command(label='Select All', accelerator="Ctrl+A", command='')
menubar.add_cascade(label='editor', menu=editormenu)

viewmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='view', menu=viewmenu)

aboutmenu = tk.Menu(menubar, tearoff=0)
aboutmenu.add_command(label='About', command='')
aboutmenu.add_command(label='Help', command='')
menubar.add_cascade(label='about', menu=aboutmenu)

MyWindow['menu'] = menubar

###工具栏
icons = ["new_file", "open_file", "save", "cut", "copy", "paste",
         "undo", "redo", "find_text"]
iconres = []

toolbar = tk.Frame(MyWindow)
toolbar.grid(row=0, column=0, sticky=tk.W, padx=10)
i=0
for icon in icons:
    toolicon = tk.PhotoImage(file="img/%s.gif" % (icon, ))
    toolbtn = tk.Button(toolbar, image=toolicon, command="")
    toolbtn.grid(row=0,column=i)
    iconres.append(toolicon)
    i=i+1

###参数栏
spany = 8
parabar = tk.Frame(MyWindow, relief='groove', bd=3)
parabar.grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)

L = tk.Label(parabar, text = 'Parameter Setting', font = ('Arial', 11), fg='blue')
L.grid(row=1, column=0, sticky=tk.E, pady=13)

L = tk.Label(parabar, text = 'Sourcedata Path', font = ('Arial', 11))
L.grid(row=2, column=0, sticky=tk.E, pady=spany)
E1 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E1.grid(row=2, column=1, sticky=tk.W, pady=spany)
E1.delete(0, tk.END)#删除Entry内容
E1.insert("insert", para['csvfilepath'])#添加Entry内容

L = tk.Label(parabar, text = 'Sample Startdate', font = ('Arial', 11))
L.grid(row=3, column=0, sticky=tk.E, pady=spany)
E2 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E2.grid(row=3, column=1, sticky=tk.W, pady=spany)

L = tk.Label(parabar, text = 'Sample Enddate', font = ('Arial', 11))
L.grid(row=4, column=0, sticky=tk.E, pady=spany)
E3 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E3.grid(row=4, column=1, sticky=tk.W, pady=spany)

L = tk.Label(parabar, text = 'Sample Days', font = ('Arial', 11))
L.grid(row=5, column=0, sticky=tk.E, pady=spany)
E4 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E4.grid(row=5, column=1, sticky=tk.W, pady=spany)

L = tk.Label(parabar, text = 'Sample Frequency', font = ('Arial', 11))
L.grid(row=6, column=0, sticky=tk.E, pady=spany)
E5 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E5.grid(row=6, column=1, sticky=tk.W, pady=spany)
E5.delete(0, tk.END)#删除Entry内容
E5.insert("insert",str(para['samplefrequeny']))

L = tk.Label(parabar, text = 'Window Column', font = ('Arial', 11))
L.grid(row=7, column=0, sticky=tk.E, pady=spany)
E6 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E6.grid(row=7, column=1, sticky=tk.W, pady=spany)
E6.delete(0, tk.END)#删除Entry内容
E6.insert("insert",str(para['wc']))

L = tk.Label(parabar, text = 'Moving Step', font = ('Arial', 11))
L.grid(row=8, column=0, sticky=tk.E, pady=spany)
E7 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E7.grid(row=8, column=1, sticky=tk.W, pady=spany)
E7.delete(0, tk.END)#删除Entry内容
E7.insert("insert",str(para['movingstep']))

L = tk.Label(parabar, text = 'Trainsample Days', font = ('Arial', 11))
L.grid(row=9, column=0, sticky=tk.E, pady=spany)
E8 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E8.grid(row=9, column=1, sticky=tk.W, pady=spany)
E8.delete(0, tk.END)#删除Entry内容
E8.insert("insert",str(para['trainsampledays']))

L = tk.Label(parabar, text = 'Feature Importance', font = ('Arial', 11))
L.grid(row=10, column=0, sticky=tk.E, pady=spany)
E9 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E9.grid(row=10, column=1, sticky=tk.W, pady=spany)
E9.delete(0, tk.END)#删除Entry内容
E9.insert("insert",str(para['featureimportance']))

L = tk.Label(parabar, text = 'PCA Feature', font = ('Arial', 11))
L.grid(row=11, column=0, sticky=tk.E, pady=spany)
com1 = ttk.Combobox(parabar, width = 8, font = ('Arial', 11))
com1.grid(row=11, column=1, sticky=tk.W, pady=spany)

L = tk.Label(parabar, text = 'Life Threshold', font = ('Arial', 11))
L.grid(row=12, column=0, sticky=tk.E, pady=spany)
E10 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E10.grid(row=12, column=1, sticky=tk.W, pady=spany)
E10.delete(0, tk.END)#删除Entry内容
E10.insert("insert",str(para['lifethreshold']))

L = tk.Label(parabar, text = 'ExpSmt Coef', font = ('Arial', 11))
L.grid(row=13, column=0, sticky=tk.E, pady=spany)
E11 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E11.grid(row=13, column=1, sticky=tk.W, pady=spany)

L = tk.Label(parabar, text = 'Num Of Predays', font = ('Arial', 11))
L.grid(row=14, column=0, sticky=tk.E, pady=spany)
E12 = tk.Entry(parabar, show = None, font = ('Arial', 11), width = 10)
E12.grid(row=14, column=1, sticky=tk.W, pady=spany)
E12.delete(0, tk.END)#删除Entry内容
E12.insert("insert",str(para['numofpredays']))

B0 = tk.Button(parabar, width=8, text = 'Save Para', font = ('Arial', 11), command = ThreadForSavePara)
B0.grid(row=15, column=0, sticky=tk.W, pady = spany)

B00 = tk.Button(parabar, width=8, text = 'Auto Proc', font = ('Arial', 11), command = ThreadForAutoStart)
B00.grid(row=15, column=1, sticky=tk.W, pady = spany)

###处理过程栏
spany1=41
stepbar = tk.Frame(MyWindow, relief='groove', bd=3)
stepbar.grid(row=1, column=1, sticky=tk.W)

L = tk.Label(stepbar, text = 'Processing Step', font = ('Arial', 11), fg='blue')
L.grid(row=1, column=0, sticky=tk.W, pady = 10)

B1 = tk.Button(stepbar, width=15, text = 'Get Sourcedata', font = ('Arial', 11), command = ThreadForGetSourceData)
B1.grid(row=2, column=0, sticky=tk.W, pady = spany1)

B2 = tk.Button(stepbar, width=15, text = 'Get Feature', font = ('Arial', 11), command = ThreadForGetFea)
B2.grid(row=3, column=0, sticky=tk.W, pady = spany1)

B3 = tk.Button(stepbar, width=15, text = 'Sort Feature', font = ('Arial', 11), command = ThreadForSortFea)
B3.grid(row=4, column=0, sticky=tk.W, pady = spany1)

B4 = tk.Button(stepbar, width=15, text = 'Get PCA Result', font = ('Arial', 11), command = ThreadForPCA)
B4.grid(row=5, column=0, sticky=tk.W, pady = spany1)

B5 = tk.Button(stepbar, width=15, text = 'Get RUL', font = ('Arial', 11), command = ThreadForGetRUL)
B5.grid(row=6, column=0, sticky=tk.W, pady = spany1)

###结果展示栏
wx = 308
hy = 280
resultbar = tk.Frame(MyWindow, relief='groove', bd=3, width=930, height=609)
resultbar.grid(row=1, column=2, padx = 11, pady = 9, sticky=tk.N)
resultbar.grid_propagate(False)

L = tk.Label(resultbar, text = 'Result Display', font = ('Arial', 11), fg='blue')
L.grid(row=0, column=0, sticky=tk.W, pady = 10)

fig1 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig1.grid(row=1, column=0)
fig1.grid_propagate(False)

fig2 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig2.grid(row=2, column=0)
fig2.grid_propagate(False)

fig3 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig3.grid(row=1, column=1)
fig3.grid_propagate(False)        

fig4 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig4.grid(row=2, column=1)
fig4.grid_propagate(False)

fig5 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig5.grid(row=1, column=3)
fig5.grid_propagate(False)

fig6 = tk.Frame(resultbar, relief='groove', bd=3, width=wx, height=hy)
fig6.grid(row=2, column=3)
fig6.grid_propagate(False)

MyWindow.mainloop()











