# -*- coding: utf-8 -*-
"""
Evaluating five different adaptive decomposition methods for EEG signal seizure detection and classification
Vinícius R. Carvalho,  Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes
Programa de Pós-Graduação em Engenharia Elétrica – Universidade Federal de Minas Gerais – Av. Antônio Carlos 6627, 31270-901, Belo Horizonte, MG, Brasil.

This script decomposes EEG signals from the Bonn University datbase (http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3)
and from the Neurology & Sleep Centre, Hauz Khas, New Delhi(https://www.researchgate.net/publication/308719109_EEG_Epilepsy_Datasets)
and decomposes each one according to five methods: Empirical Mode Decomposition (EMD), ensemble EMD,
complete EEMD with adaptive noise (CEEMDAN), Empirical Wavelet Transform (EWT) and Variational Mode Decomposition. 
Several features are then extracted from each decomposed mode and the resulting matrixes are
written in .csv files.

University of Bonn dataset should be on the folder"./data/BonnDataset/"
Neurology & Sleep Centre ND data should be on the folder "./data/NSC_ND"

@author: Vinicius Carvalho
vrcarva@ufmg.br
"""
#from __future__ import division # if python 2, uncomment this
import numpy as np
from scipy import signal
from PyEMD import EMD, EEMD, CEEMDAN
from scipy.stats import skew,kurtosis
from scipy.io import loadmat
from joblib import Parallel, delayed
import pandas as pd
import time
import ewtpy
from vmdpy import VMD
import matplotlib.pyplot as plt
import os

#Feature extraction fucntion
def featExtract(f,Fs, welchWin = 1024):
    """
    features,featLabels = featExtract(f,Fs, welchWin = 1024):
        f: input signal
        Fs: sampling frequency, in Hz
        welchWin: window size (in samples) for evaluating Welch's PSD, from which spectral features are calculated 
    Returns:
        features: calculated features
        featLabels: Feature labels - ["AM","BM","ent","pow","Cent","pk","freq","skew","kurt","Hmob","Hcomp"]
    """
    from scipy.ndimage.filters import gaussian_filter

    #AM and BM
    fhilbert = signal.hilbert(f)#hilbert transform
    fhilbert = fhilbert[150:-150]# to avoid border effects
    fphase = np.unwrap(np.angle(fhilbert))
    A = abs(fhilbert)#instantaneous amplitude
    inst_freq = np.diff(fphase)*Fs/(2*np.pi)#instantaneous frequency
    E = (np.linalg.norm(fhilbert)**2)/len(fhilbert)
    CW = np.sum(np.diff(fphase)*Fs*(A[0:-1]**2))/(2*np.pi*E)
    AM = np.sqrt(np.sum((np.diff(A)*Fs)**2))/E
    BM = np.sqrt(np.sum(((inst_freq-CW)**2)*(A[0:-1]**2))/E)
    
    #spectral features - Welch
    w, Pxx = signal.welch(f, Fs, nperseg = welchWin, noverlap = round(0.85*welchWin) )
    #smooth spectrum?
    #filterLen = 25
    #sigmaFilter = 5
    #regFilter = np.zeros(filterLen)
    #regFilter[regFilter.size//2] = 1 #prefer odd filter lengths - otherwise the gaussian is skewed
    #spectral features - FFT + gaussian filter (optional)
    #fy = np.fft.fft(f)
    #fy = abs(fy[0:int(np.ceil(fy.size/2))])#one-sided magnitude    
    #Pxx = np.convolve(fy,gaussian_filter(regFilter,sigmaFilter),mode = 'same')   
    #w = np.fft.fftfreq(f.size, 1/Fs)
    #w = w[0:int(np.ceil(w.size/2))]

    PxxNorm = Pxx/sum(Pxx)#normalized spectrum

    Sent = -sum(PxxNorm*np.log2(PxxNorm))#spectral entropy
    Spow = np.mean(Pxx**2)#spectral power
    Cent = np.sum(w*PxxNorm) #frequency centroid
    Speak = np.max(Pxx) #peak amplitude
    Sfreq = w[np.argmax(PxxNorm)]# peak frequency
    #skewness, kurtosis
    fskew = skew(f)
    fkurt = kurtosis(f)
    #Hjorth Parameters
    dy_f = np.diff(f)
    Hmob = np.sqrt(np.var(dy_f)/np.var(f))
    Hcomp =  np.sqrt(np.var(np.diff(dy_f))/np.var(dy_f))/Hmob
    
    features = [AM,BM,Sent,Spow,Cent,Speak,Sfreq,fskew,fkurt,Hmob, Hcomp]
    featLabels = ["AM","BM","ent","pow","Cent","pk","freq","skew","kurt","Hmob","Hcomp"]
    
    return features,featLabels

#load and extract features from all signals from the chosen database
def fun_loadExFeats(dbParams,ri,idx, item,Fs,Nchann,LPcutoff,Nmodes, FFTregLen = 25, gaussSigma = 5,FFTreg = 'gaussian'):
    #load eeg
    featNames = ["Group","decTime"]
    featsTuple = {"EMD":0,"EEMD":0,"CEEMDAN":0,"EWT":0,"VMD":0,"Orig":0}
    if dbase == "NSC_ND":
        fLoad = loadmat("%s/data/%s/%s%d"%(dbase,item,item,ri+1))
        f = fLoad[item][:,0]
        ltemp = int(np.ceil(f.size/2)) #to behave the same as matlab's round
        fMirr =  np.append(np.flip(f[0:ltemp-1],axis = 0),f)  
        fMirr = np.append(fMirr,np.flip(f[-ltemp-1:-1],axis = 0))
        f = np.copy(fMirr)
    if dbase == "BonnDataset":
        f = np.loadtxt("%s/data/%s/%s%.3d.txt"%(dbase,item,item,ri+1))

    #preprocessing - LP filter and remove DC
    f = f - np.mean(f)
    b, a = signal.butter(4, LPcutoff/ (0.5 * Fs), btype='low', analog=False)
    fp = signal.filtfilt(b, a, f)   
    
    #% EMD features
    tic = time.time()
    emd = EMD()
    emd.MAX_ITERATION = 2000
    IMFs = emd.emd(fp,max_imf = Nmodes)
    toc = time.time()
    featsTuple["EMD"] = toc-tic #execution time (decomposition)            
    if Nmodes != IMFs.shape[0]-1:
        print("\nCheck number of EMD modes: %s%.3d"%(item,ri+1))
    for mi in range(IMFs.shape[0]):
        featOut, labelTemp = featExtract(IMFs[mi,:], Fs, welchWin = 1024)
        featsTuple["EMD"] = np.append(featsTuple["EMD"], featOut)
        
        #write feature name header
        if ri == 0 and idx == 0:
            for ii in labelTemp:
                featNames = np.append(featNames,"%s%d"%(ii,mi))
    if IMFs.shape[0] < Nmodes+1:
        featsTuple["EMD"] = np.append(featsTuple["EMD"],np.zeros(Nfeats*(Nmodes+1-IMFs.shape[0])))
        

    #% EEMD - Ensemble Empirical Mode Decomposition
    tic = time.time()
    if __name__ == "__main__":  
        eemd = EEMD(trials = 200)
        eemd.MAX_ITERATION = 2000
        eIMFs = eemd(fp,max_imf = Nmodes)
    toc = time.time()
    featsTuple["EEMD"] = toc-tic #execution time (decomposition )   
    if Nmodes != eIMFs.shape[0]-1:
        print("\nCheck number of EEMD modes: %s%.3d"%(item,ri+1))    

    #for each mode, extract features
    for mi in range(eIMFs.shape[0]):
        featOut, labelTemp = featExtract(eIMFs[mi,:],Fs, welchWin = 1024)
        featsTuple["EEMD"]  = np.append(featsTuple["EEMD"],featOut)
    if eIMFs.shape[0] < Nmodes+1:
        featsTuple["EEMD"] = np.append(featsTuple["EEMD"],np.zeros(Nfeats*(Nmodes+1-eIMFs.shape[0])))
            
    #% CEEMDAN - Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
    tic = time.time()
    if __name__ == "__main__":            
        ceemdan = CEEMDAN()
        ceIMFs = ceemdan(fp,max_imf = Nmodes)
    toc = time.time()
    featsTuple["CEEMDAN"] = toc-tic #execution time (decomposition ) 
    if Nmodes != ceIMFs.shape[0]-1:
        print("\nCheck number of CEEMDAN modes: %s%.3d"%(item,ri+1))    

    #for each mode, extract features
    for mi in range(ceIMFs.shape[0]):
        featOut, labelTemp = featExtract(ceIMFs[mi,:],Fs, welchWin = 1024)
        featsTuple["CEEMDAN"]  = np.append(featsTuple["CEEMDAN"],featOut)
    if ceIMFs.shape[0] < Nmodes+1:
        featsTuple["CEEMDAN"] = np.append(featsTuple["CEEMDAN"],np.zeros(Nfeats*(Nmodes+1-ceIMFs.shape[0])))
                
    #%EWT features
    tic = time.time()
    ewt,_,_ = ewtpy.EWT1D(fp, N = Nmodes, log = 0,
                          detect = "locmax", 
                          completion = 0, 
                          reg = FFTreg, 
                          lengthFilter = FFTregLen,
                          sigmaFilter = gaussSigma )
    toc = time.time()
    featsTuple["EWT"]  = toc-tic #execution time (decomposition ) 
    if Nmodes != ewt.shape[1]:
        print("\nCheck number of EWT modes: %s%.3d"%(item,ri+1))        

    #for each mode, extract features
    for mi in range(Nmodes):
        featOut, labelTemp = featExtract(ewt[:,mi],Fs, welchWin = 1024)
        featsTuple["EWT"]   = np.append(featsTuple["EWT"] ,featOut)

    #% VMD features
    DC = np.mean(fp)   # no DC part imposed
    tic = time.time()
    vmd,_,_ = VMD(fp, alpha, tau, Nmodes, DC, init, tol)
    toc = time.time()
    featsTuple["VMD"]  = toc-tic #execution time (decomposition ) 
    if Nmodes != vmd.shape[0]:
        print("\nCheck number of VMD modes: %s%.3d"%(item,ri+1))        

    #for each mode, extract features
    for mi in range(Nmodes):
        featOut, labelTemp = featExtract(vmd[mi,:],Fs, welchWin = 1024)
        featsTuple["VMD"]  = np.append(featsTuple["VMD"] , featOut)
        
    #% Original non-decomposed signal features
    tic = time.time()
    featOut, labelTemp = featExtract(fp, Fs, welchWin = 1024)
    toc = time.time()
    featsTuple["Orig"]  = np.append(toc-tic, featOut)     

    return item, featsTuple
#%% main script

# General parameters
#which dataset
dataset = ["BonnDataset","NSC_ND"] #BonnDataset or NSC_ND

paramsData = {"BonnDataset": {"Nregs": 100, "Fs":173.61, 
                              "groups":["S","F","Z","N","O"],
                              "LPcutoff": 40, "channels":1,
                              "FFTregLen": 25, "gaussSigma": 5},#EWT parameters (for regularized spectrum)
              "NSC_ND": {"Nregs": 50, "Fs":200, 
                         "groups":["ictal","interictal","preictal"],
                         "LPcutoff": 70 ,"channels":1,
                         "FFTregLen": 10, "gaussSigma": 2}}#EWT parameters (for regularized spectrum)

saveFeats = 1 #if 1, save features to .csv files

#VMD parameters 
alpha = 2000 #      % moderate bandwidth constraint
tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
init = 1        #  % initialize omegas uniformly
tol = 1e-7 #

for Nmodes in [2,3,4,5,6,7,8]:#number of modes for decomposition 2 to 8
    print(Nmodes)
    
    featNames = ["Group","decTime"]
    featLabels = ["AM","BM","ent","pow","Cent","pk","freq","skew","kurt","Hmob","Hcomp"]
    Nfeats = len(featLabels) #number of features FOR EACH MODE - check if fits with featExtract
    for mi in range(Nmodes+1):
        for ii in featLabels:
            featNames = np.append(featNames,"%s%d"%(ii,mi))
    
    for dbase in dataset: #for each selected dataset
        Fs = paramsData[dbase]["Fs"]
        Nchann = paramsData[dbase]["channels"]
        LPcutoff = paramsData[dbase]["LPcutoff"]
    
        labels = np.empty(0)
        ParforOut = []
        featsEMD,featsEEMD,featsCEEMDAN,featsEWT,featsVMD,featsOrig = [],[],[],[],[],[]
        labels = []
        tic = time.time()
        for idx, item in enumerate(paramsData[dbase]["groups"]): #for each group/class
            print(item)
            #for each recording 
            #ParforOut = [fun_loadExFeats(paramsData[dbase],ri,idx, item,Fs,Nchann,LPcutoff,Nmodes) for ri in range(paramsData[dbase]["Nregs"])]
         
            ParforOut = Parallel(n_jobs=10,max_nbytes=None)(
                    delayed(fun_loadExFeats)(paramsData[dbase],ri,idx, item,Fs,Nchann,LPcutoff,Nmodes) for ri in range(paramsData[dbase]["Nregs"]))
    
            featsEMD.append(np.array([ParforOut[ri][1]["EMD"] for ri in  range(paramsData[dbase]["Nregs"])]))
            featsEEMD.append(np.array([ParforOut[ri][1]["EEMD"] for ri in  range(paramsData[dbase]["Nregs"])]))
            featsCEEMDAN.append(np.array([ParforOut[ri][1]["CEEMDAN"] for ri in  range(paramsData[dbase]["Nregs"])]))
            featsEWT.append(np.array([ParforOut[ri][1]["EWT"] for ri in  range(paramsData[dbase]["Nregs"])]))       
            featsVMD.append(np.array([ParforOut[ri][1]["VMD"] for ri in  range(paramsData[dbase]["Nregs"])]))   
            featsOrig.append(np.array([ParforOut[ri][1]["Orig"] for ri in  range(paramsData[dbase]["Nregs"])])) 
            labels.append([ParforOut[ri][0] for ri in  range(len(ParforOut))])
        toc = time.time()-tic
        print(toc)
        featsEMD = np.concatenate(featsEMD)  
        featsEEMD = np.concatenate(featsEEMD)    
        featsCEEMDAN = np.concatenate(featsCEEMDAN)
        featsEWT = np.concatenate(featsEWT)
        featsVMD = np.concatenate(featsVMD)
        featsOrig = np.concatenate(featsOrig)
        labels = np.concatenate(labels)
        
        #add group labels and save features to .csv  
        featsOrig = np.array(featsOrig,dtype = "O")    
        featsOrig = np.insert(featsOrig,0,labels,1)   
    
        featsEMD = np.array(featsEMD,dtype = "O")    
        featsEMD = np.insert(featsEMD,0,labels,1)
    
        featsEEMD = np.array(featsEEMD,dtype = "O")    
        featsEEMD = np.insert(featsEEMD,0,labels,1)
    
        featsCEEMDAN = np.array(featsCEEMDAN,dtype = "O")    
        featsCEEMDAN = np.insert(featsCEEMDAN,0,labels,1)
        
        featsEWT = np.array(featsEWT,dtype = "O")    
        featsEWT = np.insert(featsEWT,0,labels,1)
        
        featsVMD = np.array(featsVMD,dtype = "O")    
        featsVMD = np.insert(featsVMD,0,labels,1)      
        
        if saveFeats:
            with open("%s/EMDFeatsWelch_%dModes.csv"%(dbase,Nmodes), 'w') as fp1:
                 fp1.write(','.join(featNames) + '\n')
                 np.savetxt(fp1, featsEMD, '%s', ',')
            fp1.close()
            
            #save feature to .csv files
            with open("%s/EWTFeatsWelch_%dModes.csv"%(dbase,Nmodes), 'w') as fp2:
                 fp2.write(','.join(featNames[:-Nfeats]) + '\n')
                 np.savetxt(fp2, featsEWT, '%s', ',')
            fp2.close()
            
            #save feature to .csv files
            with open("%s/VMDFeatsWelch_%dModes.csv"%(dbase,Nmodes), 'w') as fp3:
                 fp3.write(','.join(featNames[:-Nfeats]) + '\n')
                 np.savetxt(fp3, featsVMD, '%s', ',')
            fp3.close()
    
            #save feature to .csv files
            with open("%s/EEMDFeatsWelch_%dModes.csv"%(dbase,Nmodes), 'w') as fp4:
                 fp4.write(','.join(featNames) + '\n')
                 np.savetxt(fp4, featsEEMD, '%s', ',')
            fp4.close()
    
            #save feature to .csv files
            with open("%s/CEEMDANFeatsWelch_%dModes.csv"%(dbase,Nmodes), 'w') as fp5:
                 fp5.write(','.join(featNames) + '\n')
                 np.savetxt(fp5, featsCEEMDAN, '%s', ',')
            fp5.close()
    
            with open("%s/ORIGFeatsWelch.csv"%(dbase), 'w') as fp6:
                 fp6.write(','.join(featNames[:2+Nfeats]) + '\n')
                 np.savetxt(fp6, featsOrig, '%s', ',')
            fp6.close()
        
         