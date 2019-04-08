# -*- coding: utf-8 -*-
"""
Evaluating adaptive decomposition methods for EEG signal seizure detection and classification
Vinícius R. Carvalho,  Márcio F.D. Moraes, Antônio P. Braga, Eduardo M.A.M. Mendes
Programa de Pós-Graduação em Engenharia Elétrica – Universidade Federal de Minas Gerais – Av. Antônio Carlos 6627, 31270-901, Belo Horizonte, MG, Brasil.

This script decomposes EEG signals from the Bonn University datbase (http://epileptologie-bonn.de/cms/front_content.php?idcat=193&lang=3&changelang=3)
and decomposes each one according to three methods: Empirical Mode Decomposition (EMD),
Empirical Wavelet Transform (EWT) and Variational Mode Decomposition. 
Several features are then extracted from each decomposed mode and the resulting matrixes are
written in .csv files

@author: Vinicius Carvalho
"""
#from __future__ import division # if python 2, uncomment this
import numpy as np
from scipy import signal
from PyEMD import EMD
from scipy.stats import skew,kurtosis
import pandas as pd

import ewtpy
from vmdpy import VMD


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
    filterLen = 25
    sigmaFilter = 5
    regFilter = np.zeros(filterLen)
    regFilter[regFilter.size//2] = 1 #prefer odd filter lengths - otherwise the gaussian is skewed
    
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

#%% main script

# General parameters
Nmodes = [7,7,7] #number of modes for decomposition [EMD, EWT, VMD]
Nfeats = 11 #number of features FOR EACH MODE - check if fits with featExtract
LPorder = 6 #lowpass filter order
LPcutoff = 40 #lowpass filter cutoff frequency (Hz)
saveFeats = 1 #if 1, save features to .csv files

#EWT parameters (for regularized spectrum)
FFTreg = 'gaussian'
FFTregLen = 25
gaussSigma = 5

#VMD parameters 
alpha = 2000 #      % moderate bandwidth constraint
tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
init = 1        #  % initialize omegas uniformly
tol = 1e-7 #

groups = ["S","F","Z","N","O"] #select desired groups
Nregs = 100 #number of eeg recordings for each group
Fs = 173.61#EEG sampling frequency

featsEMD = np.empty((0,Nmodes[0]*Nfeats))
featsEWT = np.empty((0,Nmodes[1]*Nfeats))
featsVMD = np.empty((0,Nmodes[2]*Nfeats))
labels = np.empty(0)

featNames = ["Group"]
#load eeg data
for idx, item in enumerate(groups):
    for ri in range(Nregs):
        print("\n%s%.3d"%(item,ri+1))
        #load eeg
        f = np.loadtxt("database/%s/%s%.3d.txt"%(item,item,ri+1))
        #preprocessing - LP filter and remove DC
        f = f - np.mean(f)
        b, a = signal.butter(LPorder, LPcutoff / (0.5 * Fs), btype='low', analog=False)
        fp = signal.filtfilt(b, a, f)        
        
        #%% EMD features
        emd = EMD()
        emd.MAX_ITERATION = 2000
        IMFs = emd.emd(fp,max_imf = Nmodes[0])
        if Nmodes[0] != IMFs.shape[0]-1:
            print("\nCheck number of EMD modes: %s%.3d"%(item,ri+1))
            
        featTemp = np.empty((0))
        for mi in range(Nmodes[0]):
            featOut, labelTemp = featExtract(IMFs[mi,:], Fs, welchWin = 1024)
            featTemp = np.append(featTemp, featOut, axis = 0)
            #write feature name header
            if ri == 0 and idx == 0:
                for ii in labelTemp:
                    featNames = np.append(featNames,"%s%d"%(ii,mi))
        featsEMD = np.append(featsEMD,[featTemp], axis = 0)   
        labels = np.append(labels,item)
        
        #%%EWT features
        ewt,_,_ = ewtpy.EWT1D(fp, N = Nmodes[1], log = 0,detect = "locmax", completion = 0, reg = FFTreg, lengthFilter = FFTregLen,sigmaFilter = gaussSigma )
        if Nmodes[1] != ewt.shape[1]:
            print("\nCheck number of EWT modes: %s%.3d"%(item,ri+1))        
        featTemp = np.empty((0))
        #for each mode, extract features
        for mi in range(Nmodes[1]):
            featOut, labelTemp = featExtract(ewt[:,mi],Fs, welchWin = 1024)
            featTemp  = np.append(featTemp,featOut ,axis = 0)
        featsEWT = np.append(featsEWT,[featTemp], axis = 0)   
        
        #%% VMD features
        DC = np.mean(f)   # no DC part imposed
        vmd,_,_ = VMD(fp, alpha, tau, Nmodes[2], DC, init, tol)
        if Nmodes[2] != vmd.shape[0]:
            print("\nCheck number of VMD modes: %s%.3d"%(item,ri+1))        
        featTemp = np.empty((0))
        #for each mode, extract features
        for mi in range(Nmodes[1]):
            featOut, labelTemp = featExtract(vmd[mi,:],Fs, welchWin = 1024)
            featTemp = np.append(featTemp, featOut,axis = 0)
        featsVMD = np.append(featsVMD,[featTemp], axis = 0)   
       
#add group labels and save features to .csv    
featsEMD = np.array(featsEMD,dtype = "O")    
featsEMD = np.insert(featsEMD,0,labels,1)

featsEWT = np.array(featsEWT,dtype = "O")    
featsEWT = np.insert(featsEWT,0,labels,1)

featsVMD = np.array(featsVMD,dtype = "O")    
featsVMD = np.insert(featsVMD,0,labels,1)        
if saveFeats:
    with open("EMDFeatsWelch_%dModes.csv"%Nmodes[0], 'w') as fp1:
         fp1.write(','.join(featNames) + '\n')
         np.savetxt(fp1, featsEMD, '%s', ',')
    fp1.close()
    
    #save feature to .csv files
    with open("EWTFeatsWelch_%dModes.csv"%Nmodes[1], 'w') as fp2:
         fp2.write(','.join(featNames) + '\n')
         np.savetxt(fp2, featsEWT, '%s', ',')
    fp2.close()
    
    #save feature to .csv files
    with open("VMDFeatsWelch_%dModes.csv"%Nmodes[2], 'w') as fp3:
         fp3.write(','.join(featNames) + '\n')
         np.savetxt(fp3, featsVMD, '%s', ',')
    fp3.close()



    
    