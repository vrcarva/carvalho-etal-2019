# -*- coding: utf-8 -*-
"""
Plots figures on paper, as well as additional ones
@author: Vinicius Carvalho
"""
#from __future__ import division# if python 2
import numpy as np
from scipy import signal
from PyEMD import EMD
from vmdpy import VMD
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import ewtpy


def plotModes(modes,Nmode, Fs = 173.61, tlimits = [0,15]):
    #plot each signal in modes (each signal = 1 collumn)
    if Nmode <= 4:
        sbSize = [2,2]
    elif Nmode <= 6:
        sbSize = [3,2]
    else:
        sbSize = [4,2]
    tvec = np.linspace(0,modes.shape[0]/Fs,modes.shape[0])
    

    plt.figure()
    for ind in range(Nmode):
        ax = plt.subplot(sbSize[0],sbSize[1],ind+1)
        ax.plot(tvec,modes[:,ind],'k')    
        ax.autoscale(enable=True, axis='x', tight=True)  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        plt.title('Mode %d'%(ind+1))
        plt.xlim(tlimits)
        
#%% Figure 1: EEG signals and Figure X: Decompositions
Fs = 173.61#EEG sampling frequency
tlimits = [0,15]#time axis (s)
f = {}

#show spectrum in Fig1? 
spectrum = 1 #if 1, shows spectrum of each signal

#FFT parameters
FFTreg = 'gaussian'
FFTregLen = 25
gaussSigma = 5

# Decompostion figure parameters
Nmodes = [5,5,5] #(EMD,EWT,VMD)
#VMD parameters 
alpha = 2000 #      % moderate bandwidth constraint
tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
init = 1        #  % initialize omegas uniformly
tol = 1e-7 #

f["S"] = np.loadtxt("database/S/S001.txt")
f["F"] = np.loadtxt("database/F/F001.txt")
f["Z"] = np.loadtxt("database/Z/Z001.txt")
t = np.linspace(0,len(f["S"])/Fs,len(f["S"]))

#titulos = ["Ictal","Interictal","Seizure-free"]
titulos = ["a","b","c"]

plt.figure()
#preprocessing
for kk,ind in zip(f,range(len(f))):
    f[kk] = f[kk] - np.mean(f[kk])
    b, a = signal.butter(4, 40 / (0.5 * Fs), btype='low', analog=False)
    f[kk] = signal.filtfilt(b, a, f[kk])    
    
    if spectrum == 1:
        ff = np.fft.fft(f[kk])
        ff = abs(ff[:ff.size//2])#one-sided magnitude
        regFilter = np.zeros(FFTregLen)
        regFilter[regFilter.size//2] = 1 #prefer odd filter lengths - otherwise the gaussian is skewed
        presig = np.convolve(ff,gaussian_filter(regFilter,gaussSigma),mode = 'same')     
        fftFreq = np.fft.fftfreq(f[kk].size, 1/Fs)
        fftFreq = fftFreq[:fftFreq.size//2]
        ax = plt.subplot(3,2,(ind+1)*2)
        ax.plot(fftFreq,presig,'k')    
        ax.autoscale(enable=True, axis='x', tight=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if ind < 2:
            ax.get_xaxis().set_ticks([])
        else:
            plt.xlabel("Hz")
        ax.get_yaxis().set_ticks([])        
        plt.xlim((-1,60))
        #plt.title(titulos[(ind+1)*2-1])
        ax = plt.subplot(3,2,ind*2+1)
    else:
        
        ax = plt.subplot(3,1,ind+1)
    ax.plot(t,f[kk],linewidth = 1, color = 'k')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #plt.xlabel("s")
    #plt.ylabel("$\mu$V")
    plt.title(titulos[ind])
    plt.xlim(tlimits)
    
    #scale lines
    if kk== "S":
        #plt.ylim((-2000,1500))
        #plt.plot([tlimits[-1]-1,tlimits[-1]],[-2600, -2600],'k')
        plt.plot([tlimits[-1]-1,tlimits[-1]-1],[-1600, -2600],'k',linewidth = 2)
        
        #plt.plot([tlimits[0]+.1,tlimits[0]+1.1],[-1500, -1500],'k')
        #plt.plot([tlimits[0]+.1,tlimits[0]+.1],[-1800, -800],'k')        
        
    else:
        plt.ylim((-205,200))
    if kk== "F":
        #plt.plot([tlimits[-1]-1,tlimits[-1]],[-160, -160],'k')
        plt.plot([tlimits[-1]-1,tlimits[-1]-1],[-160, -60],'k',linewidth = 2)
        
        #plt.plot([tlimits[0]+.1,tlimits[0]+.1],[-200, -300],'k') 
        
    if kk== "Z":
        plt.plot([tlimits[0]+1,tlimits[0]+2],[-200, -200],'k',linewidth = 3)
        plt.plot([tlimits[-1]-1,tlimits[-1]-1],[-200, -100],'k',linewidth = 2)
    
#%% Fig 2: EWT spectrum segmentation
        
Fs = 173.61#EEG sampling frequency
#EWT parameters
FFTreg = 'gaussian'
FFTregLen = 25
gaussSigma = 5
f1 = np.loadtxt("database/S/S013.txt")
b, a = signal.butter(4, 40 / (0.5 * Fs), btype='low', analog=False)
f1 = signal.filtfilt(b, a, f1) 
f1 = f1 - np.mean(f1) 
ewt, mfb ,boundaries = ewtpy.EWT1D(f1, 
                      N = 5, 
                      log = 0,
                      detect = "locmax", 
                      completion = 0,
                      reg = FFTreg, 
                      lengthFilter = FFTregLen,
                      sigmaFilter = gaussSigma)
ff = np.fft.fft(f1)
ff = abs(ff[:ff.size//2])#one-sided magnitude
regFilter = np.zeros(FFTregLen)
regFilter[regFilter.size//2] = 1 #prefer odd filter lengths - otherwise the gaussian is skewed
presig = np.convolve(ff,gaussian_filter(regFilter,gaussSigma),mode = 'same')     
fftFreq = np.fft.fftfreq(f1.size, 1/Fs)
fftFreq = fftFreq[:fftFreq.size//2]
plt.figure()
ax = plt.subplot(111)
#ax.plot(fftFreq,ff)
ax.plot(fftFreq,presig,'k')
for bb in boundaries:
    bb = bb*Fs/(2*np.pi)
    ax.plot([bb,bb],[0,max(presig)],'r--',linewidth = 1.)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])
plt.xlim((0,50))
plt.xlabel('Hz')
#plt.ylabel("$\mu$V²/Hz")       
#%% FIG X?? Decomposition of each method
for kk,ind in zip(f,range(len(f))):
    #fig EMD
    emd = EMD()
    emd.MAX_ITERATION = 2000
    IMFs = emd.emd(f[kk],max_imf = Nmodes[0])
    plotModes(np.flip(IMFs.T,axis = 1),Nmodes[0], tlimits = tlimits)
    plt.suptitle('EMD, %s signal'%list(f.keys())[ind])
    
    
    #fig EWT
    ewt,_,_ = ewtpy.EWT1D(f[kk], N = Nmodes[1], 
                          log = 0,
                          detect = "locmax", 
                          completion = 0, 
                          reg = FFTreg, 
                          lengthFilter = FFTregLen,
                          sigmaFilter = gaussSigma )
    plotModes(ewt,Nmodes[1], tlimits = tlimits)
    plt.suptitle('EWT, %s signal'%list(f.keys())[ind])
    
    #fig VMD
    DC = np.mean(f[kk])   #          % no DC part imposed
    vmd,_,_ = VMD(f[kk], alpha, tau, Nmodes[2], DC, init, tol)
    plotModes(vmd.T,Nmodes[2], tlimits = tlimits)
    plt.suptitle('VMD, %s signal'%list(f.keys())[ind])
        
        
    
#%% EWT tests

# General parameters
Nmodes = [5,5,5] #number of modes for decomposition [EMD, EWT, VMD]

#EWT parameters
FFTreg = 'gaussian'
FFTregLen = 25
gaussSigma = 5

#VMD parameters 
alpha = 2000 #      % moderate bandwidth constraint
tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
init = 1        #  % initialize omegas uniformly
tol = 1e-7 #

# EWT
ewt = {}
mfb = {}
boundaries = {}

plt.figure()
for kk,ind in zip(f,range(len(f))):
    ewt[kk], mfb[kk] ,boundaries[kk] = ewtpy.EWT1D(f[kk], 
                          N = Nmodes[1], 
                          log = 0,
                          detect = "locmaxmin", 
                          completion = 0,
                          reg = FFTreg, 
                          lengthFilter = FFTregLen,
                          sigmaFilter = gaussSigma)
    ff = np.fft.fft(f[kk])
    ff = abs(ff[:ff.size//2])#one-sided magnitude
    regFilter = np.zeros(FFTregLen)
    regFilter[regFilter.size//2] = 1 #prefer odd filter lengths - otherwise the gaussian is skewed
    presig = np.convolve(ff,gaussian_filter(regFilter,gaussSigma),mode = 'same')     
    fftFreq = np.fft.fftfreq(f[kk].size, 1/Fs)
    fftFreq = fftFreq[:fftFreq.size//2]
    ax = plt.subplot(3,1,ind+1)
    ax.plot(fftFreq,presig)
    for bb in boundaries[kk]:
        bb = bb*Fs/(2*np.pi)
        ax.plot([bb,bb],[0,max(presig)],'--r')
    

#%% Test figure: feature projection
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mpldatacursor import HighlightingDataCursor, DataCursor

#class for iteractive plot - select points in one subplot highlights it on the others
class IndexedHighlight(HighlightingDataCursor):
    def __init__(self, axes, **kwargs):
        # Use the first plotted Line2D in each axes
        artists = [ax.lines[0] for ax in axes]

        kwargs['display'] = 'single'
        HighlightingDataCursor.__init__(self, artists, **kwargs)
        self.highlights = [self.create_highlight(artist) for artist in artists]
        plt.setp(self.highlights, visible=False)

    def update(self, event, annotation):
        # Hide all other annotations
        plt.setp(self.highlights, visible=False)

        # Highlight everything with the same index.
        artist, ind = event.artist, event.ind
        for original, highlight in zip(self.artists, self.highlights):
            x, y = original.get_data()
            highlight.set(visible=True, xdata=x[ind], ydata=y[ind])
        DataCursor.update(self, event, annotation)

Nmodes = 6
scaleFeats = 1
groups = ["S","F","Z"]
decMethod = ["VMD"] #EMD, EWT OR VMD

for mds in decMethod:
    dfFeats = pd.read_csv('%sFeats_%dModes.csv'%(mds,Nmodes), sep=',',header=0)
    #only selected groups
    dfFeats = dfFeats[dfFeats["Group"].isin(groups)]
    x = dfFeats.iloc[:, 1:]
    y = dfFeats.iloc[:, 0].values
    if scaleFeats:#scaler
        x = StandardScaler().fit_transform(x)
    
    
    x_TSNE = TSNE(n_components=2).fit_transform(x)
    pca = PCA(n_components=2)
    x_PCA = pca.fit_transform(x)
    
    #Projection of all features
    plt.figure()
    for ci in groups:
        indxs = y==ci
        plt.scatter(x_PCA[indxs,0],x_PCA[indxs,1])
    plt.legend(groups)
    plt.title("PCA Projected Features - %s"%mds)
    
    plt.figure()
    for ci in groups:
        indxs = y==ci
        plt.scatter(x_TSNE[indxs,0],x_TSNE[indxs,1])
    plt.legend(groups)
    plt.title("T-SNE Projected Features - %s"%mds)
    
    #plot projections of each mode separately - **iteractive plot**
    fig, axes =  plt.subplots(nrows=2, ncols=3)
    axflats = axes.flat
    for mi in range(Nmodes):
        fidxs = [a[-1]== '%d'%mi for a in list(dfFeats.columns.values)]
        #CHOOSE: T-SNE OR PCA
        #x_proj = TSNE(n_components=2).fit_transform(x[:,fidxs[1:]])
        pca = PCA(n_components=2)
        x_proj = pca.fit_transform(x[:,fidxs[1:]])    
        
        axflats[mi].plot(x_proj[:,0],x_proj[:,1],'.',c=[1,1,1],markersize = 3, label = '_nolegend_')
        for ci in groups:
            indxs = y==ci
            axflats[mi].plot(x_proj[indxs,0],x_proj[indxs,1],'.',markersize = 5)   
        axflats[mi].set_title("T-SNE: %s, mode %d"%(mds,mi+1)) #CHOOSE: title, tsne or pca
        
        
    IndexedHighlight(axflats[:Nmodes], point_labels=[str(i) for i in range(20)])    
    axflats[mi].legend(groups)







