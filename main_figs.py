# -*- coding: utf-8 -*-
"""
Plots figures on paper.
@author: Vinicius Carvalho
"""

#from __future__ import division# if python 2
import numpy as np
from scipy import signal
from PyEMD import EMD, EEMD, CEEMDAN
from vmdpy import VMD
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import ewtpy
from scipy.io import loadmat

def plotModes(modes,Nmode, Fs = 173.61, tlimits = [0,20]): #plots signal IMFs/modes
    if Nmode <= 6:
        sbSize = [Nmode,1]#
    else:
        sbSize = [4,2]
    tvec = np.linspace(0,modes.shape[0]/Fs,modes.shape[0])
    
    plt.figure(figsize = (1.92,2.5))
    for ind in range(Nmode):
        ax = plt.subplot(sbSize[0],sbSize[1],ind+1)
        ax.plot(tvec,modes[:,ind],'k',linewidth = 0.7)    
        ax.autoscale(enable=True, axis='x', tight=True)  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        #ax.spines['left'].set_visible(False)
        ax.get_xaxis().set_ticks([])
        #ax.get_yaxis().set_ticks([])
        #plt.title('Mode %d'%(ind+1))
        plt.xlim(tlimits)
    plt.tight_layout()
        
#%% EEG segments + spectra: Figs 1 & 2 (UoB and NSC-ND)

paramsData = {"BonnDataset": {"Nregs": 100, "Fs":173.61, "groups":["S","F","Z"],"LPcutoff": 40, "tlimits":[1,6]},
              "NSC_ND": {"Nregs": 50, "Fs":200, "groups":["ictal","interictal","preictal"],"LPcutoff": 70 ,"tlimits":[0,5.35]}}
dBase = "NSC_ND" #"NSC_ND" or "BonnDataset"
#show spectrum in Fig1? 
spectrum = 1 #if 1, shows spectrum of each signal

#FFT parameters
FFTreg = 'gaussian'
if dBase == "BonnDataset":
    FFTregLen = 25
    gaussSigma = 5
else:
    FFTregLen = 5
    gaussSigma = 1    

PLTtitles = {"BonnDataset": ["Ictal","Interictal","Normal"], "NSC_ND":["Ictal","Interictal","Preictal"]}
#titulos = ["a","b","c"]
Fs = paramsData[dBase]["Fs"]

f = {}
for item in paramsData[dBase]["groups"]:
    if dBase == "NSC_ND":
        fLoad = loadmat("%s/data/%s/%s%d"%(dBase,item,item,12))
        f[item] = fLoad[item][:,0]
    if dBase == "BonnDataset":
        f[item] = np.loadtxt("%s/data/%s/%s%.3d.txt"%(dBase,item,item,1))   
t = np.linspace(0,len(f[paramsData[dBase]["groups"][0]])/Fs,len(f[paramsData[dBase]["groups"][0]]))

plt.figure(figsize = (3.54,2.7))#(3.54,2.43) or (5.51,3.8)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams.update({'font.size': 7})
#preprocessing
for kk,ind in zip(f,range(len(f))):
    f[kk] = f[kk] - np.mean(f[kk])
    b, a = signal.butter(4, paramsData[dBase]["LPcutoff"] / (0.5 * Fs), btype='low', analog=False)
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
        ax.plot(fftFreq,presig,'k',linewidth = 1.8)    
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
        plt.xlim((-1,40))
        #plt.title(titulos[(ind+1)*2-1])
        plt.title("{} spectrum".format(PLTtitles[dBase][ind]),fontsize = 10)
        ax = plt.subplot(3,2,ind*2+1)
    else:
        ax = plt.subplot(3,1,ind+1)
    ax.plot(t,f[kk],linewidth = 0.6, color = 'k')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #plt.xlabel("s")
    #plt.ylabel("$\mu$V")
    plt.title("{} EEG".format(PLTtitles[dBase][ind]),fontsize = 10)
    plt.xlim(paramsData[dBase]["tlimits"])
     #scale lines
    if dBase == "BonnDataset":
        if kk== "S":
            #plt.ylim((-2000,1500))
            #plt.plot([tlimits[-1]-1,tlimits[-1]],[-2600, -2600],'k')
            plt.plot([paramsData[dBase]["tlimits"][-1]-1,paramsData[dBase]["tlimits"][-1]-1],[-1600, -2600],'k',linewidth = 2)
            #plt.plot([tlimits[0]+.1,tlimits[0]+1.1],[-1500, -1500],'k')
            #plt.plot([tlimits[0]+.1,tlimits[0]+.1],[-1800, -800],'k')         
        else:
            plt.ylim((-205,200))
        if kk== "F":
            #plt.plot([tlimits[-1]-1,tlimits[-1]],[-160, -160],'k')
            plt.plot([paramsData[dBase]["tlimits"][-1]-1,paramsData[dBase]["tlimits"][-1]-1],[-160, -60],'k',linewidth = 2)
            #plt.plot([tlimits[0]+.1,tlimits[0]+.1],[-200, -300],'k') 
        if kk== "Z":
            plt.plot([paramsData[dBase]["tlimits"][0]+1,paramsData[dBase]["tlimits"][0]+2],[-200, -200],'k',linewidth = 3)
            plt.plot([paramsData[dBase]["tlimits"][-1]-1,paramsData[dBase]["tlimits"][-1]-1],[-200, -100],'k',linewidth = 2)
    if dBase == "NSC_ND":  
        if kk== "ictal":
            plt.plot([paramsData[dBase]["tlimits"][-1]-0.1,paramsData[dBase]["tlimits"][-1]-0.1],[-500, 0],'k',linewidth = 2)       

        if kk == "interictal":
            plt.plot([paramsData[dBase]["tlimits"][-1]-0.1,paramsData[dBase]["tlimits"][-1]-0.1],[-25, 0],'k',linewidth = 2)
        if kk== "preictal":
            plt.plot([paramsData[dBase]["tlimits"][0]+1,paramsData[dBase]["tlimits"][0]+2],[-70, -70],'k',linewidth = 3)
            plt.plot([paramsData[dBase]["tlimits"][-1]-0.1,paramsData[dBase]["tlimits"][-1]-0.1],[-50, 0],'k',linewidth = 2)
plt.tight_layout()

plt.savefig("FigEEG_%s_90mm.pdf"%dBase)
    
#%% Fig 4: EWT spectrum segmentation
        
Fs = 173.61#EEG sampling frequency
#EWT parameters
FFTreg = 'gaussian'
FFTregLen = 25
gaussSigma = 5
plt.rcParams.update({'font.size': 8})
f1 = np.loadtxt("BonnDataset/data/S/S013.txt")
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
plt.figure(figsize = (3.54,2.5))
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
plt.tight_layout()
plt.savefig("Fig4.pdf",dpi = 300)
plt.savefig("Fig4.tiff",dpi = 300)
#plt.ylabel("$\mu$V²/Hz")       

#%% Fig 5
#%% Fig 5 get execution times for each decomposition method
from numpy import genfromtxt
import seaborn as sns
Nmodes_list = [2,3,4,5,6,7,8]
dBase = "NSC_ND"
methds = ["EMD","EEMD","CEEMDAN","EWT","VMD"]
dfTime = [0]*len(Nmodes_list)
for idxN, Nmodes in enumerate(Nmodes_list):
    
    AA = [0]*len(methds)
    for mi, met in enumerate(methds):
        AA[mi] = {}
        AA[mi]["Time"] = genfromtxt('%s/%sFeatsWelch_%dModes.csv'%(dBase,met,Nmodes), delimiter=',')[1:,1]
        AA[mi] = pd.DataFrame.from_dict(AA[mi])
        AA[mi]["Method"] = met

    dfTime[idxN] = pd.concat(AA)
    dfTime[idxN].insert(2,"Nmodes",Nmodes)
    
dfTime = pd.concat(dfTime, ignore_index = True)
    
plt.figure(figsize = (7,1.6))
for mi, met in enumerate(methds):
    ax1 =plt.subplot(1,5,mi+1)
    sns.boxplot(data = dfTime[dfTime["Method"]==met], y = "Time", x = "Nmodes",
                fliersize = 1, color = ".25",linewidth = 1)    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)   
    if mi >= 1:
        ax1.set_ylabel('')  
    else:
        ax1.set_ylabel('sec')
    plt.xlabel('Modes')
    plt.title(met)
plt.tight_layout()    
plt.savefig("ExecTime_%s.pdf"%dBase)

#%% Figs 6,7: EEG segment decomposition
# Decompostion figure parameters
Nmodes = [3,3,3,4,3] #(EMD,EEMD, CEEMDAN, EWT,VMD)
#Nmodes = [2,4,2,5,6] #(EMD,EEMD, CEEMDAN, EWT,VMD)
#VMD parameters 
alpha = 2000 #      % moderate bandwidth constraint
tau = 0       #     % noise-tolerance (no strict fidelity enforcement)
init = 1        #  % initialize omegas uniformly
tol = 1e-7 #

for kk,ind in zip(f,range(len(f))):
    #fig EMD
    emd = EMD()
    emd.MAX_ITERATION = 2000
    IMFs = emd.emd(f[kk],max_imf = Nmodes[0])
    plotModes(np.flip(IMFs.T,axis = 1),Nmodes[0]+1, tlimits = paramsData[dBase]["tlimits"])
    #plt.suptitle('EMD, %s signal'%list(f.keys())[ind])
    plt.savefig("DecEMD%d_%s_%s.pdf"%(Nmodes[0],dBase,kk))
    
    #EEMD
    if __name__ == "__main__":  
        eemd = EEMD()
        eemd.MAX_ITERATION = 2000
        eIMFs = eemd(f[kk],max_imf = Nmodes[1])
    plotModes(np.flip(eIMFs.T,axis = 1),Nmodes[1]+1, tlimits = paramsData[dBase]["tlimits"])
    #plt.suptitle('EMD, %s signal'%list(f.keys())[ind])
    plt.savefig("DecEEMD%d_%s_%s.pdf"%(Nmodes[1],dBase,kk))

    #CEEMDAN 
    if __name__ == "__main__":            
        ceemdan = CEEMDAN()
        ceIMFs = ceemdan(f[kk],max_imf = Nmodes[2])    
    plotModes(np.flip(ceIMFs.T,axis = 1),Nmodes[2]+1, tlimits = paramsData[dBase]["tlimits"]) 
    plt.savefig("DecCEEMDAN%d_%s_%s.pdf"%(Nmodes[2],dBase,kk))
    
    #fig EWT
    ewt,_,_ = ewtpy.EWT1D(f[kk], N = Nmodes[3], 
                          log = 0,
                          detect = "locmax", 
                          completion = 0, 
                          reg = FFTreg, 
                          lengthFilter = FFTregLen,
                          sigmaFilter = gaussSigma )
    plotModes(ewt,Nmodes[3], tlimits = paramsData[dBase]["tlimits"])
    #plt.suptitle('EWT, %s signal'%list(f.keys())[ind])
    plt.savefig("DecEWT%d_%s_%s.pdf"%(Nmodes[3],dBase,kk))
    #fig VMD
    DC = np.mean(f[kk])   #          % no DC part imposed
    vmd,_,_ = VMD(f[kk], alpha, tau, Nmodes[4], DC, init, tol)
    plotModes(vmd.T,Nmodes[4], tlimits = paramsData[dBase]["tlimits"])
    #plt.suptitle('VMD, %s signal'%list(f.keys())[ind])
    plt.savefig("DecVMD%d_%s_%s.pdf"%(Nmodes[4],dBase,kk))   
#
for fi in f:
    plt.figure(figsize = (5.4,2))
    ax = plt.subplot(111)
    ax.plot(t,f[fi],'k')
    ax.autoscale(enable=True, axis='x', tight=True)  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    plt.xlim(paramsData[dBase]["tlimits"])
    plt.locator_params(axis='y', nbins=4)
    plt.ylabel("$\mu$V")
    plt.tight_layout()
    plt.savefig("DecOrig_%s_%s.pdf"%(dBase,fi))
        
     

#%% Interactive plots: feature projection
    
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

"""#%% EWT tests

# General parameters
Nmodes = [6,6,5] #number of modes for decomposition [EMD, EWT, VMD]

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
"""





