from fastai.vision import Learner, Tensors, Rank0Tensor, flatten_model, listify
from torch import nn
import torch
from fastai.metrics import error_rate
from fastai.callbacks import HookCallback
import numpy as np
import math
import matplotlib.pyplot as plt

def splitAtFirstParenthesis(s,showDetails,shapeData):
    pos=len(s.split('(')[0])
    ret = s[:pos]
    if (showDetails):
        ret += shapeData + '\n' + s[pos:]
    return ret

class ActivationsHistogram(HookCallback):
    "Callback that record histogram of activations."
    "NOTE: the property name will be 'activations_histogram'"

    def __init__(self, learn:Learner, do_remove:bool=True,
                hMin=-1,
                hMax=1,
                nBins=100,
                useClasses=False, # if true compute histogram of classes in the last layer
                liveChart = True, # show live chart of last layer  
                modulesId=-1 # array of modules to keep
                ):
        self.hMin = hMin or (-hMax)
        self.hMax = hMax
        self.nBins = nBins
        self.liveChart = liveChart
        #self.allModules = [m for m in flatten_model(learn.model) if hasattr(m, 'weight')]
        self.allModules = [m for m in flatten_model(learn.model)]
        self.useClasses = useClasses
        #modules = None
        modules = self.allModules
        if modulesId:
            modules = [self.allModules[i] for i in listify(modulesId)]
        self.allModules = modules if modules else self.allModules
        self.c = learn.data.c # Number of Calsses
        super().__init__(learn,modules,do_remove)
        
    def mkHist(self, x, useClasses):
        ret = x.clone().detach().cpu() # Need this for Pytorch 1.0
        #ret = x.clone().detach() #Pytorch 1.1 # WARNING: so bad performance on GPU! (x10)
        if useClasses:
            ret = torch.stack([ret[:,i].histc(self.nBins, self.hMin, self.hMax) for i in range(ret.shape[1])],dim=1) #histogram by class...
        else:
            ret = ret.histc(self.nBins, self.hMin, self.hMax).unsqueeze(1) # histogram by activation
        return ret

    def on_train_begin(self, **kwargs):
        "Initialize stats."
        super().on_train_begin(**kwargs)
        self.stats_hist = None # empty at start
        self.stats_valid_hist = None # empty at start
        self.stats_epoch = []
        self.cur_epoch = -1
        self.cur_train_batch = -1
        self.stats_valid_epoch = []
        self.shape_out={}

    def on_epoch_begin(self, **kwargs):
        self.cur_epoch += 1

    def on_batch_begin(self, train, **kwargs):
        if train: 
            self.cur_train_batch += 1

    def hook(self, m:nn.Module, i:Tensors, o:Tensors)->Rank0Tensor:
        if (isinstance(o,torch.Tensor)) and (m not in self.shape_out):
            outShape = o.shape; 
            self.shape_out[m]=outShape;
        return self.mkHist(o,self.useClasses)
    
    def on_batch_end(self, train, **kwargs):
        "Take the stored results and puts it in `self.stats_hist`"
        hasValues = True if ((len(self.hooks.stored)>0) and (not (self.hooks.stored[0] is None))) else False
        stacked = torch.stack(self.hooks.stored).unsqueeze(1)  if hasValues else None
        if train and hasValues:
            if self.stats_hist is None: self.stats_hist = stacked #start
            else: self.stats_hist = torch.cat([self.stats_hist,stacked],dim=1) #cat
        if (not train) and hasValues:
            if self.stats_valid_hist is None: self.stats_valid_hist = stacked #start
            else: self.stats_valid_hist = torch.cat([self.stats_valid_hist,stacked],dim=1) #cat
            
    def on_epoch_end(self, **kwargs):
        self.stats_epoch.append(self.cur_train_batch)
        start = 0 if 1==len(self.stats_epoch) else self.stats_epoch[-2]
        end = self.stats_epoch[-1]
        startValid = 0 if 0==len(self.stats_valid_epoch) else self.stats_valid_epoch[-1] 
        if not(self.stats_hist is None):
            hScale = 1
            domain = self.stats_hist[-1][start:end]
            yy = np.arange(self.hMin,self.hMax,(self.hMax-self.hMin)/self.nBins)
            xx,_ = self.computeXY(domain,hScale,.25,start) # average on last quarter of epoch
            xx = xx.sum(-1) # useClasses support
            toDisplay = [(xx,yy)]
            
            if not (self.stats_valid_hist is None):
                domainValid = self.stats_valid_hist[-1][startValid:] #till end
                xxv,_ = self.computeXY(domainValid,hScale,1,start) # validation average all available data
                xxv = xxv.sum(-1) # useClasses support
                toDisplay += [(xxv,yy)]
                self.stats_valid_epoch += [self.stats_valid_hist.shape[1]]
            if self.liveChart:
                rec = rec = self.learn.recorder
                rec.pbar.update_graph(toDisplay) #, (0,max(xx)), (self.hMin,self.hMax))            
        
    def on_train_end(self, **kwargs):
        "Polish the final result."
        super().on_train_end(**kwargs)
        
    @staticmethod
    def get_color_value_from_map(idx:int, cmap='Reds', scale=1):
        return plt.get_cmap(cmap)(idx*scale)

    @staticmethod
    def getHistImg(act,useClasses):
        dd = act.squeeze(2) if not useClasses else act.sum(dim=2) # Reassemble...
        dd = dd.log() # Scale for visualizaion
        dd = dd.t() # rotate
        #dd = dd.flip(0) # Swap pos/neg Y
        return dd

    @staticmethod
    def computeXY(l,hscale,perc,hshift=0):
        start = int(l.shape[0]*(1-perc))
        end = l.shape[0]
        m = l[start:end].mean(dim=0) # all data mean
        xx = hshift + m*hscale
        yy = +np.array(range(l.shape[1]))
        return xx,yy

    @staticmethod
    def plotPerc(ax,l,hscale,perc,hshift=0,colorById=False,linewidth=1,addLabel=False):
        xx,yy = ActivationsHistogram.computeXY(l,hscale,perc,hshift)
        if colorById:
            classes = xx.shape[1] 
            for i in range(classes):
                xx_cur = xx[:,i]
                color = ActivationsHistogram.get_color_value_from_map(i/classes, cmap='rainbow')
                label = i if addLabel else None
                ax.plot(xx_cur,yy,linewidth=linewidth, color=color, label=label);
        else:
            color = [1-perc,1-perc,1-perc]
            ax.plot(xx,yy,linewidth=linewidth, color=color);

    def plotActsHist(self, cols=3, figsize=(20,10), toDisplay=None, hScale = .05, showEpochs=True, showLayerInfo=False, aspectAuto=True):
        histsTensor = self.activations_histogram.stats_hist
        hists = [histsTensor[i] for i in range(histsTensor.shape[0])]
        if toDisplay: hists = [hists[i] for i in listify(toDisplay)] # optionally focus

        n=len(hists)
        cols = cols or 3
        cols = min(cols,n)
        rows = int(math.ceil(n/cols))
        fig = plt.figure(figsize=figsize)
        grid = plt.GridSpec(rows, cols, figure=fig)
        #fig.suptitle('Activations Y:log(histogram) X:batch', fontsize=16)
        #axs = axs.flatten() if getattr(axs,'flatten',None) else [axs]

        for i,l in enumerate(hists):
            img=self.getHistImg(l,self.useClasses)
            cr = math.floor(i/cols)
            cc = i%cols
            main_ax = fig.add_subplot(grid[cr,cc])
            main_ax.imshow(img); 
            layerId = listify(toDisplay)[i] if toDisplay else i 
            m = self.allModules[layerId]
            outShapeText = f'  (out: {list(self.shape_out[m])})' if (m in self.shape_out) else ''
            title = f'L:{layerId}' + '\n' + splitAtFirstParenthesis(str(m),showLayerInfo,outShapeText)
            main_ax.set_title(title)
            imgH=img.shape[0]
    #        plt.set_yticks([0,imgH/2,imgH],(str(-GLOBAL_HIST_AMPLITUDE),'0',str(GLOBAL_HIST_AMPLITUDE)))
            main_ax.set_yticks([])
            main_ax.set_ylabel(str(self.hMin) + " : " + str(self.hMax))
            if aspectAuto: main_ax.set_aspect('auto')
            imgW=img.shape[1]
            imgH=img.shape[0]
            ratioH=-self.hMin/(self.hMax-self.hMin)
            zeroPosH = imgH*ratioH
            main_ax.plot([0,imgW],[zeroPosH,zeroPosH],'r') # X Axis
            if (showEpochs):
                start = 0
                nEpochs = len(self.activations_histogram.stats_epoch)
                for i,hh in enumerate(self.activations_histogram.stats_epoch):
                    if(i<(nEpochs-1)): main_ax.plot([hh,hh],[0,imgH],color=[0,0,1])
                    end = hh # rolling
                    domain = l[start:end]
                    #from pdb import set_trace; set_trace() #DEBUG
                    domain_mean = domain.mean(-1) # mean on classes
                    if self.useClasses:
                        self.plotPerc(main_ax,domain,hScale,1,start,colorById=True,addLabel=(0==i)) #plot all
                        main_ax.legend(loc='upper left')
                    else:
                        self.plotPerc(main_ax,domain_mean,hScale,.5,start)
                    self.plotPerc(main_ax,domain_mean,hScale,1,start,linewidth=1.5)
                    start = hh
            main_ax.set_xlim([0,imgW])
            main_ax.set_ylim([0,imgH])