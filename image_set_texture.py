#%% import necessary modules
import numpy as np; 
import matplotlib.pyplot as plt; 
import pandas as pd
from scipy import interpolate
from scipy import signal
from scipy.stats import mannwhitneyu
from IPython import display        # for update plot in a loop 
from suite2p.io import BinaryFile
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import time

#%%
### get experiment name
expName = '135_007_04'; #'134_016_03','135_001_04','135_002_08'
monk_num = expName[:3]; 
site_num = expName[4:7]; 
run_num = expName[8:10]; 

data_path = '/Volumes/TJ_exHDD1/Imaging'; 

### get txt file directory
txt_path = f'{data_path}/{monk_num}/txt/';

### get suite2p file directory
suite2p_path = f'{data_path}/{monk_num}/tif/s{site_num}/{expName}/suite2p/plane0/';

### we need to read two txt files from txt_path
scan_txt = txt_path + expName + '_scan.txt'; 
trial_txt = txt_path + expName + '_trial.txt'; 

### get frame_onset_times
with open(scan_txt) as f:
    lines = f.readlines()
frame_ons = lines[0].split()    
frame_ons = np.array(list(map(int, frame_ons)));

### get stimulus information
stim_info = pd.read_csv(trial_txt, sep="\s", header=None)
stim_info = stim_info.iloc[:,[0,1,3,5]]; 
stim_info.columns = ['stim_on','stim_dur','sid','tid']; 

### texture index
tex_info = pd.DataFrame(columns = ['tid','coarse','direc','regular','contrast']); 
tex_info['tid'] = np.arange(21); 
tex_info['coarse'] = [32.0157804764288, 55.5924623925692, 35.7606799318989,
                      39.5993056690866, 30.4505356014797, 40.4502368043715,
                      29.1527866287278, 46.4563402606496, 39.6232525256571,
                      32.9566997636019, 52.0712202038079, 51.8405806737030,
                      43.0425180048118, 51.7188458245172, 39.4513605722589,
                      49.9780770730261, 48.5444831852984, 47.4897125823739,
                      44.3212617542613, 43.9132940392281, 33.0318004784413]; 
tex_info['direc'] = [0.996131161862129, 0.914312578684831, 0.337702335275088,
                     0.606583826619111, 0.475464273836179, 0.858683395242427,
                     0.473714358800029, 0.336251014262550, 0.303253139529931,
                     0.735790106981539, 0.323560154326412, 0.582075448418279,
                     0.411128325733624, 0.443056428535791, 0.326484467891403,
                     0.388746066849274, 0.499295775875140, 0.504878632708230, 
                     0.759175474385315, 0.340504886269147, 0.336105137109351]; 
tex_info['regular']  = [0.983573868072502, 0.849588391302584, 0.0843552739315498, 
                        0.0515873265285548, 0.108613330247710, 0.601804724767620, 
                        0.704152625430751, 0.123053253803318, 0.247371183890118, 
                        0.818156424297648, 0.272003886063917, 0.207488306329971, 
                        0.0951533950830653, 0.298922801525871, 0.0467497821904151, 
                        0.0646212591201993, 0.134692500662104, 0.181402732543146,
                        0.366735228042927, 0.126228348580076, 0.348915116002899]; 
tex_info['contrast']  = [64.0739779033683, 63.3927822792507, 45.1579286546735, 
                         57.1191270649080, 55.3604161651943, 8.36801575349820, 
                         52.9275546545688, 52.1561745388627, 88.7282349118098,
                         29.5184331062705, 74.5801514245769, 38.1957424610751,
                         19.6964929472516, 49.2410925518496, 24.2601451906754,
                         37.1080580340397, 24.5323702263551, 31.2951110754488,
                         49.8360491573582, 29.2591842669018, 32.6219975861567]; 

#%%
### load F.npy, iscell.npy
cell_resp = np.load(suite2p_path+'F.npy'); 
#cell_resp = np.load(suite2p_path+'spks.npy'); 
nCells = np.shape(cell_resp)[0]; 


#%% Construct StimResp
sids = sorted(stim_info['sid'].unique()); 
total_condnums = len(sids); 

StimResp = []
for i in range(total_condnums):
    StimResp.append(dict()); 
    StimResp[i]['sid'] = np.nan; 
    StimResp[i]['tid'] = np.nan; 
    StimResp[i]['coarse'] = np.nan;     
    StimResp[i]['direc'] = np.nan;     
    StimResp[i]['regular'] = np.nan;     
    StimResp[i]['contrast'] = np.nan;                 
    StimResp[i]['trials'] = []; 
    StimResp[i]['stim_on'] = []; 
    StimResp[i]['stim_off'] = [];    
    StimResp[i]['neurons'] = [];  

#%% fill the StimResp
condnum = 0; 

for sid in range(total_condnums): 
    StimResp[sid]['sid'] = sid; 
    StimResp[sid]['tid'] = sid//4; 
    StimResp[sid]['coarse'] = tex_info['coarse'][sid//4]; 
    StimResp[sid]['direc'] = tex_info['direc'][sid//4]; 
    StimResp[sid]['regular'] = tex_info['regular'][sid//4]; 
    StimResp[sid]['contrast'] = tex_info['contrast'][sid//4]; 

    trials = np.where((stim_info['sid']==sid))[0]; 
    for t in np.arange(len(trials)):
        trial = trials[t]; 
        stim_on = stim_info['stim_on'][trial]; 
        stim_dur = stim_info['stim_dur'][trial]; 

        if (frame_ons[0]<stim_on-1000) & (frame_ons[-1]>stim_on+stim_dur+1000): 

            StimResp[sid]['trials'].append(trial); 
            StimResp[sid]['stim_on'].append(stim_on); 
            StimResp[sid]['stim_off'].append(stim_on + stim_dur); 

            ### check frames within the time of interest
            frame_indices = np.where((frame_ons>stim_on-1000) &
                                    (frame_ons<stim_on+stim_dur+1000))[0]; 

            time_in_window = frame_ons[frame_indices]-stim_on; 
            t_start = time_in_window[0]; 
            t_end = time_in_window[-1]; 

            t_vals = np.arange(-500, stim_dur+500+1, 1); 

            resp_old = cell_resp[:,frame_indices]; 
            interpol_func = interpolate.interp1d(time_in_window, resp_old, kind='cubic'); 
            resp_new = interpol_func(t_vals); 

            StimResp[sid]['neurons'].append(resp_new); 

#%% Construct experiment
experiment = dict(); 
experiment['name'] = expName; 
experiment['tex_info'] = tex_info; 
experiment['nConds'] = total_condnums; 
experiment['StimResp'] = StimResp; 


#%% Save processed data
import json;
import gzip;

class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types 
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj,np.ndarray): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#f = gzip.GzipFile(suite2p_path+expName+'_experiment.json.gz','w');
#f.write(json.dumps(experiment, cls=NumpyEncoder).encode('utf-8'));
#f.close();


#%% Time course of activation
for i in range(len(StimResp)):
    if i == 0:
        respDynamic = np.mean(np.array(StimResp[i]['neurons']),axis=0);
    else:
        respDynamic += np.mean(np.array(StimResp[i]['neurons']),axis=0); 
respDynamic = respDynamic/len(StimResp);         

plt.figure(figsize=(5,3));
plt.plot(np.arange(-500,stim_dur+500+1),np.mean(respDynamic,axis=0))
plt.xlabel('Time from stimulus onset (ms)')
plt.ylabel('Activation level')

plt.savefig(suite2p_path+expName+'_dynamics.pdf')
#plt.show();

#%% make mResp (numCondition x numCells)
mResp = np.empty((total_condnums,nCells)); 
mResp[:] = np.nan; 
tWin1 = np.arange(0,501); 
tWin2 = np.arange(500,501+stim_dur+500); 

for i in range(len(StimResp)):
    resp = np.array(StimResp[i]['neurons']); 
    resp = np.mean(resp,axis=0); 
    mResp[i,:] = np.mean(resp[:,tWin2],axis=1)-np.mean(resp[:,tWin1],axis=1); 

#%% Cells segmented in the image
stats = np.load(suite2p_path + 'stat.npy', allow_pickle=True)
ops = np.load(suite2p_path + 'ops.npy', allow_pickle=True).item()

Ly, Lx = np.shape(ops['refImg'])

colors_all = np.array(plt.get_cmap('Paired').colors); 
im = np.zeros((Ly, Lx, 3), dtype=np.float32); 
for i in range(3):
    im[:,:,i] = ops['meanImgE']; 

plt.figure(); 
plt.imshow(im)
for n in range(0, nCells):
    ypix = stats[n]['ypix'][~stats[n]['overlap']]
    xpix = stats[n]['xpix'][~stats[n]['overlap']]
    
    c_row = n%len(colors_all); 
    plt.plot(xpix,ypix,color=colors_all[c_row,:], alpha=0.6)
    plt.text(np.mean(xpix),np.mean(ypix),f'{n}')
plt.title(f'N = {nCells}')

plt.tight_layout(); 
#plt.show();


#%% Plot individual neuron
%matplotlib qt

TexMod = np.zeros((nCells,4)) # coarse, direction. regular, contrast

for i in range(nCells):
    plt.figure(); 
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(100,200,800,800)

    ax1 = plt.subplot(3,2,1); 
    ax2 = plt.subplot(3,2,2); 
    ax3 = plt.subplot(3,2,3); 
    ax4 = plt.subplot(3,2,4); 
    ax5 = plt.subplot(3,2,5);     
    ax6 = plt.subplot(3,2,6); 

    # cell in the image
    ax1.clear(); 
    ax1.imshow(im, aspect='auto'); 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    ax1.plot(xpix,ypix,color=(1,1,0), alpha=0.6); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 

    # response dynamics
    ax2.clear()
    coarse_order = tex_info['coarse'].argsort()[::-1].values; 
    coarseAll = (mResp[coarse_order*4,i] + mResp[coarse_order*4+1,i] + 
                 mResp[coarse_order*4+2,i] + mResp[coarse_order*4+3,i])/4; 
    direc_order = tex_info['direc'].argsort()[::-1].values; 
    direcAll = (mResp[direc_order*4,i] + mResp[direc_order*4+1,i] + 
                 mResp[direc_order*4+2,i] + mResp[direc_order*4+3,i])/4; 
    regular_order = tex_info['regular'].argsort()[::-1].values; 
    regularAll = (mResp[regular_order*4,i] + mResp[regular_order*4+1,i] + 
                  mResp[regular_order*4+2,i] + mResp[regular_order*4+3,i])/4; 
    contrast_order = tex_info['contrast'].argsort()[::-1].values; 
    contrastAll = (mResp[contrast_order*4,i] + mResp[contrast_order*4+1,i] + 
                   mResp[contrast_order*4+2,i] + mResp[contrast_order*4+3,i])/4; 
    ax2.bar([1,2,4,5,7,8,10,11],
            [np.mean(coarseAll[:11]),np.mean(coarseAll[10:]),
             np.mean(direcAll[:11]),np.mean(direcAll[10:]),
             np.mean(regularAll[:11]),np.mean(regularAll[10:]),
             np.mean(contrastAll[:11]),np.mean(contrastAll[10:])])
    yMax = np.max([np.mean(coarseAll[:11]),np.mean(coarseAll[10:]),
                   np.mean(direcAll[:11]),np.mean(direcAll[10:]),
                   np.mean(regularAll[:11]),np.mean(regularAll[10:]),
                   np.mean(contrastAll[:11]),np.mean(contrastAll[10:])]);              

    _, p_coa = mannwhitneyu(coarseAll[:11],coarseAll[10:]); 
    if p_coa < 0.01:
        ax2.text(1.5, yMax, '**', color='red', fontsize=20); 
    elif p_coa < 0.05:                   
        ax2.text(1.5, yMax, '*', color='red', fontsize=20); 

    if p_coa < 0.05:
        if np.mean(coarseAll[:11])>np.mean(coarseAll[10:]):
            TexMod[i,0] = 1; 
        else:
            TexMod[i,0] = -1; 


    _, p_dir = mannwhitneyu(direcAll[:11],direcAll[10:]); 
    if p_dir < 0.01:
        ax2.text(4.5, yMax, '**', color='red', fontsize=20); 
    elif p_dir < 0.05:                   
        ax2.text(4.5, yMax, '*', color='red', fontsize=20); 

    if p_dir < 0.05:
        if np.mean(direcAll[:11])>np.mean(direcAll[10:]):
            TexMod[i,1] = 1; 
        else:
            TexMod[i,1] = -1; 


    _, p_reg = mannwhitneyu(regularAll[:11],regularAll[10:]); 
    if p_reg < 0.01:
        ax2.text(7.5, yMax, '**', color='red', fontsize=20); 
    elif p_reg < 0.05:                   
        ax2.text(7.5, yMax, '*', color='red', fontsize=20);  

    if p_reg < 0.05:
        if np.mean(regularAll[:11])>np.mean(regularAll[10:]):
            TexMod[i,2] = 1; 
        else:
            TexMod[i,2] = -1; 


    _, p_con = mannwhitneyu(contrastAll[:11],contrastAll[10:]); 
    if p_con < 0.01:
        ax2.text(10.5, yMax, '**', color='red', fontsize=20); 
    elif p_con < 0.05:                   
        ax2.text(10.5, yMax, '*', color='red', fontsize=20); 

    if p_con < 0.05:
        if np.mean(contrastAll[:11])>np.mean(contrastAll[10:]):
            TexMod[i,3] = 1; 
        else:
            TexMod[i,3] = -1; 


    ax2.set_xticks([1,2,4,5,7,8,10,11],
                   ['coa','fin','dir','n.dir',
                    'reg','i.reg','h.Con','l.Con']);              
    ax2.set_ylabel('Activation level')

    # Coarseness
    ax3.clear();         
    ax3.plot(mResp[coarse_order*4,i],'-o',label='rot0',alpha=0.5); 
    ax3.plot(mResp[coarse_order*4+1,i],'-o',label='rot45',alpha=0.5); 
    ax3.plot(mResp[coarse_order*4+2,i],'-o',label='rot90',alpha=0.5);     
    ax3.plot(mResp[coarse_order*4+3,i],'-o',label='rot135',alpha=0.5);         
    ax3.plot(coarseAll,'-o',linewidth=2,color='k')
    ax3.set_xlabel('Texture ID sorted (high to low)'); 
    ax3.set_ylabel('Activation level'); 
    ax3.set_title('Coarseness');
    ax3.legend();

    # Directionality
    ax4.clear();         
    ax4.plot(mResp[direc_order*4,i],'-o',label='rot0',alpha=0.5); 
    ax4.plot(mResp[direc_order*4+1,i],'-o',label='rot45',alpha=0.5); 
    ax4.plot(mResp[direc_order*4+2,i],'-o',label='rot90',alpha=0.5);     
    ax4.plot(mResp[direc_order*4+3,i],'-o',label='rot135',alpha=0.5);         
    ax4.plot(direcAll,'-o',linewidth=2,color='k'); 
    ax4.set_xlabel('Texture ID sorted (high to low)'); 
    ax4.set_ylabel('Activation level'); 
    ax4.set_title('Directionality');
    ax4.legend();

    # Regularity
    ax5.clear();         
    ax5.plot(mResp[regular_order*4,i],'-o',label='rot0',alpha=0.5); 
    ax5.plot(mResp[regular_order*4+1,i],'-o',label='rot45',alpha=0.5); 
    ax5.plot(mResp[regular_order*4+2,i],'-o',label='rot90',alpha=0.5);     
    ax5.plot(mResp[regular_order*4+3,i],'-o',label='rot135',alpha=0.5);        
    ax5.plot(regularAll,'-o',linewidth=2,color='k'); 
    ax5.set_xlabel('Texture ID sorted (high to low)'); 
    ax5.set_ylabel('Activation level'); 
    ax5.set_title('Regularity');
    ax5.legend();

    # Contrast
    ax6.clear();         
    ax6.plot(mResp[contrast_order*4,i],'-o',label='rot0',alpha=0.5); 
    ax6.plot(mResp[contrast_order*4+1,i],'-o',label='rot45',alpha=0.5); 
    ax6.plot(mResp[contrast_order*4+2,i],'-o',label='rot90',alpha=0.5);     
    ax6.plot(mResp[contrast_order*4+3,i],'-o',label='rot135',alpha=0.5);         
    ax6.plot(contrastAll,'-o',linewidth=2,color='k'); 
    ax6.set_xlabel('Texture ID sorted (high to low)'); 
    ax6.set_ylabel('Activation level'); 
    ax6.set_title('Contrast');
    ax6.legend();

    plt.tight_layout()

    plt.pause(1); 
    plt.close(); 

#plt.show();

#%% TexMod map
%matplotlib inline 

plt.figure(figsize=(12,8));

## coarse map
ax1 = plt.subplot(2,2,1); 
ax1.imshow(im); 
for i in range(nCells):

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    if TexMod[i,0]==1:
        ax1.plot(xpix,ypix,color=[1,0,0], alpha=0.8); 
    elif TexMod[i,0]==-1:    
        ax1.plot(xpix,ypix,color=[0,0,1], alpha=0.8); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax1.set_title('Coarseness'); 

## direction map
ax2 = plt.subplot(2,2,2); 
ax2.imshow(im); 
for i in range(nCells):

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    if TexMod[i,1]==1:
        ax2.plot(xpix,ypix,color=[1,0,0], alpha=0.8); 
    elif TexMod[i,1]==-1:    
        ax2.plot(xpix,ypix,color=[0,0,1], alpha=0.8); 
    ax2.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax2.set_title('Directionality'); 

## regularity map
ax3 = plt.subplot(2,2,3); 
ax3.imshow(im); 
for i in range(nCells):

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    if TexMod[i,2]==1:
        ax3.plot(xpix,ypix,color=[1,0,0], alpha=0.8); 
    elif TexMod[i,2]==-1:    
        ax3.plot(xpix,ypix,color=[0,0,1], alpha=0.8); 
    ax3.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax3.set_title('Regularity');

## contrast map
ax4 = plt.subplot(2,2,4); 
ax4.imshow(im); 
for i in range(nCells):

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    if TexMod[i,3]==1:
        ax4.plot(xpix,ypix,color=[1,0,0], alpha=0.8); 
    elif TexMod[i,3]==-1:    
        ax4.plot(xpix,ypix,color=[0,0,1], alpha=0.8); 
    ax4.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax4.set_title('Contrast'); 


plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_TexMap1.pdf')

#plt.show();

