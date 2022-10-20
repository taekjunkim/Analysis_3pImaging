#%% import necessary modules
import numpy as np; 
import matplotlib.pyplot as plt; 
import pandas as pd
from scipy import interpolate
from scipy import signal
from IPython import display        # for update plot in a loop 
from suite2p.io import BinaryFile
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
import time

#%%
### get experiment name
expName = '135_008_02'; #'*134_001_08*','135_008_02','132_007_01','131_005_01','134_002_03'
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
stim_info.columns = ['stim_on','stim_dur','sf','dir']; 

#%%
### load F.npy, iscell.npy
cell_resp = np.load(suite2p_path+'F.npy'); 
#cell_resp = np.load(suite2p_path+'spks.npy'); 
nCells = np.shape(cell_resp)[0]; 


#%% Construct StimResp
dirs = sorted(stim_info['dir'].unique()); 
sfs = sorted(stim_info['sf'].unique()); 
total_condnums = len(dirs)*len(sfs); 

StimResp = []
for i in range(total_condnums):
    StimResp.append(dict()); 
    StimResp[i]['dir'] = np.nan; 
    StimResp[i]['sf'] = np.nan; 
    StimResp[i]['trials'] = []; 
    StimResp[i]['stim_on'] = []; 
    StimResp[i]['stim_off'] = [];    
    StimResp[i]['neurons'] = [];  

#%% fill the StimResp
condnum = 0; 
for d, dir in enumerate(dirs):
    for s, sf in enumerate(sfs):
        StimResp[condnum]['dir'] = dir; 
        StimResp[condnum]['sf'] = sf; 

        trials = np.where((stim_info['dir']==dir) & (stim_info['sf']==sf))[0]
        for t in np.arange(len(trials)):
            trial = trials[t]; 
            stim_on = stim_info['stim_on'][trial]; 
            stim_dur = stim_info['stim_dur'][trial]; 

            if (frame_ons[0]<stim_on-1000) & (frame_ons[-1]>stim_on+stim_dur+1000): 

                StimResp[condnum]['trials'].append(trial); 
                StimResp[condnum]['stim_on'].append(stim_on); 
                StimResp[condnum]['stim_off'].append(stim_on + stim_dur); 

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

                StimResp[condnum]['neurons'].append(resp_new); 
        condnum += 1;     

#%% Construct experiment
experiment = dict(); 
experiment['name'] = expName; 
experiment['dir'] = dirs; 
experiment['sf'] = sfs; 
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
#plt.figure(figsize=(12,8)); 
#ax1 = plt.subplot(2,2,1); 
#ax2 = plt.subplot(2,2,2); 
#ax3 = plt.subplot(2,2,3); 
#ax4 = plt.subplot(2,2,4); 

%matplotlib qt

for i in range(nCells):
    plt.figure(); 
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(100,200,800,800)

    ax1 = plt.subplot(3,2,1); 
    ax2 = plt.subplot(3,2,2); 
    ax3 = plt.subplot(3,2,3); 
    ax4 = plt.subplot(3,2,4); 
    ax5 = plt.subplot(3,2,5);     


    # cell in the image
    ax1.clear(); 
    ax1.imshow(im, aspect='auto'); 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    ax1.plot(xpix,ypix,color=(1,1,0), alpha=0.6); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 

    # dir, sf tuning
    ax2.clear();         
    respNow = mResp[:,i].reshape(len(dirs),len(sfs)).T; 
    for s in range(len(sfs)):
        ax2.plot(dirs,respNow[s,:],'-o',label=f'sf = {sfs[s]}'); 
    ax2.set_xlabel('Direction (deg)'); 
    ax2.set_ylabel('Activation'); 
    ax2.set_title(f'cell#: {i}')
    ax2.legend()

    # dir tuning
    ax3.clear();         
    ax3.plot(dirs,np.mean(respNow,axis=0),'-ko'); 
    ax3.set_xlabel('Direction (deg)'); 
    ax3.set_ylabel('Activation'); 

    # sf tuning
    ax4.clear();         
    ax4.plot(sfs,np.mean(respNow,axis=1),'-ko'); 
    ax4.set_xscale('log'); 
    ax4.set_xlabel('SF (c/deg)'); 
    ax4.set_ylabel('Activation'); 

    # Response Dynamics
    ax5.clear();         
    ax5.plot(np.arange(-500,stim_dur+500+1),respDynamic[i,:])
    ax5.set_xlabel('Time from stimulus onset (ms)')
    ax5.set_ylabel('Activation level')


    plt.tight_layout()
    #display.display(plt.gcf())
    #display.clear_output(wait=True)    

    plt.pause(0.5); 
    #time.sleep(0.5); 
    plt.close(); 

#plt.show();

#%% Dir map
%matplotlib inline 

plt.figure(figsize=(12,8));

## dir map
ax1 = plt.subplot(2,2,1); 
ax3 = plt.subplot(2,2,3); 
rainbow_colors1 = plt.get_cmap('hsv',len(dirs)); 

ax1.imshow(im); 
for i in range(nCells):
    respNow = mResp[:,i].reshape(len(dirs),len(sfs)).T;     
    dir_resp = np.mean(respNow, axis=0); 

    dir_idx = np.where(dir_resp==np.max(dir_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = rainbow_colors1(dir_idx/(len(dirs)-1)); 
    ax1.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 

for i in np.arange(len(dirs)):
    color_now = rainbow_colors1(i/(len(dirs)-1)); 
    ax3.plot(dirs[i], 1, marker=(2, 0, dirs[i]), 
             color=color_now[:3], markersize=20)
    ax3.text(dirs[i], 0.98, f'{dirs[i]}')
ax3.set_axis_off()


## ori map
ax2 = plt.subplot(2,2,2); 
ax4 = plt.subplot(2,2,4); 
rainbow_colors2 = plt.get_cmap('hsv',len(dirs)/2); 

ax2.imshow(im); 
for i in range(nCells):
    respNow = mResp[:,i].reshape(len(dirs),len(sfs)).T;     
    dir_resp = np.mean(respNow, axis=0); 
    ori_resp = dir_resp[:int(len(dirs)/2)] \
               + dir_resp[int(len(dirs)/2):]

    ori_idx = np.where(ori_resp==np.max(ori_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = rainbow_colors2(ori_idx/(len(dirs)/2-1)); 
    ax2.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax2.text(np.mean(xpix),np.mean(ypix),f'{i}'); 

for i in np.arange(int(len(dirs)/2)):
    color_now = rainbow_colors2(i/(len(dirs)/2-1)); 
    ax4.plot(dirs[i], 1, marker=(2, 0, dirs[i]), 
             color=color_now[:3], markersize=20)
    ax4.text(dirs[i], 0.98, f'{dirs[i]}')
ax4.set_axis_off()

plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_DIRmap1.pdf')

#plt.show();

#%% Dir map2
f = BinaryFile(Ly=ops['Ly'],
               Lx=ops['Lx'],
               read_filename=ops['reg_file']);      


def get_dirmap(which_ori):
    dirmap = []; 

    for i in np.arange(len(which_ori)):
        stim_on = stim_info['stim_on'][which_ori[i]]; 

        frame_indices1 = np.where((frame_ons>stim_on-500) &
                                  (frame_ons<stim_on))[0]; 
        frame_indices1 = frame_indices1.tolist(); 

        frame_indices2 = np.where((frame_ons>stim_on) &
                                  (frame_ons<stim_on+stim_dur+500))[0]; 
        frame_indices2 = frame_indices2.tolist(); 

        if len(frame_indices1)>1:
            if len(dirmap) == 0:
                dirmap = np.mean(f[frame_indices2],axis=0)-np.mean(f[frame_indices1],axis=0); 
            else:
                dirmap += np.mean(f[frame_indices2],axis=0)-np.mean(f[frame_indices1],axis=0);
    dirmap = dirmap/len(which_ori); 

    ### z-scored
    dirmap = (dirmap - np.mean(dirmap)) / np.std(dirmap); 

    return dirmap; 
    
    """
    ### get meanImgE
    spatscale_pix = 12; 
    aspect = 1; 

    I = dirmap.copy(); 
    diameter = 4*np.ceil(np.array([spatscale_pix * aspect, spatscale_pix])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I    

    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
         ops['xrange'][0]:ops['xrange'][1]] = mimg0

    return mimg; 
    """

plt.figure(figsize=(12,12)); 

for i in range(int(len(dirs)/2)):
    dir1 = dirs[i]; 
    dir2 = dirs[int(i + len(dirs)/2)]; 

    ori_now = np.where((stim_info['dir']==dir1) | 
                       (stim_info['dir']==dir2))[0]; 

    map_now = get_dirmap(ori_now); 

    plt.subplot(4,2,i+1); 
    plt.imshow(map_now, vmin=-10, vmax = 10, cmap='gray'); 
    plt.title(f'Orientation = {dir1}')

plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_DIRmap2.pdf')


#%% SF map
plt.figure(figsize=(12,8));

## SF map
ax1 = plt.subplot(2,2,1); 
ax3 = plt.subplot(2,2,3); 
cool_colors1 = plt.get_cmap('cool',len(sfs)); 

ax2 = plt.subplot(2,2,2); 
ax4 = plt.subplot(2,2,4); 
rainbow_colors1 = plt.get_cmap('hsv',len(sfs)); 

ax1.imshow(im);  # Cells segmented in the image
for i in range(nCells):
    respNow = mResp[:,i].reshape(len(dirs),len(sfs)).T;     
    sf_resp = np.mean(respNow, axis=1); 

    sf_idx = np.where(sf_resp==np.max(sf_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = cool_colors1(sf_idx/(len(sfs)-1)); 
    ax1.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax1.set_title('SF map')

for i in np.arange(len(sfs)):
    color_now = cool_colors1(i/(len(sfs)-1)); 
    ax3.plot(i, 1, 'o', 
             color=color_now[:3], markersize=20)
    ax3.text(i, 0.98, f'{sfs[i]}')
ax3.set_axis_off()

ax2.imshow(im);  # Cells segmented in the image
for i in range(nCells):
    respNow = mResp[:,i].reshape(len(dirs),len(sfs)).T;     
    sf_resp = np.mean(respNow, axis=1); 

    sf_idx = np.where(sf_resp==np.max(sf_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = rainbow_colors1(sf_idx/(len(sfs)-1)); 
    ax2.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax2.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax2.set_title('SF map')

for i in np.arange(len(sfs)):
    color_now = rainbow_colors1(i/(len(sfs)-1)); 
    ax4.plot(i, 1, 'o', 
             color=color_now[:3], markersize=20)
    ax4.text(i, 0.98, f'{sfs[i]}')
ax4.set_axis_off()

plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_SFmap.pdf')


# %%
