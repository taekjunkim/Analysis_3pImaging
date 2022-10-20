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
expName = '135_008_04'; # '132_006_05', '134_001_02','134_015_02','134_018_01'
                        # '135_002_01',
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
stim_info = stim_info.iloc[:,[0,1,3,5,7]]; 
stim_info.columns = ['stim_on','stim_dur','cx','cy','sf']; 

#%%
### load F.npy, iscell.npy
cell_resp = np.load(suite2p_path+'F.npy'); 
#cell_resp = np.load(suite2p_path+'spks.npy'); 
nCells = np.shape(cell_resp)[0]; 


#%% Construct StimResp
cxs = sorted(stim_info['cx'].unique()); 
cys = sorted(stim_info['cy'].unique()); 
sfs = sorted(stim_info['sf'].unique()); 
total_condnums = len(cxs)*len(cys)*len(sfs); 

StimResp = []
for i in range(total_condnums):
    StimResp.append(dict()); 
    StimResp[i]['cx'] = np.nan; 
    StimResp[i]['cy'] = np.nan;     
    StimResp[i]['sf'] = np.nan; 
    StimResp[i]['trials'] = []; 
    StimResp[i]['stim_on'] = []; 
    StimResp[i]['stim_off'] = [];    
    StimResp[i]['neurons'] = [];  

#%% fill the StimResp
condnum = 0; 
for x, cx in enumerate(cxs):
    for y, cy in enumerate(cys):
        for s, sf in enumerate(sfs):
            StimResp[condnum]['cx'] = cx; 
            StimResp[condnum]['cy'] = cy; 
            StimResp[condnum]['sf'] = sf; 

            trials = np.where((stim_info['cx']==cx) &
                              (stim_info['cy']==cy) & 
                              (stim_info['sf']==sf))[0]

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
experiment['cx'] = cxs; 
experiment['cy'] = cys; 
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

    # Response Dynamics
    ax2.clear();         
    ax2.plot(np.arange(-500,stim_dur+500+1),respDynamic[i,:])
    ax2.set_xlabel('Time from stimulus onset (ms)')
    ax2.set_ylabel('Activation level')

    # cx, cy, sf tuning: cx x cy x sf
    respNow = mResp[:,i].reshape(len(cxs),len(cys),len(sfs)); 

    # cx, cy map
    ax3.clear();         
    sMap = np.mean(respNow,axis=2).T; 
    ax3.imshow(sMap, origin='lower', aspect='auto'); 
    ax3.set_xlabel('Horizontal (deg)');
    ax3.set_xticks(np.arange(np.shape(sMap)[1]),cxs);
    ax3.set_ylabel('Vertical (deg)');
    ax3.set_yticks(np.arange(np.shape(sMap)[0]),cys);
    ax3.set_title('spatial map'); 

    # cx tuning
    cx_resp = np.mean(respNow, axis=1); 
    for s in range(len(sfs)):
        ax5.plot(cxs, cx_resp[:,s], '-o', label=f'sf = {sfs[s]}'); 
    ax5.set_xlabel('Horizontal (deg)');         
    ax5.set_ylabel('Activation'); 
    ax5.legend(); 

    # cy tuning
    cy_resp = np.mean(respNow, axis=0); 
    for s in range(len(sfs)):
        ax4.plot(cy_resp[:,s], cys, '-o', label=f'sf = {sfs[s]}'); 
    ax4.set_xlabel('Activation');         
    ax4.set_ylabel('Vertical (deg)'); 
    ax4.legend(); 

    plt.tight_layout()
    #display.display(plt.gcf())
    #display.clear_output(wait=True)    

    plt.pause(0.5); 
    #time.sleep(0.5); 
    plt.close(); 

#plt.show();

#%% Horizontal map
%matplotlib inline 

plt.figure(figsize=(12,8));

ax1 = plt.subplot(2,2,1); 
ax3 = plt.subplot(2,2,3); 
cool_colors1 = plt.get_cmap('cool',len(cxs)); 

ax1.imshow(im); 
for i in range(nCells):

    respNow = mResp[:,i].reshape(len(cxs),len(cys),len(sfs));   # cx x cy x sf  
    cx_resp = np.mean(respNow, axis=1);  # cx x sf
    cx_resp = np.mean(cx_resp,axis=1);   # cx

    cx_idx = np.where(cx_resp==np.max(cx_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = cool_colors1(cx_idx/(len(cxs)-1)); 
    ax1.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax1.set_title('Horizontal (deg)')

for i in np.arange(len(cxs)):
    color_now = cool_colors1(i/(len(cxs)-1)); 
    ax3.plot(i, 1, marker='o', 
             color=color_now[:3], markersize=20)
    ax3.text(i, 0.98, f'{cxs[i]}')
ax3.set_axis_off()


## Vertical map
ax2 = plt.subplot(2,2,2); 
ax4 = plt.subplot(2,2,4); 
cool_colors1 = plt.get_cmap('cool',len(cys)); 

ax2.imshow(im); 
for i in range(nCells):

    respNow = mResp[:,i].reshape(len(cxs),len(cys),len(sfs));   # cx x cy x sf  
    cy_resp = np.mean(respNow, axis=0);  # cy x sf
    cy_resp = np.mean(cy_resp,axis=1);   # cy

    cy_idx = np.where(cy_resp==np.max(cy_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    color_now = cool_colors1(cy_idx/(len(cys)-1)); 
    ax2.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax2.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax2.set_title('Vertical (deg)')    

for i in np.arange(len(cys)):
    color_now = cool_colors1(i/(len(cys)-1)); 
    ax4.plot(i, 1, marker='o', 
             color=color_now[:3], markersize=20)
    ax4.text(i, 0.98, f'{cys[i]}')
ax4.set_axis_off()

plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_HVmap1.pdf')

#plt.show();

#%% HV map2
f = BinaryFile(Ly=ops['Ly'],
               Lx=ops['Lx'],
               read_filename=ops['reg_file']);      


def get_HVmap(which_pos):
    HVmap = []; 

    for i in np.arange(len(which_pos)):
        stim_on = stim_info['stim_on'][which_pos[i]]; 

        frame_indices1 = np.where((frame_ons>stim_on-500) &
                                  (frame_ons<stim_on))[0]; 
        frame_indices1 = frame_indices1.tolist(); 

        frame_indices2 = np.where((frame_ons>stim_on) &
                                  (frame_ons<stim_on+stim_dur+500))[0]; 
        frame_indices2 = frame_indices2.tolist(); 

        if len(frame_indices1)>1:
            if len(HVmap) == 0:
                HVmap = np.mean(f[frame_indices2],axis=0)-np.mean(f[frame_indices1],axis=0); 
            else:
                HVmap += np.mean(f[frame_indices2],axis=0)-np.mean(f[frame_indices1],axis=0);
    HVmap = HVmap/len(which_pos); 

    ### z-scored
    HVmap = (HVmap - np.mean(HVmap)) / np.std(HVmap); 

    return HVmap; 
    

plt.figure(figsize=(12,8)); 

i = 0; 
for yi in range(int(len(cys))):
    for xi in range(int(len(cxs))):

        cy = cys[-(yi+1)]; 
        cx = cxs[xi]; 


        pos_now = np.where((stim_info['cx']==cx) & 
                           (stim_info['cy']==cy))[0]; 
        map_now = get_HVmap(pos_now); 

        plt.subplot(len(cys),len(cxs),i+1); 
        plt.imshow(map_now, vmin=-10, vmax = 10, cmap='gray'); 
        plt.title(f'Pos: x = {cx}, y = {cy}')
        i += 1; 

plt.tight_layout(); 
plt.savefig(suite2p_path+expName+'_HVmap2.pdf')


# %%
