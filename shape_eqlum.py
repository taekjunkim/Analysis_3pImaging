#%% import necessary modules
import numpy as np; 
import matplotlib.pyplot as plt; 
import pandas as pd
from scipy.interpolate import griddata
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
expName = '135_003_04'; #'*134_019_07 (short latency)*','135_003_04','134_010_08','134_014_03'
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

#%%
### get stimulus information
stim_info = pd.read_csv(trial_txt, sep="\s", header=None)
stim_info = stim_info.iloc[:,[0,1,3,5,7,9,11,13,15]]; 
stim_info.columns = ['stim_on','stim_dur','shape_id','rot_id',
                     'R','G','B','color_id','lum']; 

#%%
### load F.npy, iscell.npy
cell_resp = np.load(suite2p_path+'F.npy'); 
#cell_resp = np.load(suite2p_path+'spks.npy'); 
nCells = np.shape(cell_resp)[0]; 


#%% Construct StimResp
shapes = sorted(stim_info['shape_id'].unique()); 
rots = sorted(stim_info['rot_id'].unique()); 
color_ids = sorted(stim_info['color_id'].unique()); 
lums = sorted(stim_info['lum'].unique()); 

total_shapes = len(shapes)*len(rots); 
total_colors = len(color_ids)*len(lums); 
total_condnums = total_shapes * total_colors; 

StimResp = []
for i in range(total_condnums):
    StimResp.append(dict()); 
    StimResp[i]['shape_id'] = np.nan; 
    StimResp[i]['rot_id'] = np.nan; 
    StimResp[i]['color_id'] = np.nan; 
    StimResp[i]['lum'] = np.nan; 
    StimResp[i]['trials'] = []; 
    StimResp[i]['stim_on'] = []; 
    StimResp[i]['stim_off'] = [];    
    StimResp[i]['neurons'] = [];  

#%% color information
color_table = pd.DataFrame(); 
color_table['c_num'] = np.arange(10); 
color_table['CIE_x'] = [0.17, 0.225, 0.25, 0.28, 0.305,
                        0.33, 0.335, 0.39, 0.415, 0.5]; 
color_table['CIE_y'] = [0.09, 0.27, 0.21, 0.45, 0.39,
                        0.33, 0.205, 0.385, 0.325, 0.32]; 


#%% fill the StimResp
stim_table = pd.DataFrame(); 
stim_table['shape_id'] = np.nan; 
stim_table['rot_id'] = np.nan; 
stim_table['shapes_num'] = np.nan; 
stim_table['color_id'] = np.nan; 
stim_table['lum'] = np.nan; 
stim_table['colors_num'] = np.nan; 

condnum = 0; 
for s in range(total_shapes):
    for c in range(total_colors):
        StimResp[condnum]['shape_id'] = shapes[s//len(rots)]; 
        StimResp[condnum]['rot_id'] = s%len(rots); 

        StimResp[condnum]['color_id'] = c%len(color_ids); 
        StimResp[condnum]['lum'] = lums[(c//len(color_ids))]; 

        stim_table.loc[condnum,'shape_id'] = shapes[s//len(rots)]; 
        stim_table.loc[condnum,'rot_id'] = s%len(rots); 
        stim_table.loc[condnum,'shapes_num'] = s; 
        stim_table.loc[condnum,'color_id'] = c%len(color_ids); 
        stim_table.loc[condnum,'lum'] = lums[(c//len(color_ids))]; 
        stim_table.loc[condnum,'colors_num'] = c; 

        trials = np.where((stim_info['shape_id']==shapes[s//len(rots)]) & 
                          (stim_info['rot_id']==s%len(rots)) &
                          (stim_info['color_id']==c%len(color_ids)) &
                          (stim_info['lum']==lums[(c//len(color_ids))]))[0]; 

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

#%% Time course of activation
for i in range(len(StimResp)):
    if i == 0:
        respDynamic = np.mean(np.array(StimResp[i]['neurons']),axis=0);
    else:
        respDynamic += np.mean(np.array(StimResp[i]['neurons']),axis=0); 
respDynamic = respDynamic/len(StimResp);         

plt.figure(); 
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
    mngr.window.setGeometry(100,200,900,900)

    ax1 = plt.subplot(3,3,1); 
    ax2 = plt.subplot(3,3,2); 
    ax5 = plt.subplot(3,3,5); 
    ax8 = plt.subplot(3,3,8); 

    ax4 = plt.subplot(3,3,4); 

    ax3 = plt.subplot(3,3,3); 
    ax6 = plt.subplot(3,3,6);     
    ax9 = plt.subplot(3,3,9); 


    # cell in the image
    ax1.clear(); 
    ax1.imshow(im, aspect='auto'); 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    ax1.plot(xpix,ypix,color=(1,1,0), alpha=0.6); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 

    # make color response matrix
    color_resp = []; 
    for c in range(total_colors):
        cond_now = np.where(stim_table['colors_num']==c)[0]
        resp_now = np.mean(mResp[cond_now,i]); 
        color_resp.append(resp_now); 

    color_mtx = np.array(color_resp).reshape(-1,len(color_ids)); 

    # normalize color_mtx
    size_gain = 30; 
    color_mtx = color_mtx - np.min(color_mtx); 
    color_mtx = size_gain * color_mtx / np.max(color_mtx); 
    color_mtx = color_mtx.astype('int')    

    # color in dark
    ax2.clear();         
    """
    points = np.empty((len(color_ids),2)); 
    points[:,0] = color_table['CIE_x'].values; 
    points[:,1] = color_table['CIE_y'].values; 
    grid_x, grid_y = np.meshgrid(np.arange(0.17,0.5,0.005), np.arange(0.09,0.45,0.005)); 

    z_new = griddata(points, color_mtx[0,:], 
                     (grid_x, grid_y), method='cubic'); 
    ax2.imshow(z_new.T, cmap='binary', origin='lower'); 
    """
    for c in range(len(color_ids)):
        row_color = np.where((stim_info['color_id']==c) & (stim_info['lum']==lums[0]))[0][0]; 
        color_now = np.array([stim_info['R'][row_color], 
                              stim_info['G'][row_color],
                              stim_info['B'][row_color]]); 
        color_now = color_now/255;      
        color_now = color_now**(1/2.2);                          
        ax2.plot(color_table['CIE_x'][c], color_table['CIE_y'][c], 'o',
                    ms=color_mtx[0,c], color=color_now); 

    ax2.set_title('Lum: Darker than BG'); 
    ax2.set_xlabel('CIE_x'); 
    ax2.set_ylabel('CIE_y'); 

    # color in medium
    ax5.clear();         
    """
    z_new = griddata(points, color_mtx[1,:], 
                     (grid_x, grid_y), method='cubic'); 
    ax3.imshow(z_new.T, cmap='binary', origin='lower'); 
    """
    for c in range(len(color_ids)):
        row_color = np.where((stim_info['color_id']==c) & (stim_info['lum']==lums[1]))[0][0]; 
        color_now = np.array([stim_info['R'][row_color], 
                              stim_info['G'][row_color],
                              stim_info['B'][row_color]]); 
        color_now = color_now/255;          
        color_now = color_now**(1/2.2);                      
        ax5.plot(color_table['CIE_x'][c], color_table['CIE_y'][c], 'o',
                    ms=color_mtx[1,c], color=color_now); 

    ax5.set_title('Lum: Same as BG');     
    ax5.set_xlabel('CIE_x'); 
    ax5.set_ylabel('CIE_y'); 

    # color in bright
    ax8.clear();         
    """
    z_new = griddata(points, color_mtx[2,:], 
                     (grid_x, grid_y), method='cubic'); 
    ax4.imshow(z_new.T, cmap='binary', origin='lower'); 
    """    
    for c in range(len(color_ids)):
        row_color = np.where((stim_info['color_id']==c) & (stim_info['lum']==lums[2]))[0][0]; 
        color_now = np.array([stim_info['R'][row_color], 
                              stim_info['G'][row_color],
                              stim_info['B'][row_color]]); 
        color_now = color_now/255;          
        color_now = color_now**(1/2.2);                      
        ax8.plot(color_table['CIE_x'][c], color_table['CIE_y'][c], 'o',
                    ms=color_mtx[2,c], color=color_now); 

    ax8.set_title('Lum: Brigher than BG');     
    ax8.set_xlabel('CIE_x'); 
    ax8.set_ylabel('CIE_y'); 

    # Response dynamics
    ax4.clear();         
    ax4.plot(np.arange(-500,stim_dur+500+1),respDynamic[i,:])
    ax4.set_xlabel('Time from stimulus onset (ms)')
    ax4.set_ylabel('Activation level')


    # tuning curves by CIE_x
    ax3.clear(); 
    ax3.plot(np.arange(len(color_ids)), color_mtx[0,:], 'o-',
             color=[0,0,0], label='Darker than BG'); 
    ax3.plot(np.arange(len(color_ids)), color_mtx[1,:], 'o-',
             color=[0.3,0.3,0.3], label='Same as BG'); 
    ax3.plot(np.arange(len(color_ids)), color_mtx[2,:], 'o-',
             color=[0.6,0.6,0.6], label='Brighter than BG'); 
    ax3.plot(np.arange(len(color_ids)), np.mean(color_mtx, axis=0), 'o-',
             color=[1,0,0], label='Average'); 
    ax3.set_title('Color tuning');     
    ax3.set_xlabel('Low CIE_x to High CIE_x'); 
    ax3.set_ylabel('Activation'); 

    # tuning curves by CIE_y
    order_now = color_table.sort_values(by='CIE_y')['c_num'].values; 
    ax6.clear(); 
    ax6.plot(np.arange(len(color_ids)), color_mtx[0,order_now], 'o-',
             color=[0,0,0], label='Darker than BG'); 
    ax6.plot(np.arange(len(color_ids)), color_mtx[1,order_now], 'o-',
             color=[0.3,0.3,0.3], label='Same as BG'); 
    ax6.plot(np.arange(len(color_ids)), color_mtx[2,order_now], 'o-',
             color=[0.6,0.6,0.6], label='Brighter than BG'); 
    ax6.plot(np.arange(len(color_ids)), np.mean(color_mtx[:,order_now], axis=0), 'o-',
             color=[1,0,0], label='Average'); 
    ax6.set_title('Color tuning');     
    ax6.set_xlabel('Low CIE_y to High CIE_y'); 
    ax6.set_ylabel('Activation');     

    # Luminance
    ax9.clear(); 
    ax9.bar(0, np.mean(color_mtx, axis=1)[0],
            color=[0,0,0], label='Darker than BG'); 
    ax9.bar(1, np.mean(color_mtx, axis=1)[1],
            color=[0.3,0.3,0.3], label='Same as BG'); 
    ax9.bar(2, np.mean(color_mtx, axis=1)[2],
            color=[0.6,0.6,0.6], label='Brighter than BG'); 

    ax9.set_xticks([0,1,2])
    ax9.set_xticklabels(['Darker', 'Same', 'Brighter']); 
    ax9.set_ylabel('Activation'); 

    plt.tight_layout()
    #display.display(plt.gcf())
    #display.clear_output(wait=True)    

    plt.pause(1); 
    #time.sleep(0.5); 
    plt.close(); 

#plt.show();
#%% Color map & Lum map
%matplotlib inline 

plt.figure(figsize=(6,8));

## color map
ax1 = plt.subplot(2,1,1); 
ax2 = plt.subplot(2,1,2); 

ax1.imshow(im); 
for i in range(nCells):

    # make color response matrix
    color_resp = []; 
    for c in range(total_colors):
        cond_now = np.where(stim_table['colors_num']==c)[0]
        resp_now = np.mean(mResp[cond_now,i]); 
        color_resp.append(resp_now); 

    color_mtx = np.array(color_resp).reshape(-1,len(color_ids)); 

    # normalize color_mtx
    size_gain = 30; 
    color_mtx = color_mtx - np.min(color_mtx); 
    color_mtx = size_gain * color_mtx / np.max(color_mtx); 
    color_mtx = color_mtx.astype('int')    

    color_resp = np.mean(color_mtx, axis=0); 
    color_idx = np.where(color_resp==np.max(color_resp))[0]; 

    ypix = stats[i]['ypix'][~stats[i]['overlap']]; 
    xpix = stats[i]['xpix'][~stats[i]['overlap']]; 

    row_color = np.where((stim_info['color_id']==color_idx[0]) & 
                         (stim_info['lum']==lums[2]))[0][0]; 
    color_now = np.array([stim_info['R'][row_color], 
                            stim_info['G'][row_color],
                            stim_info['B'][row_color]]); 
    color_now = color_now/255;          
    color_now = color_now**(1/2.2);                      

    ax1.plot(xpix,ypix,color=color_now[:3], alpha=0.8); 
    ax1.text(np.mean(xpix),np.mean(ypix),f'{i}'); 
ax1.set_title('Color map');         

for c in range(len(color_ids)):
    row_color = np.where((stim_info['color_id']==c) & (stim_info['lum']==lums[2]))[0][0]; 
    color_now = np.array([stim_info['R'][row_color], 
                        stim_info['G'][row_color],
                        stim_info['B'][row_color]]); 
    color_now = color_now/255;          
    color_now = color_now**(1/2.2);                      
    ax2.plot(color_table['CIE_x'][c], color_table['CIE_y'][c], 'o',
                ms=20, color=color_now); 

ax2.set_title('Colors tested');     
ax2.set_xlabel('CIE_x'); 
ax2.set_ylabel('CIE_y'); 

plt.tight_layout()
plt.savefig(suite2p_path+expName+'_Color_map.pdf')
# %%
