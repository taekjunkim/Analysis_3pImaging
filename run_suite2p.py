### run this code in the 'suite2p' environment
#%% import modules
import os
import matplotlib.pyplot as plt
import numpy as np
import suite2p

#%% Figure style setting
import matplotlib as mpl
mpl.rcParams.update({
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'figure.subplot.wspace': .01,
    'figure.subplot.hspace': .01,
    'figure.figsize': (18, 13),
    'ytick.major.left': True,
})
jet = mpl.cm.get_cmap('jet')
jet.set_bad(color='k')

#%% Set pipeline parameters
ops = suite2p.default_ops()
ops['fs'] = 6.8; 
ops['sparse_mode'] = 0; 
ops['anatomical_only'] = 3; 
ops['diameter'] = 12;     # try 0 first, if it doesn't look nice, try 12, 24, 48
#ops['threshold_scaling'] = 2.0; 
#ops['spatial_scale'] = 2;    # only work in the sparse mode
#ops['tau'] = 1.0;      # 0.7 for GCaMP6f, 1.0 for GCaMP6m, 1.25-1.5 for GCaMP6s
print(ops)

#%%
tiff_path = '/Volumes/TJ_exHDD1/Imaging/134/tif/s019/134_019_07'
db = {'data_path': [tiff_path],}
output_ops = suite2p.run_s2p(ops=ops, db=db)

#%% 
# ops dictionary contains all the keys that went into the analysis, 
# plus new keys that contain additional metrics/outputs calculated 
# during the pipeline run
print(set(output_ops.keys()).difference(ops.keys()))


#%% Registration

plt.subplot(2, 2, 1)
plt.imshow(output_ops['refImg'], cmap='gray', )
plt.title("Reference Image for Registration");

# maximum of recording over time
plt.subplot(2, 2, 2)
plt.imshow(output_ops['max_proj'], cmap='gray')
plt.title("Registered Image, Max Projection");

plt.subplot(2, 2, 3)
plt.imshow(output_ops['meanImg'], cmap='gray')
plt.title("Mean registered image")

plt.subplot(2, 2, 4)
plt.imshow(output_ops['meanImgE'], cmap='gray')
plt.title("High-pass filtered Mean registered image");

plt.tight_layout()

#%% OFFSET
# The rigid offsets of the frame from the reference are saved 
# in output_ops['yoff'] and output_ops['xoff']. 
# The nonrigid offsets are saved in output_ops['yoff1'] and 
# output_ops['xoff1'], and each column is the offsets for
#  a block (128 x 128 pixels by default).

plt.figure(figsize=(18,8))

plt.subplot(4,1,1)
plt.plot(output_ops['yoff'][:1000])
plt.ylabel('rigid y-offsets')

plt.subplot(4,1,2)
plt.plot(output_ops['xoff'][:1000])
plt.ylabel('rigid x-offsets')

plt.subplot(4,1,3)
plt.plot(output_ops['yoff1'][:1000])
plt.ylabel('nonrigid y-offsets')

plt.subplot(4,1,4)
plt.plot(output_ops['xoff1'][:1000])
plt.ylabel('nonrigid x-offsets')
plt.xlabel('frames')

plt.tight_layout()

#%% DETECTION
# ROIs are found by searching for sparse signals 
# that are correlated spatially in the FOV. 
# The ROIs are saved in stat.npy as a list of dictionaries 
# which contain the pixels of the ROI and their weights 
# (stat['ypix'], stat['xpix'], and stat['lam']). 
# It also contains other spatial properties of the ROIs 
# such as their aspect ratio and compactness, 
# and properties of the signal such as the skewness of the fluorescence signal.

ops = np.load(output_ops['save_path'] + '/ops.npy', allow_pickle=True).item()
stats_file = output_ops['save_path'] + '/stat.npy'; 
iscell = np.load(output_ops['save_path'] + '/iscell.npy', allow_pickle=True)[:, 0].astype(int)
stats = np.load(stats_file, allow_pickle=True)
print(stats[0].keys())

Ly, Lx = np.shape(output_ops['refImg'])


n_cells = len(stats)

h = np.random.rand(n_cells)
hsvs = np.zeros((2, Ly, Lx, 3), dtype=np.float32)

for i, stat in enumerate(stats):
    ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
    hsvs[iscell[i], ypix, xpix, 0] = h[i]
    hsvs[iscell[i], ypix, xpix, 1] = 1
    hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()

from colorsys import hsv_to_rgb
rgbs = np.array([hsv_to_rgb(*hsv) for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

plt.figure(figsize=(20,15))
plt.gcf().suptitle(f'{tiff_path[-10:]}', fontsize=16)

plt.subplot(3, 2, 1)
plt.imshow(output_ops['max_proj'], cmap='gray')
plt.title("Registered Image, Max Projection")

plt.subplot(3, 2, 2)
plt.imshow(ops['meanImgE'], cmap='gray')
plt.title("High-pass filtered Mean registered image");

plt.subplot(3, 2, 3)
plt.imshow(rgbs[1])
plt.title("All Cell ROIs")

plt.subplot(3, 2, 4)
colors_all = np.array(plt.get_cmap('Paired').colors); 
im = np.zeros((Ly, Lx, 3), dtype=np.float32); 
for i in range(3):
    im[:,:,i] = ops['meanImgE']; 
plt.imshow(im)
for n in range(0,n_cells):
    ypix = stats[n]['ypix'][~stats[n]['overlap']]
    xpix = stats[n]['xpix'][~stats[n]['overlap']]
    
    c_row = n%len(colors_all); 
    plt.plot(xpix,ypix,color=colors_all[c_row,:], alpha=0.6)
    plt.text(np.mean(xpix),np.mean(ypix),f'{n}')

plt.subplot(3, 2, 5)
plt.imshow(rgbs[0])
plt.title("All non-Cell ROIs");


plt.tight_layout()
plt.savefig(output_ops['save_path']+"/ROIs.pdf", format="pdf", bbox_inches="tight")


#%% TRACES

f_cells = np.load(output_ops['save_path']+'/F.npy')
f_neuropils = np.load(output_ops['save_path']+'/Fneu.npy')
spks = np.load(output_ops['save_path']+'/spks.npy')
f_cells.shape, f_neuropils.shape, spks.shape

plt.figure(figsize=[20,20])
plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92);
rois = np.arange(len(f_cells))[::(f_cells.shape[0]//5)]
for i, roi in enumerate(rois):
    plt.subplot(len(rois), 1, i+1, )
    f = f_cells[roi]
    f_neu = f_neuropils[roi]
    sp = spks[roi]
    # Adjust spks range to match range of fluroescence traces
    fmax = np.maximum(f.max(), f_neu.max())
    fmin = np.minimum(f.min(), f_neu.min())
    frange = fmax - fmin 
    sp /= sp.max()
    sp *= frange
    plt.plot(f, label="Cell Fluorescence")
    plt.plot(f_neu, label="Neuropil Fluorescence")
    plt.plot(sp + fmin, label="Deconvolved")
    plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
    plt.ylabel(f"ROI {roi}", rotation=0)
    plt.xlabel("frame")
    if i == 0:
        plt.legend(bbox_to_anchor=(0.93, 2))

plt.tight_layout()

#%%
#@title Run cell to look at registered frames
"""
from suite2p.io import BinaryFile
import cv2

f = BinaryFile(Ly=output_ops['Ly'],
               Lx=output_ops['Lx'],
               read_filename=output_ops['reg_file']); 

height, width = np.shape(f[0][0])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(tiff_path+'/suite2p/plane0/frames.avi', fourcc, 120, (width, height)) 



for i in range(output_ops['nframes']):
    frame_now = f[i][0]; 
    frame_now = (frame_now - np.min(frame_now)); 
    frame_now = frame_now/(np.max(frame_now)*0.25); 
    frame_now = np.uint8(frame_now*255); 
    frame_now = cv2.cvtColor(frame_now, cv2.COLOR_GRAY2BGR)

    out.write(frame_now)

out.release()

"""
# %%
