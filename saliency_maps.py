# Before run this file, execute the 'fine_tuning_vgg16.py' script which loads
# the trained network that will be used in this file (object called 'network')

#%% Includes


import matplotlib.pyplot as plt
import numpy as np
import os


#%%


def generate_occluded_imgs(image, wnd_size):
    nb_images = image.shape[0] * image.shape[1]
    images = np.zeros((nb_images,) + image.shape, dtype=np.uint8)
    coords = np.zeros((nb_images, 2), dtype=np.uint32)
    counter = 0
    h_size = int(wnd_size / 2)
    for u in range(h_size, image.shape[0]-h_size):
        for v in range(h_size, image.shape[1]-h_size):
            tmp = np.array(image, copy=True)
            tmp[u-h_size:u+h_size, v-h_size:v+h_size, :] = 0  # mask color
            images[counter] = tmp.astype(np.uint8)
            coords[counter] = (u, v)
            counter += 1
    images = images[:counter]
    coords = coords[:counter]
    return np.squeeze(images), coords


#%% Init input data


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


np.random.seed(42)
idx = np.random.permutation(ts_data[0].shape[0])
data = ts_data[0][idx]
ground_truth = ts_data[1][idx]


#%% Compute bins


idx = np.argsort(ground_truth)
data = data[idx]
ground_truth = ground_truth[idx]

thr = 2.5
bins = []
cur_bin = [ground_truth[0]]
cur_idx = [0]

for i in range(1,ground_truth.shape[0]):
    if ground_truth[i] - cur_bin[0] <= thr:
        cur_bin.append(ground_truth[i])
        cur_idx.append(i)
    else:
        bins.append(cur_idx)
        cur_bin = [ground_truth[i]]
        cur_idx = [i]

bins.append(cur_idx)


#%% Extract valid indices


def get_data_and_groundtruth_of_bin(bin_id):
    valid_idx = bins[bin_id]
    
    X = data[valid_idx]
    Y = ground_truth[valid_idx]
    
    np.random.seed(42)
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    Y = Y[idx]
    return X, Y

X, Y = get_data_and_groundtruth_of_bin(0)
print(X.shape, Y.shape)


#%% Compute saliency maps


for bin_id in range(len(bins)):
    print('### Computing BIN', bin_id)
    X, Y = get_data_and_groundtruth_of_bin(bin_id)

    N = np.minimum(X.shape[0], 12)
    smap = np.zeros((N,) + X.shape[1:3], dtype=np.float32)
    wnd_size = 7
    vertical_flip = False
    
    for idx in range(N):
        print('Image', idx + 1, '/', N)
        img = X[idx]
        if vertical_flip:
            img = np.flip(img, axis=1)  # vertical flip
        Fgt = Y[idx]
        images, coords = generate_occluded_imgs(img, wnd_size)
    
        xx = np.linspace(0, images.shape[0] - 1, images.shape[0] * (1 / wnd_size), dtype=np.int32)
        Fp = np.array(network.predict(images[xx])[0].tolist())
        WER = Fp - Fgt
        
        for i, (u, v) in enumerate(coords[xx]):
            smap[idx, u, v] = WER[i]

    # Show smoothed saliency map
    from scipy.ndimage.filters import gaussian_filter
    
    M = np.zeros((224,224), dtype=np.float32)
    V = np.zeros((224,224), dtype=np.float32)
    SSM = np.zeros(smap.shape, dtype=np.float32)
    
    for idx in range(N):
        SSM[idx] = gaussian_filter(smap[idx], wnd_size / 2)


    # Show results
    for idx in range(N):
        img = X[idx]
        if vertical_flip:
            img = np.flip(img, axis=1)
        plt.figure(0)
        ax = plt.subplot(3,4,idx+1)
        ax.set_title(str(idx))
        ax.imshow(img)
        ax.imshow(SSM[idx], alpha=0.60)
    
    
    for i in range(N):
        M += SSM[i]
    M = M / N
    
    for i in range(N):
        V += np.power(SSM[i] - M, 2)
    V = V / N

    '''plt.figure(1)
    plt.imshow(M)
    
    plt.figure(2)
    plt.imshow(V)
    
    plt.show()'''
    
    print('Mean map', np.mean(M), np.std(M))
    print('Variance map', np.mean(V), np.std(V))


    # Save results
    import pickle
    with open('results_' + str(bin_id) + '.pkl', 'wb') as fp:
        info = 'info, bin_id, valid_idx, SSM, MeanMap, VarianceMap'
        pickle.dump((info, bin_id, valid_idx, SSM, M, V), fp)
