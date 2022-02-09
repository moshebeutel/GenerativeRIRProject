import os
import numpy as np
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTHONHASHSEED']=str(42)
import tensorflow as tf
from Loca import *
from utils import *
# from Preprocess_Audio import directory_to_input_tensor
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from visualization import *
import random as python_random
from scipy import stats


def max_pool_smooth(data, k_size):
    data = np.concatenate(
        [np.expand_dims(data[:, :, :data.shape[2] // 2], 3),
         np.expand_dims(data[:, :, data.shape[2] // 2:], 3)], axis=3)
    data = data.transpose([0, 2, 3, 1])

    out = tf.nn.max_pool2d(data, ksize=[k_size, 1], strides=[1, 1], padding='SAME')
    with tf.Session() as sess:
        out = sess.run(out)
    # out = out.numpy()

    out = np.concatenate([out[:, :, 0, :], out[:, :, 1, :]], axis=1)
    out = out.transpose([0, 2, 1])
    return out


def extract_indices_in_region(burst_locations, region):
    """

    :param burst_locations: burst locations matrix
    :param region: a limited region we are interested in. given in the following format:
                    [[a,b],[c,d]] where a,b limit x values, and c,d limit y values
    :return: a list of indices of bursts that reside within the required region
    """
    [[a,b],[c,d]] = region
    indices = []
    for i in range(burst_locations.shape[0]):
        if a <= burst_locations[i,0] <= b and c <= burst_locations[i,1] <= d:
            indices.append(i)
    return indices


def preprocess(data, z_score=False, join_elements=False):
    N = data.shape[0]
    # Delete 2 probleamtic columns:
    data_pre = np.concatenate([data[:, :, :127], data[:, :, 129:-2]], axis=2)
    # Norimalize data:
    data_pre = data_pre / (np.pi)
    if z_score:
        data_pre = stats.zscore(data_pre.reshape(-1, 254), axis=0)
        # remove nan etries (due to constant features):
        data_pre[np.where(np.isnan(data_pre))] = 0
        idcs_zeros = np.argwhere(np.all(data_pre == 0, axis=0))
        data_pre = np.delete(data_pre, idcs_zeros, axis=1)
        # If we want to delete elements too big for viewing:
        # data_pre[np.where(data_pre > 3)] = 0
        # data_pre[np.where(data_pre < -3)] = 0
        data_pre = data_pre.reshape(N, 7, -1)

    if join_elements:
        d = data_pre.shape[2]
        data_pre = np.concatenate((data_pre[:,:,:d//2],data_pre[:,:,d//2:]), axis=1)

    return data_pre


np.random.seed(42)
python_random.seed(42)
tf.compat.v1.set_random_seed(42)

# tf.config.list_physical_devices('GPU')
# # To change gpu device:
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


print('debug checkpoint')

# # Save the input tensor:

save_dir = 'E:\Desktop\Books\Master related\Scripts and Simulations\LOCA with Simulated RIRs\\2D plus dense\White d=1cm,' \
           ' r=3.3cm\order 0 with L=[6,6,2.4], z_bursts = 0\matlab calculation'.replace('\\','/')

# np.save('F:\Master\Simulated Audio Data\Generative Models Project Data\\target\data_phase.npy', data_phase)
# # Load the input tensor:
# data = np.load('E:\Desktop\Books\Master related\Scripts, Simulations\LOCA with Simulated RIRs\\2D'.replace('\\','/') + '/data_2D.npy')
# data = data[:,:,:122]
# data = np.load('E:\Desktop\Books\Master related\Scripts, Simulations\LOCA with Simulated RIRs\\2D plus dense\White'.replace('\\','/') + '/data_2D_plus.npy')
# load_dir = 'E:\Desktop\Books\Master related\Scripts, Simulations\LOCA with Simulated RIRs\\2D L array dense'.replace('\\','/')
load_dir = save_dir
data = np.load(load_dir + '/' + 'data_phase.npy')
burst_locations = np.load(load_dir + '/burst_locations.npy')

# Visualize first bursts if you want:
show_data(data, n_bursts=110)
show_data(data, n_bursts=1000)

# fix data -
# data = np.concatenate([data[:,:,1:127],data[:,:,130:-2]], axis=2)
# # Norimalize data:
# data = data/(np.pi)
# data = preprocess(data, z_score=True, join_elements=False)
# data = preprocess(data, z_score=False, join_elements=True)
# data = preprocess(data, z_score=False, join_elements=False)
#

################################
# Training LOCA Model:

# Divide into training and validation:

N, M, d = data.shape
indices = np.random.permutation(N)
indices_train, indices_val = indices[:N * 9 // 10], indices[N * 9 // 10:]
#
data_train = data[indices_train, :, :]
data_val = data[indices_val, :, :]

# Predict variance:
pred_var = lambda radius, count : radius ** 2 * (count / (count - 1) / 2)
burst_var = pred_var(0.033, 6)
burst_var = burst_var


# Setting input parameters to the LOCA model
amount_epochs = 2000


# data_train = data
params = {}
params['bursts_var'] = burst_var

params['activation_enc'] = 'l_relu' # The activation function defined in the encoder
# params['activation_enc'] = 'tanh' # The activation function defined in the encoder
#Options: 'relu'- Relu,   'l_relu'- Leaky Relu,    'sigmoid'-sigmoid,   'tanh'- tanh, 'none'- none
params['activation_dec'] = 'tanh' # The activation function defined in the decoder
params['dropout'] = None
params['l2_reg'] = False
# params['l2_reg'] = 1000

# params['encoder_layers'] = [d, 400, 400, 400, 400, 200, 200, 200, 2]  # The amount of neurons in each layer of the encoder
# params['decoder_layers'] = [200, 200, 200, 200, 200, d]  # The amount of neurons in each layer of the decoder
params['encoder_layers'] = [d, 200, 200,200, 200, 200, 2]  # The amount of neurons in each layer of the encoder
params['decoder_layers'] = [200, 200, 200, 200, 200, d]  # The amount of neurons in each layer of the decoder


model = Loca(**params)

evaluate_every = 100
# batch_size = 512
batch_size = 1024
# batch_size = 2048
# batch_size = 4096

# lrs = [ 1e-3, 1e-4, 3e-5, 1e-6, 1e-7, 1e-8, 1e-9]
large_lrs = [1e-3, 7e-4]
medium_lrs = [3e-4, 1e-4, 3e-5]
small_lrs = [1e-5, 3e-6, 1e-6]




for lr in large_lrs[0:1]:
    for i in range(1):
        model.train(data_train, amount_epochs=15000, lr=lr, batch_size=batch_size,evaluate_every=evaluate_every,
                    data_val=data_val, verbose=True, train_only_decoder=False, mutual_train=True, tol=None,
                    initial_training=False, whlr_reclr_ratio=1)

model.best_white
model.best_rec
visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=False)
visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=True)

for lr in medium_lrs:
    for i in range(5):
        model.train(data_train, amount_epochs=1000, lr=lr, batch_size=batch_size,evaluate_every=evaluate_every,
                    data_val=data_val, verbose=True, train_only_decoder=False, mutual_train=False, tol=250,
                    initial_training=False, save_best=True, whlr_reclr_ratio=1)
visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=False)

#####
# Learning Plots:
#####
fig = plot_learning_curves(model,evaluate_every, training_only=False)
# Save learning curve:
saveFolder = 'E:\Desktop\Books\Master related\Scripts and Simulations\LOCA with Simulated RIRs\\2D plus dense\White d=3cm\Fixed LOCA\Arch8'.replace('\\','/')
fig.savefig(saveFolder + '/Learning Curve')


##################
# Data Visualization
##################



# visualize_embedding_xy_combined_2D(model, data)
fig4 = visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=False)
fig3 = visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=True)
fig = plt.gcf()
fig.savefig(saveFolder + '/embedding')
# plt.colorbar()

# Visualize Calibration
embedding, _ = model.test(data)
R,bias,scaling = calibrate_data_b(burst_locations, np.mean(embedding, axis=1),scaling=True)
calibrated_embedding = scaling*np.matmul(embedding,R)+bias

plt.figure(figsize=(20,12))
ax = plt.gca()
ax.scatter(calibrated_embedding[:, 0, 1],calibrated_embedding[:, 0, 0], c='red', label='Calibrated LOCA embedding of the locations', s=10)
ax.scatter(burst_locations[:, 1],burst_locations[:, 0], label='Original locations', s=10)
plt.title('Calibrated Embedding Space vs True Space')

ax.legend(loc='upper right')
plt.xlabel(r'Y',fontsize=35)
plt.ylabel(r'X',fontsize=35)






saveFolder = '/home/dsi/idancohen/Master - Simulations/r=3.33cm, d=1cm, s = [3,4,1], white, square [1,10]x[1,10], z=0/square = [1,7]x[1,7]/arch1/model'.replace('\\','/')

# Save model:
if 0: # save net code
    saver = tf.train.Saver(model.nnweights_enc+model.nnweights_dec)
    save_path = saver.save(model.sess, saveFolder+"/model")

# Load model:
model_loaded = Loca(**params)
if 0 :
    saver = tf.train.Saver(model_loaded.nnweights_enc + model_loaded.nnweights_dec)
    saver.restore(model_loaded.sess, saveFolder+"/model")

model = Loca(**params)
if 0 :
    saver = tf.train.Saver(model.nnweights_enc + model.nnweights_dec)
    saver.restore(model.sess, saveFolder+"/model")


#











