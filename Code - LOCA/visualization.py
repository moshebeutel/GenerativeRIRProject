import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def show_data(data, n_bursts=None, burst_range=None):
    """
    :param data: Data tensor of shape N x M x d.
    :param n_bursts: Number of bursts to visualize. If None, will show all bursts.
    :param burst_range: np.array of length 2.Range of bursts to be shown
    :return: figure containing visualization of the data.
    """
    fig = plt.figure()
    ax = fig.gca()
    if n_bursts is None and burst_range is None:
        ax.imshow(data[:, :, :].reshape(-1,data.shape[2]))
    elif n_bursts is not None:
        ax.imshow(data[:n_bursts, :, :].reshape(-1,data.shape[2]))
    elif burst_range is not None:
        ax.imshow(data[burst_range[0]:burst_range[1], :, :].reshape(-1, data.shape[2]))
    return fig


def show_data_phase_abs(data1, data2, n_bursts=None, burst_range=None):
    """
    :param data: Data tensor of shape N x M x d.
    :param n_bursts: Number of bursts to visualize. If None, will show all bursts.
    :param burst_range: np.array of length 2.Range of bursts to be shown
    :return: figure containing visualization of the data.
    """
    fig, axes = plt.subplots(1,2)
    # ax = fig.gca()
    if n_bursts is None and burst_range is None:
        axes[0].imshow(data1[:, :, :].reshape(-1,data1.shape[2]))
        axes[0].set_title('RTF Phase Part')
        axes[1].imshow(data2[:, :, :].reshape(-1,data2.shape[2]))
        axes[1].set_title('RTF Abs Part')
    elif n_bursts is not None:
        axes[0].imshow(data1[:n_bursts, :, :].reshape(-1,data1.shape[2]))
        axes[0].set_title('RTF Phase Part')
        axes[1].imshow(data2[:n_bursts, :, :].reshape(-1,data2.shape[2]))
        axes[1].set_title('RTF Abs Part')
    elif burst_range is not None:
        axes[0].imshow(data1[burst_range[0]:burst_range[1], :, :].reshape(-1, data1.shape[2]))
        axes[1].imshow(data2[burst_range[0]:burst_range[1], :, :].reshape(-1, data2.shape[2]))
    return fig


def show_data_real_imag(data1, data2, n_bursts=None, burst_range=None):
    """
    :param data: Data tensor of shape N x M x d.
    :param n_bursts: Number of bursts to visualize. If None, will show all bursts.
    :param burst_range: np.array of length 2.Range of bursts to be shown
    :return: figure containing visualization of the data.
    """
    fig, axes = plt.subplots(1,2)
    # ax = fig.gca()
    if n_bursts is None and burst_range is None:
        axes[0].imshow(data1[:, :, :].reshape(-1,data1.shape[2]))
        axes[0].set_title('RTF Real Part')
        axes[1].imshow(data2[:, :, :].reshape(-1,data2.shape[2]))
        axes[1].set_title('RTF Imaginary Part')
    elif n_bursts is not None:
        axes[0].imshow(data1[:n_bursts, :, :].reshape(-1,data1.shape[2]))
        axes[0].set_title('RTF Real Part')
        axes[1].imshow(data2[:n_bursts, :, :].reshape(-1,data2.shape[2]))
        axes[1].set_title('RTF Imaginary Part')
    elif burst_range is not None:
        axes[0].imshow(data1[burst_range[0]:burst_range[1], :, :].reshape(-1, data1.shape[2]))
        axes[1].imshow(data2[burst_range[0]:burst_range[1], :, :].reshape(-1, data2.shape[2]))
    return fig



def plot_learning_curves(model, evaluate_every, training_only=False):
    if training_only == False:
        fig, ax = plt.subplots(1, 2)
        line1, = ax[0].plot((np.arange(len(model.train_white_losses_list)) + 1) * evaluate_every,
                            model.train_white_losses_list)
        line2, = ax[0].plot((np.arange(len(model.train_white_losses_list)) + 1) * evaluate_every,
                            model.val_white_losses_list)
        ax[0].set_title('Whitening Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend([line1, line2], ['Training data', 'Validation data'])

        line3, = ax[1].plot((np.arange(len(model.train_recon_losses_list)) + 1) * evaluate_every,
                            model.train_recon_losses_list)
        line4, = ax[1].plot((np.arange(len(model.train_recon_losses_list)) + 1) * evaluate_every,
                            model.val_recon_losses_list)
        ax[1].set_title('Reconstruction Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[0].legend([line3, line4], ['Training data', 'Validation data'])

    else:
        fig, ax = plt.subplots(1, 2)
        line1, = ax[0].plot((np.arange(len(model.train_white_losses_list)) + 1) * evaluate_every,
                            model.train_white_losses_list)
        ax[0].set_title('Whitening Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend([line1], ['Training data'])

        line3, = ax[1].plot((np.arange(len(model.train_recon_losses_list)) + 1) * evaluate_every,
                            model.train_recon_losses_list)
        ax[1].set_title('Reconstruction Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[0].legend([line3], ['Training data'])

    return fig


def visualize_embedding_xy_separated_2D(model, data, burst_locations, centers_only=False):
    embedding, _ = model.test(data)
    xmin = burst_locations[:, 0].min()
    xmax = burst_locations[:, 0].max()
    xrange = xmax - xmin
    ymin = burst_locations[:, 1].min()
    ymax = burst_locations[:, 1].max()
    yrange = ymax - ymin

    fig, ax = plt.subplots(1, 2)

    if not centers_only:
        ax[0].set_title('Color bursts by x location')
        ax[1].set_title('Color bursts by y location')
        xcolors = np.repeat((burst_locations[:, 0] - xmin) / xrange, data.shape[1])
        ycolors = np.repeat((burst_locations[:, 1] - ymin) / yrange, data.shape[1])
        ax[0].scatter(embedding.reshape(-1, 2)[:, 0], embedding.reshape(-1, 2)[:, 1], c=xcolors)
        ax[1].scatter(embedding.reshape(-1, 2)[:, 0], embedding.reshape(-1, 2)[:, 1], c=ycolors)

    else:
        ax[0].set_title('Color bursts\' centers by x location')
        ax[1].set_title('Color bursts\' centers by y location')
        xcolors = (burst_locations[:, 0] - xmin) / xrange
        ycolors = (burst_locations[:, 1] - ymin) / yrange
        ax[0].scatter(embedding[:, 0, 0], embedding[:, 0, 1], c=xcolors)
        ax[1].scatter(embedding[:, 0, 0], embedding[:, 0, 1], c=ycolors)
    return fig


# If data was inserted when X and Y coordinates are sorted:
def visualize_embedding_xy_combined_2D(model, data):
    embedding, _ = model.test(data)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot()
    colors = cm.viridis(np.linspace(0, 1, embedding.shape[0]))
    for i in range(embedding.shape[0]):
        burst_emb = embedding[i, :, :]
        ax.scatter(burst_emb[:, 0], burst_emb[:, 1], color=colors[i])
    return