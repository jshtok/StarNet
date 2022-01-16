import numpy as np


def py_returnCAMmap(activation, weights_LR):
    print(activation.shape)

    if activation.shape[0] == 1: # only one image
        n_feat, w, h = activation[0].shape
        act_vec = np.reshape(activation[0], [n_feat, w*h])
        n_top = weights_LR.shape[0]
        out = np.zeros([w, h, n_top])

        for t in range(n_top):
            weights_vec = np.reshape(weights_LR[t], [1, weights_LR[t].shape[0]])
            heatmap_vec = np.dot(weights_vec,act_vec)
            heatmap = np.reshape( np.squeeze(heatmap_vec) , [w, h])
            out[:,:,t] = heatmap
    else: # 10 images (over-sampling)
         raise Exception('Not implemented')

    return out