import numpy as np

def format_for_nn_training(imgs_np: list):
    """
    From a list of numpy arrays, it generates the proper NN output.
    For example. If input is a 3 mult-stream network with batch of 2: [#_streams,batch_size,width,height,depth,bands]
    :param imgs_np:
    :param batch_size:
    :return:
    """
    batch_size = len(imgs_np)
    tot_num_streams = len(imgs_np[0])
    nn_array = []
    for cur_stream_idx in range(tot_num_streams):
        input_single_batch = []
        # Iterate each image = stream
        for cur_batch_idx in range(batch_size):
            input_single_batch.append(np.expand_dims(imgs_np[cur_batch_idx][cur_stream_idx], axis=4))
        input_single_batch = np.array(input_single_batch)
        nn_array.append(input_single_batch)
    return nn_array

def format_for_nn_classification(imgs_np: list):
    """
    From a list of numpy arrays, it generates the proper NN output.
    For example. If input is a 3 mult-stream network with batch of 2: [#_streams,batch_size,width,height,depth,bands]
    EXAMPLE: INPUT: [3][168,168,168]   --> output [3,1,168,168,168,1]
    :param imgs_np:
    :param batch_size:
    :return:
    """
    tot_num_streams = len(imgs_np)
    nn_array = []
    for cur_img_idx in range(tot_num_streams):
        temp = np.expand_dims(np.expand_dims(imgs_np[cur_img_idx], axis=0), axis=4)
        nn_array.append(temp)
    return nn_array