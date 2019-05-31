import sys
import numpy as np
import tensorflow as tf
import torch

sys.path.append('..')
from utils import build_model


def load_param(checkpoint_file, conversion_table, model_name):
    """
    Load parameters according to conversion_table.

    Args:
        checkpoint_file (string): pretrained checkpoint model file in tensorflow
        conversion_table (dict): { pytorch tensor in a model : checkpoint variable name }
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param_name = str(model_name) + '/' + tf_param_name
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)
        if tf_param_name.endswith('kernel'):  # for weight(kernel), we should do transpose
            tf_param = np.transpose(tf_param)
        if 'depthwise' in tf_param_name:
            tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        pyt_param.data = torch.from_numpy(tf_param)


def load_efficientnet(model, checkpoint_file, model_name='efficientnet-b0'):
    """
    Load PyTorch EfficientNet from TensorFlow checkpoint file
    """

    # This will store the entire conversion table
    conversion_table = {}
    merge = lambda dict1, dict2: {**dict1, **dict2}

    # All the weights not in the conv blocks
    conversion_table_for_weights_outside_blocks = {
        model._conv_stem.weight: 'stem/conv2d/kernel',  # [3, 3, 3, 32]),
        model._bn0.bias: 'stem/tpu_batch_normalization/beta',  # [32]),
        model._bn0.weight: 'stem/tpu_batch_normalization/gamma',  # [32]),
        model._bn0.running_mean: 'stem/tpu_batch_normalization/moving_mean',  # [32]),
        model._bn0.running_var: 'stem/tpu_batch_normalization/moving_variance',  # [32]),
        model._conv_head.weight: 'head/conv2d/kernel',  # [1, 1, 320, 1280]),
        model._bn1.bias: 'head/tpu_batch_normalization/beta',  # [1280]),
        model._bn1.weight: 'head/tpu_batch_normalization/gamma',  # [1280]),
        model._bn1.running_mean: 'head/tpu_batch_normalization/moving_mean',  # [32]),
        model._bn1.running_var: 'head/tpu_batch_normalization/moving_variance',  # [32]),
        model._fc.bias: 'head/dense/bias',  # [1000]),
        model._fc.weight: 'head/dense/kernel',  # [1280, 1000]),
    }
    conversion_table = merge(conversion_table, conversion_table_for_weights_outside_blocks)

    # The first conv block is special because it does not have _expand_conv
    conversion_table_for_first_block = {
        model._blocks[0]._project_conv.weight: 'blocks_0/conv2d/kernel',  # 1, 1, 32, 16]),
        model._blocks[0]._depthwise_conv.weight: 'blocks_0/depthwise_conv2d/depthwise_kernel',  # [3, 3, 32, 1]),
        model._blocks[0]._se_reduce.bias: 'blocks_0/se/conv2d/bias',  # , [8]),
        model._blocks[0]._se_reduce.weight: 'blocks_0/se/conv2d/kernel',  # , [1, 1, 32, 8]),
        model._blocks[0]._se_expand.bias: 'blocks_0/se/conv2d_1/bias',  # , [32]),
        model._blocks[0]._se_expand.weight: 'blocks_0/se/conv2d_1/kernel',  # , [1, 1, 8, 32]),
        model._blocks[0]._bn1.bias: 'blocks_0/tpu_batch_normalization/beta',  # [32]),
        model._blocks[0]._bn1.weight: 'blocks_0/tpu_batch_normalization/gamma',  # [32]),
        model._blocks[0]._bn1.running_mean: 'blocks_0/tpu_batch_normalization/moving_mean',
        model._blocks[0]._bn1.running_var: 'blocks_0/tpu_batch_normalization/moving_variance',
        model._blocks[0]._bn2.bias: 'blocks_0/tpu_batch_normalization_1/beta',  # [16]),
        model._blocks[0]._bn2.weight: 'blocks_0/tpu_batch_normalization_1/gamma',  # [16]),
        model._blocks[0]._bn2.running_mean: 'blocks_0/tpu_batch_normalization_1/moving_mean',
        model._blocks[0]._bn2.running_var: 'blocks_0/tpu_batch_normalization_1/moving_variance',
    }
    conversion_table = merge(conversion_table, conversion_table_for_first_block)

    # Conv blocks
    for i in range(1, len(model._blocks)):
        conversion_table_block = {
            model._blocks[i]._expand_conv.weight: 'blocks_' + str(i) + '/conv2d/kernel',
            model._blocks[i]._project_conv.weight: 'blocks_' + str(i) + '/conv2d_1/kernel',
            model._blocks[i]._depthwise_conv.weight: 'blocks_' + str(i) + '/depthwise_conv2d/depthwise_kernel',
            model._blocks[i]._se_reduce.bias: 'blocks_' + str(i) + '/se/conv2d/bias',
            model._blocks[i]._se_reduce.weight: 'blocks_' + str(i) + '/se/conv2d/kernel',
            model._blocks[i]._se_expand.bias: 'blocks_' + str(i) + '/se/conv2d_1/bias',
            model._blocks[i]._se_expand.weight: 'blocks_' + str(i) + '/se/conv2d_1/kernel',
            model._blocks[i]._bn0.bias: 'blocks_' + str(i) + '/tpu_batch_normalization/beta',
            model._blocks[i]._bn0.weight: 'blocks_' + str(i) + '/tpu_batch_normalization/gamma',
            model._blocks[i]._bn0.running_mean: 'blocks_' + str(i) + '/tpu_batch_normalization/moving_mean',
            model._blocks[i]._bn0.running_var: 'blocks_' + str(i) + '/tpu_batch_normalization/moving_variance',
            model._blocks[i]._bn1.bias: 'blocks_' + str(i) + '/tpu_batch_normalization_1/beta',
            model._blocks[i]._bn1.weight: 'blocks_' + str(i) + '/tpu_batch_normalization_1/gamma',
            model._blocks[i]._bn1.running_mean: 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_mean',
            model._blocks[i]._bn1.running_var: 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_variance',
            model._blocks[i]._bn2.bias: 'blocks_' + str(i) + '/tpu_batch_normalization_2/beta',
            model._blocks[i]._bn2.weight: 'blocks_' + str(i) + '/tpu_batch_normalization_2/gamma',
            model._blocks[i]._bn2.running_mean: 'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_mean',
            model._blocks[i]._bn2.running_var: 'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_variance',
        }
        conversion_table = merge(conversion_table, conversion_table_block)

    # Load TensorFlow parameters into PyTorch model
    load_param(checkpoint_file, conversion_table, model_name)
    return conversion_table


if __name__ == '__main__':
    # Convert and save as a PyTorch file for easier loading in the future
    tf_file_to_load = 'efficientnet-b0/model.ckpt'
    model_name = 'efficientnet-b0'
    model = build_model(model_name)
    load_efficientnet(model, tf_file_to_load, model_name=model_name)
    output_file = model_name + '.pth'
    torch.save(model.state_dict(), output_file)
    print('Saved model to', output_file)
