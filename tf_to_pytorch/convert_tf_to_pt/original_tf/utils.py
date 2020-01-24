# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_function  # pylint:disable=g-direct-tensorflow-import


def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.97,
                        decay_epochs=2.4,
                        total_steps=None,
                        warmup_epochs=5):
  """Build learning rate."""
  if lr_decay_type == 'exponential':
    assert steps_per_epoch is not None
    decay_steps = steps_per_epoch * decay_epochs
    lr = tf.train.exponential_decay(
        initial_lr, global_step, decay_steps, decay_factor, staircase=True)
  elif lr_decay_type == 'cosine':
    assert total_steps is not None
    lr = 0.5 * initial_lr * (
        1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
  elif lr_decay_type == 'constant':
    lr = initial_lr
  else:
    assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

  if warmup_epochs:
    logging.info('Learning rate warmup_epochs: %d', warmup_epochs)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    warmup_lr = (
        initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
            warmup_steps, tf.float32))
    lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

  return lr


def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
  """Build optimizer."""
  if optimizer_name == 'sgd':
    logging.info('Using SGD optimizer')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  elif optimizer_name == 'momentum':
    logging.info('Using Momentum optimizer')
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum)
  elif optimizer_name == 'rmsprop':
    logging.info('Using RMSProp optimizer')
    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                          epsilon)
  else:
    logging.fatal('Unknown optimizer: %s', optimizer_name)

  return optimizer


class TpuBatchNormalization(tf.layers.BatchNormalization):
  # class TpuBatchNormalization(tf.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t, num_shards_per_group):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                         % (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 8)
    logging.info('TpuBatchNormalization with num_shards_per_group %s',
                 num_shards_per_group)
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(
          shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


class BatchNormalization(tf.layers.BatchNormalization):
  """Fixed default name of BatchNormalization to match TpuBatchNormalization."""

  def __init__(self, name='tpu_batch_normalization', **kwargs):
    super(BatchNormalization, self).__init__(name=name, **kwargs)


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random_uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.div(inputs, survival_prob) * binary_tensor
  return output


def archive_ckpt(ckpt_eval, ckpt_objective, ckpt_path):
  """Archive a checkpoint if the metric is better."""
  ckpt_dir, ckpt_name = os.path.split(ckpt_path)

  saved_objective_path = os.path.join(ckpt_dir, 'best_objective.txt')
  saved_objective = float('-inf')
  if tf.gfile.Exists(saved_objective_path):
    with tf.gfile.GFile(saved_objective_path, 'r') as f:
      saved_objective = float(f.read())
  if saved_objective > ckpt_objective:
    logging.info('Ckpt %s is worse than %s', ckpt_objective, saved_objective)
    return False

  filenames = tf.gfile.Glob(ckpt_path + '.*')
  if filenames is None:
    logging.info('No files to copy for checkpoint %s', ckpt_path)
    return False

  # Clear the old folder.
  dst_dir = os.path.join(ckpt_dir, 'archive')
  if tf.gfile.Exists(dst_dir):
    tf.gfile.DeleteRecursively(dst_dir)
  tf.gfile.MakeDirs(dst_dir)

  # Write checkpoints.
  for f in filenames:
    dest = os.path.join(dst_dir, os.path.basename(f))
    tf.gfile.Copy(f, dest, overwrite=True)
  ckpt_state = tf.train.generate_checkpoint_state_proto(
      dst_dir,
      model_checkpoint_path=ckpt_name,
      all_model_checkpoint_paths=[ckpt_name])
  with tf.gfile.GFile(os.path.join(dst_dir, 'checkpoint'), 'w') as f:
    f.write(str(ckpt_state))
  with tf.gfile.GFile(os.path.join(dst_dir, 'best_eval.txt'), 'w') as f:
    f.write('%s' % ckpt_eval)

  # Update the best objective.
  with tf.gfile.GFile(saved_objective_path, 'w') as f:
    f.write('%f' % ckpt_objective)

  logging.info('Copying checkpoint %s to %s', ckpt_path, dst_dir)
  return True


def get_ema_vars():
  """Get all exponential moving average (ema) variables."""
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    # We maintain mva for batch norm moving mean and variance as well.
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  return list(set(ema_vars))


class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, tf.layers.Layer):
  """Wrap keras DepthwiseConv2D to tf.layers."""

  pass


class EvalCkptDriver(object):
  """A driver for running eval inference.

  Attributes:
    model_name: str. Model name to eval.
    batch_size: int. Eval batch size.
    image_size: int. Input image size, determined by model name.
    num_classes: int. Number of classes, default to 1000 for ImageNet.
    include_background_label: whether to include extra background label.
  """

  def __init__(self,
               model_name,
               batch_size=1,
               image_size=224,
               num_classes=1000,
               include_background_label=False):
    """Initialize internal variables."""
    self.model_name = model_name
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.include_background_label = include_background_label
    self.image_size = image_size

  def restore_model(self, sess, ckpt_dir, enable_ema=True, export_ckpt=None):
    """Restore variables from checkpoint dir."""
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if enable_ema:
      ema = tf.train.ExponentialMovingAverage(decay=0.0)
      ema_vars = get_ema_vars()
      var_dict = ema.variables_to_restore(ema_vars)
      ema_assign_op = ema.apply(ema_vars)
    else:
      var_dict = get_ema_vars()
      ema_assign_op = None

    tf.train.get_or_create_global_step()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_dict, max_to_keep=1)
    saver.restore(sess, checkpoint)

    if export_ckpt:
      if ema_assign_op is not None:
        sess.run(ema_assign_op)
      saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
      saver.save(sess, export_ckpt)

  def build_model(self, features, is_training):
    """Build model with input features."""
    del features, is_training
    raise ValueError('Must be implemented by subclasses.')

  def get_preprocess_fn(self):
    raise ValueError('Must be implemented by subclsses.')

  def build_dataset(self, filenames, labels, is_training):
    """Build input dataset."""
    batch_drop_remainder = False
    if 'condconv' in self.model_name and not is_training:
      # CondConv layers can only be called with known batch dimension. Thus, we
      # must drop all remaining examples that do not make up one full batch.
      # To ensure all examples are evaluated, use a batch size that evenly
      # divides the number of files.
      batch_drop_remainder = True
      num_files = len(filenames)
      if num_files % self.batch_size != 0:
        tf.logging.warn('Remaining examples in last batch are not being '
                        'evaluated.')
    filenames = tf.constant(filenames)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    def _parse_function(filename, label):
      image_string = tf.read_file(filename)
      preprocess_fn = self.get_preprocess_fn()
      image_decoded = preprocess_fn(
          image_string, is_training, image_size=self.image_size)
      image = tf.cast(image_decoded, tf.float32)
      return image, label

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(self.batch_size,
                            drop_remainder=batch_drop_remainder)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

  def run_inference(self,
                    ckpt_dir,
                    image_files,
                    labels,
                    enable_ema=True,
                    export_ckpt=None):
    """Build and run inference on the target images and labels."""
    label_offset = 1 if self.include_background_label else 0
    with tf.Graph().as_default(), tf.Session() as sess:
      images, labels = self.build_dataset(image_files, labels, False)
      probs = self.build_model(images, is_training=False)
      if isinstance(probs, tuple):
        probs = probs[0]

      self.restore_model(sess, ckpt_dir, enable_ema, export_ckpt)

      prediction_idx = []
      prediction_prob = []
      for _ in range(len(image_files) // self.batch_size):
        out_probs = sess.run(probs)
        idx = np.argsort(out_probs)[::-1]
        prediction_idx.append(idx[:5] - label_offset)
        prediction_prob.append([out_probs[pid] for pid in idx[:5]])

      # Return the top 5 predictions (idx and prob) for each image.
      return prediction_idx, prediction_prob

  def eval_example_images(self,
                          ckpt_dir,
                          image_files,
                          labels_map_file,
                          enable_ema=True,
                          export_ckpt=None):
    """Eval a list of example images.

    Args:
      ckpt_dir: str. Checkpoint directory path.
      image_files: List[str]. A list of image file paths.
      labels_map_file: str. The labels map file path.
      enable_ema: enable expotential moving average.
      export_ckpt: export ckpt folder.

    Returns:
      A tuple (pred_idx, and pred_prob), where pred_idx is the top 5 prediction
      index and pred_prob is the top 5 prediction probability.
    """
    classes = json.loads(tf.gfile.Open(labels_map_file).read())
    pred_idx, pred_prob = self.run_inference(
        ckpt_dir, image_files, [0] * len(image_files), enable_ema, export_ckpt)
    for i in range(len(image_files)):
      print('predicted class for image {}: '.format(image_files[i]))
      for j, idx in enumerate(pred_idx[i]):
        print('  -> top_{} ({:4.2f}%): {}  '.format(j, pred_prob[i][j] * 100,
                                                    classes[str(idx)]))
    return pred_idx, pred_prob

  def eval_imagenet(self, ckpt_dir, imagenet_eval_glob,
                    imagenet_eval_label, num_images, enable_ema, export_ckpt):
    """Eval ImageNet images and report top1/top5 accuracy.

    Args:
      ckpt_dir: str. Checkpoint directory path.
      imagenet_eval_glob: str. File path glob for all eval images.
      imagenet_eval_label: str. File path for eval label.
      num_images: int. Number of images to eval: -1 means eval the whole
        dataset.
      enable_ema: enable expotential moving average.
      export_ckpt: export checkpoint folder.

    Returns:
      A tuple (top1, top5) for top1 and top5 accuracy.
    """
    imagenet_val_labels = [int(i) for i in tf.gfile.GFile(imagenet_eval_label)]
    imagenet_filenames = sorted(tf.gfile.Glob(imagenet_eval_glob))
    if num_images < 0:
      num_images = len(imagenet_filenames)
    image_files = imagenet_filenames[:num_images]
    labels = imagenet_val_labels[:num_images]

    pred_idx, _ = self.run_inference(
        ckpt_dir, image_files, labels, enable_ema, export_ckpt)
    top1_cnt, top5_cnt = 0.0, 0.0
    for i, label in enumerate(labels):
      top1_cnt += label in pred_idx[i][:1]
      top5_cnt += label in pred_idx[i][:5]
      if i % 100 == 0:
        print('Step {}: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(
            i, 100 * top1_cnt / (i + 1), 100 * top5_cnt / (i + 1)))
        sys.stdout.flush()
    top1, top5 = 100 * top1_cnt / num_images, 100 * top5_cnt / num_images
    print('Final: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%'.format(top1, top5))
    return top1, top5
