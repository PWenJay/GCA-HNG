from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import copy
import yaml
import numpy as np
import os.path as osp
from easydict import EasyDict
from ast import literal_eval
from packaging import version
import torch
from torch.nn import init


class AttrDict(EasyDict):
    IMMUTABLE = '__immutable__'

    def __init__(self, *args):
        super(EasyDict, self).__init__(*args)

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Dataset to train on
__C.TRAIN.DATASET = 'data/CARS196/class/train'

# Models saved in
__C.TRAIN.MODEL_PATH = 'saved_models/models'

# Images sampled per class
__C.TRAIN.NUM_PER_CLASS = 2

# Number of sampled classes
__C.TRAIN.NUM_SAMPLED_CLASSES = 40

# Tensorboard log file path
__C.TRAIN.BOARD_LOG_PATH = 'saved_models/tensorboard'

# Dataset to train on
__C.TRAIN.EPOCHES = 300

__C.TRAIN.SCALE = (0.08, 1)

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 8


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Datasets to test on
# Available dataset list: datasets.dataset_catalog.DATASETS.keys()
# If multiple datasets are listed, testing is performed on each one sequentially
__C.TEST.DATASET = 'data/CARS196/class/test'

# eval iter
__C.TEST.EVAL_EPOCH = 10

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The backbone encoder body to use
__C.MODEL.BACKBONE = ''  # support Googlenet Resnet50 DINO

# pretrain backbone
__C.MODEL.PRETRAINED = True

# norm embedding output
__C.MODEL.NORM = False

# eva+max pooling
__C.MODEL.POOLING = False

# embedding size
__C.MODEL.EMBEDDING_SIZE = 512

__C.MODEL.GNN_LAYER = 2

__C.MODEL.ATT_HEAD = 8

# ---------------------------------------------------------------------------- #
# loss options
# ---------------------------------------------------------------------------- #
__C.LOSS = AttrDict()

#using hard mining

__C.LOSS.DIV_WEIGHT = 10

__C.LOSS.DOT_WEIGHT = 20

__C.LOSS.ALPHA = 90.0

__C.LOSS.BETA = 10000

__C.LOSS.SOFTMAX_FACTOR = 10000

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.1

#classifier lr
__C.SOLVER.CLASSIFIER_LR = 0.001

#GNN lr
__C.SOLVER.GNN_LR = 0.0003

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'

__C.SOLVER.EMBEDDING_RATIO = 1.0

# Some LR Policies (by example):
# 'step'
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** (cur_iter // SOLVER.STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.METRIC_GAMMA = 0.1

__C.SOLVER.GNN_GAMMA = 0.1

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies

# StepLR decay steps
__C.SOLVER.METRIC_DECAY_STEPS = 10

# StepLR decay steps
__C.SOLVER.GNN_DECAY_STEPS = 10

__C.SOLVER.BN_FREEZE = False

__C.SOLVER.PATCH_FREEZE = False

# CLIP Gradient L2 Nrom
__C.SOLVER.GRAD_CLIP = True

__C.SOLVER.GRAD_L2_CLIP = 0.35

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_EPOCH = 1

# ---------------------------------------------------------------------------- #
# MISC options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing)
__C.NUM_GPUS = 1

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__)))

# Output basedir
__C.OUTPUT_DIR = 'Outputs'

# ------------------------------
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# [Infered value]
__C.CUDA = False

__C.DEBUG = False

# [Infered value]
__C.PYTORCH_VERSION_LESS_THAN_040 = False


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        assert __C.MODEL.IMAGENET_PRETRAINED_WEIGHTS, \
            "Path to the weight file must not be empty to load imagenet pertrained resnets."
    if version.parse(torch.__version__) < version.parse('0.4.0'):
        __C.PYTORCH_VERSION_LESS_THAN_040 = True
        # create alias for PyTorch version less than 0.4.0
        init.uniform_ = init.uniform
        init.normal_ = init.normal
        init.constant_ = init.constant
        init.kaiming_normal_ = init.kaiming_normal
        torch.nn.utils.clip_grad_norm_ = torch.nn.utils.clip_grad_norm
        def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
            tensor.requires_grad = requires_grad
            tensor._backward_hooks = backward_hooks
            return tensor
        torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f,Loader=yaml.FullLoader))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
