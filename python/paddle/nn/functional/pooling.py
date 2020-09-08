#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define pooling functions
from ...fluid.layers import pool2d  #DEFINE_ALIAS
from ...fluid.layers import pool3d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool2d  #DEFINE_ALIAS
from ...fluid.layers import adaptive_pool3d  #DEFINE_ALIAS
from ...fluid import core
from ...fluid.framework import in_dygraph_mode
from ...fluid.layers import utils, LayerHelper, unsqueeze, squeeze
from ...fluid.data_feeder import check_type, check_variable_and_dtype

__all__ = [
    'pool2d',
    'pool3d',
    'adaptive_pool2d',
    'adaptive_pool3d',
    'avg_pool1d',
    'avg_pool2d',
    'avg_pool3d',
    'max_pool1d',
    'max_pool2d',
    'max_pool3d',
    'adaptive_avg_pool1d',
    'adaptive_avg_pool2d',
    'adaptive_avg_pool3d',
    'adaptive_max_pool1d',
    'adaptive_max_pool2d',
    'adaptive_max_pool3d',
]


def _is_list_or_tuple(input):
    return isinstance(input, (list, tuple))


def _check_input(x, dimension):
    if len(x.shape) != dimension:
        raise ValueError(
            "Excepted Input X is {}-D tensor, but received {}-D {}".format(
                dimension, len(x.shape), type(x)))


def _check_instance(x, x_name, types=(int, float)):

    if not isinstance(x, types):
        raise ValueError("Excepted {} type for {} but received type: {}. ".
                         format(types, x_name, type(x)))


def _zero_padding_in_batch_and_channel(padding, channel_last):
    if channel_last:
        return list(padding[0]) == [0, 0] and list(padding[-1]) == [0, 0]
    else:
        return list(padding[0]) == [0, 0] and list(padding[1]) == [0, 0]


def _exclude_padding_in_batch_and_channel(padding, channel_last):
    padding_ = padding[1:-1] if channel_last else padding[2:]
    padding_ = [elem for pad_a_dim in padding_ for elem in pad_a_dim]
    return padding_


def _channel_last(data_format, num_dims):
    if num_dims == 1:
        if data_format not in ['NCL', 'NLC']:
            raise ValueError(
                "Attr(data_format) should be 'NCL' or 'NLC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NLC" else False
    if num_dims == 2:
        if data_format not in ['NCHW', 'NHWC']:
            raise ValueError(
                "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NHWC" else False
    if num_dims == 3:
        if data_format not in ['NCDHW', 'NDHWC']:
            raise ValueError(
                "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
                "Attr(data_format): %s" % str(data_format))
        else:
            return True if data_format == "NDHWC" else False


def _update_padding_nd(padding, num_dims, channel_last=False, ceil_mode=False):
    if isinstance(padding, str):
        padding = padding.upper()
        if padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown padding: '{}'. It can only be 'SAME' or 'VALID'.".
                format(padding))
        if padding == "VALID":
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")

            padding_algorithm = "VALID"
            padding = [0] * num_dims
        else:
            padding_algorithm = "SAME"
            padding = [0] * num_dims
    elif _is_list_or_tuple(padding):
        # for padding like
        # [(pad_before, pad_after), (pad_before, pad_after), ...]
        # padding for batch_dim and channel_dim included
        if len(padding) == 2 + num_dims and _is_list_or_tuple(padding[0]):
            if not _zero_padding_in_batch_and_channel(padding, channel_last):
                raise ValueError(
                    "Non-zero padding({}) in the batch or channel dimensions "
                    "is not supported.".format(padding))
            padding_algorithm = "EXPLICIT"
            padding = _exclude_padding_in_batch_and_channel(padding,
                                                            channel_last)
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_before, pad_after, pad_before, pad_after, ...]
        elif len(padding) == 2 * num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, 2 * num_dims, 'padding')
            if utils._is_symmetric_padding(padding, num_dims):
                padding = padding[0::2]
        # for padding like [pad_d1, pad_d2, ...]
        elif len(padding) == num_dims and isinstance(padding[0], int):
            padding_algorithm = "EXPLICIT"
            padding = utils.convert_to_list(padding, num_dims, 'padding')
        else:
            raise ValueError("Invalid padding: {}".format(padding))
    # for integer padding
    else:
        padding_algorithm = "EXPLICIT"
        padding = utils.convert_to_list(padding, num_dims, 'padding')
    return padding, padding_algorithm


def _expand_low_nd_padding(padding):
    #1d to 2d fake input
    if len(padding) == 2:
        padding = [0] * 2 + padding
    elif len(padding) == 1:
        padding = [0] + padding
    else:
        raise ValueError(
            "The size of padding's dimmention should be 1 or 2. But got padding={}".
            format(padding))
    return padding


def avg_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               count_include_pad=True,
               ceil_mode=False,
               name=None):
    """
    This API implements average pooling 1d operation,
    See more details in :ref:`api_nn_pooling_AvgPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L]. where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type is float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `True`.
        ceil_mode (bool): ${ceil_mode_comment}Whether to use the ceil function to calculate output height and width.
            If it is set to False, the floor function will be used. The default value is False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ValueError: If `padding` is a list or tuple but its length is greater than 1.
        ShapeError: If the input is not a 3-D tensor.
        ShapeError: If the output's shape calculated is not greater than 0.

    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()
          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          out = F.avg_pool1d(data, kernel_size=2, stride=2, padding=0)
          # out shape: [1, 3, 16]
    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'avg_pool1d')
    _check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = utils.convert_to_list(kernel_size, 1, 'kernel_size')
    kernel_size = [1] + kernel_size
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 1, 'pool_stride')
        stride = [1] + stride

    channel_last = _channel_last("NCL", 1)
    padding, padding_algorithm = _update_padding_nd(
        padding, 1, channel_last=channel_last, ceil_mode=ceil_mode)

    # use 2d to implenment 1d should expand padding in advance.
    padding = _expand_low_nd_padding(padding)

    if in_dygraph_mode():
        output = core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'strides', stride, 'paddings', padding, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)
        return squeeze(output, [2])

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    return squeeze(pool_out, [2])


def avg_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=None,
               data_format="NCHW",
               name=None):
    """
    This API implements average pooling 2d operation.
    See more details in :ref:`api_nn_pooling_AvgPool2d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If it is a tuple or list,
            it must contain two integers, (kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The stride size. If it is a tuple or list,
            it must contain two integers, (stride_Height, stride_Width).
            Otherwise, the stride size will be a square of an int.

        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is `true`.
        divisor_override (float): if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()
          # avg pool2d
          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          out = F.avg_pool2d(x,
                                kernel_size=2,
                                stride=2, padding=0)
          # out.shape [1, 3, 16, 16]
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'avg_pool2d')
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    channel_last = _channel_last(data_format, 2)
    padding, padding_algorithm = _update_padding_nd(
        padding, 2, channel_last, ceil_mode=ceil_mode)

    if in_dygraph_mode():
        output = core.ops.pool2d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'global_pooling',
            False, 'padding_algorithm', padding_algorithm, 'strides', stride,
            'paddings', padding, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)
        if divisor_override is None:
            return output
        else:
            _check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1]) / divisor_override

    op_type = 'pool2d'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs={"Out": pool_out},
        attrs={
            "pooling_type": "avg",
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    if divisor_override is None:
        return pool_out
    else:
        _check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1]) / divisor_override


def avg_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               count_include_pad=True,
               divisor_override=None,
               data_format="NCDHW",
               name=None):
    """
    This API implements average pooling 3d operation.
    See more details in :ref:`api_nn_pooling_AvgPool3d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W], where `N` represents the batch size, `C` represents
                          the number of channels, `D`, `H` and `W` represent the depth, height and width of the feature respectively.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        count_include_pad (bool): Whether to exclude padding points in average pooling
                          mode, default is True.
        divisor_override (int|float) if specified, it will be used as divisor, otherwise kernel_size will be used. Default None.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle.fluid as fluid
          import paddle
          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          # avg pool3d
          out = paddle.nn.functional.avg_pool3d(
                                            x,
                                            kernel_size = 2,
                                            stride = 2,
                                            padding=0)
          # out.shape: [1, 3, 16, 16, 16]
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    channel_last = _channel_last(data_format, 3)
    padding, padding_algorithm = _update_padding_nd(
        padding, 3, channel_last=channel_last, ceil_mode=ceil_mode)

    if in_dygraph_mode():
        output = core.ops.pool3d(
            x, 'pooling_type', 'avg', 'ksize', kernel_size, 'strides', stride,
            'paddings', padding, 'global_pooling', False, 'padding_algorithm',
            padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
            'use_mkldnn', False, 'exclusive', not count_include_pad,
            'data_format', data_format)
        if divisor_override is None:
            return output
        else:
            _check_instance(divisor_override, "divisor_override")
            return output * (kernel_size[0] * kernel_size[1] *
                             kernel_size[2]) / divisor_override

    op_type = "pool3d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'avg',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": not count_include_pad,
            "data_format": data_format,
        })

    if divisor_override is None:
        return pool_out
    else:
        _check_instance(divisor_override, "divisor_override")
        return pool_out * (kernel_size[0] * kernel_size[1] *
                           kernel_size[2]) / divisor_override


def max_pool1d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               name=None):
    """
    This API implements max pooling 1d opereation.
    See more details in :ref:`api_nn_pooling_MaxPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 3-D tensor with
                          shape [N, C, L], where `N` is batch size, `C` is the number of channels,
                          `L` is the length of the feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain an integer.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain an integer.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An integer, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 1, which means the feature map is zero padded by the size of `padding[0]` on every sides.
            4. A list[int] or tuple(int) whose length is 2. It has the form [pad_before, pad_after].
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        return_indices (bool): Whether return the max indices along with the outputs. default is `False`.
        ceil_mode (bool): Whether to use the ceil function to calculate output height and width. False is the default.
            If it is set to False, the floor function will be used. Default False.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.

    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the input is not a 3-D tensor.
        ShapeError: If the output's shape calculated is not greater than 0.

    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          paddle.disable_static()
          data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
          pool_out = F.max_pool1d(data, kernel_size=2, stride=2, padding=0)
          # pool_out shape: [1, 3, 16]
          pool_out, indices = F.max_pool1d(data, kernel_size=2, stride=2, padding=0, return_indices=True)
          # pool_out shape: [1, 3, 16],  indices shape: [1, 3, 16]
    """
    """NCL to NCHW"""
    data_format = "NCHW"
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool1d')
    _check_input(x, 3)
    x = unsqueeze(x, [2])
    kernel_size = [1] + utils.convert_to_list(kernel_size, 1, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = [1] + utils.convert_to_list(stride, 1, 'pool_stride')

    padding, padding_algorithm = _update_padding_nd(
        padding, 1, ceil_mode=ceil_mode)

    # use 2d to implenment 1d should expand padding in advance.
    padding = _expand_low_nd_padding(padding)

    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'ksize', kernel_size, 'global_pooling', False, 'strides', stride,
            'paddings', padding, 'padding_algorithm', padding_algorithm,
            'use_cudnn', True, 'ceil_mode', ceil_mode, 'use_mkldnn', False,
            'exclusive', True, 'data_format', data_format)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

    op_type = 'max_pool2d_with_index'
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])


def max_pool2d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               data_format="NCHW",
               name=None):
    """
    This API implements max pooling 2d operation.
    See more details in :ref:`api_nn_pooling_MaxPool2d` .

    Args:
        x (Tensor): The input tensor of pooling operator which is a 4-D tensor with
                          shape [N, C, H, W]. The format of input tensor is `"NCHW"` or
                          `"NHWC"`, where `N` is batch size, `C` is the number of channels,
                          `H` is the height of the feature, and `W` is the width of the
                          feature. The data type if float32 or float64.
        kernel_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two integers, (kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be a square of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain two integers, (stride_Height, stride_Width).
            Otherwise, the pool stride size will be a square of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 2, [pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 4. [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): when True, will use `ceil` instead of `floor` to compute the output shape
        return_indices (bool): Whether to return the max indices along with the outputs. Default False, only support `"NCHW"` data format
        data_format (string): The data format of the input and output data. An optional string from: `"NCHW"`, `"NHWC"`.
                        The default is `"NCHW"`. When it is `"NCHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()
          # max pool2d
          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32]).astype(np.float32))
          out = F.max_pool2d(x,
                                kernel_size=2,
                                stride=2, padding=0)
          # output.shape [1, 3, 16, 16]
          # for return_indices=True
          out, max_indices = F.max_pool2d(x,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0,
                                             return_indices=True)
          # out.shape [1, 3, 16, 16], max_indices.shape [1, 3, 16, 16],
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool2d')
    kernel_size = utils.convert_to_list(kernel_size, 2, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 2, 'pool_stride')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    channel_last = True if data_format == "NHWC" else False

    padding, padding_algorithm = _update_padding_nd(
        padding, num_dims=2, channel_last=channel_last, ceil_mode=ceil_mode)

    if data_format == "NHWC" and return_indices:
        raise ValueError(
            "When setting return_indices to true, data_format must be set to NCHW in API:max_pool2d"
        )

    if in_dygraph_mode():
        if data_format == "NCHW":
            output = core.ops.max_pool2d_with_index(
                x, 'ksize', kernel_size, 'global_pooling', False, 'strides',
                stride, 'paddings', padding, 'padding_algorithm',
                padding_algorithm, 'use_cudnn', True, 'ceil_mode', ceil_mode,
                'use_mkldnn', False, 'exclusive', True, 'data_format',
                data_format)
            return output if return_indices else output[0]
        elif data_format == "NHWC" and not return_indices:
            output = core.ops.pool2d(
                x, 'pooling_type', 'max', 'ksize', kernel_size,
                'global_pooling', False, 'padding_algorithm', padding_algorithm,
                'strides', stride, 'paddings', padding, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output

    op_type = 'max_pool2d_with_index' if data_format == "NCHW" else "max_pool2d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": True,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_indices else pool_out


def max_pool3d(x,
               kernel_size,
               stride=None,
               padding=0,
               return_indices=False,
               ceil_mode=False,
               data_format="NCDHW",
               name=None):
    """
    This API implements max pooling 2d operation.
    See more details in :ref:`api_nn_pooling_MaxPool3d` .
    Args:
        x (Tensor): The input tensor of pooling operator, which is a 5-D tensor with
                          shape [N, C, D, H, W]. The format of input tensor is `"NCDHW"` or `"NDHWC"`, where N represents batch size, C represents the number of channels, D, H and W represent the depth, height and width of the feature respectively.
        kernel_size (int|list|tuple): The pool kernel size. If the kernel size
            is a tuple or list, it must contain three integers,
            (kernel_size_Depth, kernel_size_Height, kernel_size_Width).
            Otherwise, the pool kernel size will be the cube of an int.
        stride (int|list|tuple): The pool stride size. If pool stride size is a tuple or list,
            it must contain three integers, [stride_Depth, stride_Height, stride_Width).
            Otherwise, the pool stride size will be a cube of an int.
        padding (string|int|list|tuple): The padding size. Padding could be in one of the following forms.
            1. A string in ['valid', 'same'].
            2. An int, which means the feature map is zero padded by size of `padding` on every sides.
            3. A list[int] or tuple(int) whose length is 3, [pad_depth, pad_height, pad_weight] whose value means the padding size of each dimension.
            4. A list[int] or tuple(int) whose length is 6. [pad_depth_front, pad_depth_back, pad_height_top, pad_height_bottom, pad_width_left, pad_width_right] whose value means the padding size of each side.
            5. A list or tuple of pairs of integers. It has the form [[pad_before, pad_after], [pad_before, pad_after], ...]. Note that, the batch dimension and channel dimension should be [0,0] or (0,0).
            The default value is 0.
        ceil_mode (bool): ${ceil_mode_comment}
        return_indices (bool): Whether to return the max indices along with the outputs. Default False. Only support "NDCHW" data_format.
        data_format (string): The data format of the input and output data. An optional string from: `"NCDHW"`, `"NDHWC"`.
                        The default is `"NCDHW"`. When it is `"NCDHW"`, the data is stored in the order of:
                        `[batch_size, input_channels, input_depth, input_height, input_width]`.
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of pooling result. The data type is same as input tensor.
    Raises:
        ValueError: If `padding` is a string, but not "SAME" or "VALID".
        ValueError: If `padding` is "VALID", but `ceil_mode` is True.
        ShapeError: If the output's shape calculated is not greater than 0.
    Examples:
        .. code-block:: python
          import paddle
          import paddle.nn.functional as F
          import numpy as np
          paddle.disable_static()
          # max pool3d
          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          output = F.max_pool2d(x,
                                kernel_size=2,
                                stride=2, padding=0)
          output.shape [1, 3, 16, 16, 16]
          # for return_indices=True
          x = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32, 32, 32]).astype(np.float32))
          output, max_indices = paddle.nn.functional.max_pool3d(x,
                                        kernel_size = 2,
                                        stride = 2,
                                        padding=0,
                                        return_indices=True)
          # output.shape [None, 3, 16, 16, 16], max_indices.shape [None, 3, 16, 16, 16],
    """
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'max_pool3d')
    kernel_size = utils.convert_to_list(kernel_size, 3, 'pool_size')
    if stride is None:
        stride = kernel_size
    else:
        stride = utils.convert_to_list(stride, 3, 'pool_stride')

    channel_last = _channel_last(data_format, 3)

    padding, padding_algorithm = _update_padding_nd(
        padding, 3, channel_last=channel_last, ceil_mode=ceil_mode)

    if data_format == "NDHWC" and return_indices:
        raise ValueError(
            "When setting return_indices to true, data_format must be set to NCDHW in API:max_pool3d"
        )

    if in_dygraph_mode():
        if data_format == "NCDHW":
            output = core.ops.max_pool3d_with_index(
                x, 'pooling_type', 'max', 'ksize', kernel_size, 'strides',
                stride, 'paddings', padding, 'global_pooling', False,
                'padding_algorithm', padding_algorithm, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output if return_indices else output[0]
        elif data_format == "NDHWC" and not return_indices:
            output = core.ops.pool3d(
                x, 'pooling_type', 'max', 'ksize', kernel_size,
                'global_pooling', False, 'padding_algorithm', padding_algorithm,
                'strides', stride, 'paddings', padding, 'use_cudnn', True,
                'ceil_mode', ceil_mode, 'use_mkldnn', False, 'exclusive', True,
                'data_format', data_format)
            return output

    op_type = "max_pool3d_with_index" if data_format == "NCDHW" else "max_pool3d"
    helper = LayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=op_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": kernel_size,
            "global_pooling": False,
            "strides": stride,
            "paddings": padding,
            "padding_algorithm": padding_algorithm,
            "use_cudnn": True,
            "ceil_mode": ceil_mode,
            "use_mkldnn": False,
            "exclusive": False,
            "data_format": data_format,
        })

    return (pool_out, mask) if return_indices else pool_out


def adaptive_avg_pool1d(x, output_size, name=None):
    """
    This API implements adaptive average pooling 1d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
                it must contain one int.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.
    Returns:
            Tensor: The output tensor of adaptive average pooling result. The data type is same
                      as input tensor.
    Raises:
            ValueError: 'output_size' should be an integer or list or tuple with length as 1.
    Examples:
        .. code-block:: python
              # average adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = sum(input[:, :, lstart: lend])/(lstart - lend)
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()
              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_average_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
    """
    pool_type = 'avg'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'adaptive_pool2d')
    _check_input(x, 3)
    check_type(output_size, 'pool_size', (int), 'adaptive_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = "pool2d"
    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.pool2d(x, 'pooling_type', pool_type, 'ksize',
                                   pool_size, 'adaptive', True)
        return squeeze(pool_out, [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}
    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return squeeze(pool_out, [2])


def adaptive_avg_pool2d(x, output_size, data_format='NCHW', name=None):
    """
    This API implements adaptive average pooling 2d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool2d` .

    Args:
        x (Tensor): The input tensor of adaptive avg pool2d operator, which is a 4-D tensor.
                          The data type can be float32 or float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain two element, (H, W). H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCHW", "NHWC". The default is "NCHW". When it is "NCHW", the data is stored in
            the order of: [batch_size, input_channels, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of avg adaptive pool2d result. The data type is same as input tensor.
    Raises:
        ValueError: If `data_format` is not "NCHW" or "NHWC".
    Examples:
        .. code-block:: python
            # adaptive avg pool2d
            # suppose input data in shape of [N, C, H, W], `output_size` is [m, n],
            # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
            # of input data into m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(m):
            #         for j in range(n):
            #             hstart = floor(i * H / m)
            #             hend = ceil((i + 1) * H / m)
            #             wstart = floor(i * W / n)
            #             wend = ceil((i + 1) * W / n)
            #             output[:, :, i, j] = avg(input[:, :, hstart: hend, wstart: wend])
            #
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 32, 32]
            out = paddle.nn.functional.adaptive_avg_pool2d(
                            x = x,
                            output_size=[3, 3])
            # out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_avg_pool2d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool2d')

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCHW' or 'NHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCHW":
        in_h, in_w = x.shape[2:4]
    else:
        in_h, in_w = x.shape[1:3]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        output = core.ops.pool2d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool2d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_avg_pool3d(x, output_size, data_format='NCDHW', name=None):
    """
    This API implements adaptive average pooling 3d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveAvgPool3d` .

    Args:
        x (Tensor): The input tensor of adaptive avg pool3d operator, which is a 5-D tensor.
                          The data type can be float32, float64.
        output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list,
            it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means
            the size will be the same as that of the input.
        data_format (str): The data format of the input and output data. An optional string
            from: "NCDHW", "NDHWC". The default is "NCDHW". When it is "NCDHW", the data is stored in
            the order of: [batch_size, input_channels, input_depth, input_height, input_width].
        name(str, optional): For detailed information, please refer
                             to :ref:`api_guide_Name`. Usually name is no need to set and
                             None by default.
    Returns:
        Tensor: The output tensor of avg adaptive pool3d result. The data type is same as input tensor.
    Raises:
        ValueError: If `data_format` is not "NCDHW" or "NDHWC".
    Examples:
        .. code-block:: python
            # adaptive avg pool3d
            # suppose input data in shape of [N, C, D, H, W], `output_size` is [l, m, n],
            # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
            # of input data into l * m * n grids averagely and performs poolings in each
            # grid to get output.
            # adaptive avg pool performs calculations as follow:
            #
            #     for i in range(l):
            #         for j in range(m):
            #             for k in range(n):
            #                 dstart = floor(i * D / l)
            #                 dend = ceil((i + 1) * D / l)
            #                 hstart = floor(j * H / m)
            #                 hend = ceil((j + 1) * H / m)
            #                 wstart = floor(k * W / n)
            #                 wend = ceil((k + 1) * W / n)
            #                 output[:, :, i, j, k] =
            #                     avg(input[:, :, dstart:dend, hstart: hend, wstart: wend])
            import paddle
            import numpy as np
            paddle.disable_static()
            input_data = np.random.rand(2, 3, 8, 32, 32)
            x = paddle.to_tensor(input_data)
            # x.shape is [2, 3, 8, 32, 32]
            out = paddle.nn.functional.adaptive_avg_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
            # out.shape is [2, 3, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_avg_pool3d')
    check_type(data_format, 'data_format', str, 'adaptive_avg_pool3d')

    if data_format not in ["NCDHW", "NDHWC"]:
        raise ValueError(
            "Attr(data_format) should be 'NCDHW' or 'NDHWC'. Received "
            "Attr(data_format): %s." % str(data_format))

    if data_format == "NCDHW":
        in_l, in_h, in_w = x.shape[2:5]
    else:
        in_l, in_h, in_w = x.shape[1:4]

    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        output = core.ops.pool3d(x, 'pooling_type', 'avg', 'ksize', output_size,
                                 'global_pooling', False, 'adaptive', True,
                                 'data_format', data_format)
        return output

    l_type = 'pool3d'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": "avg",
            "ksize": output_size,
            "adaptive": True,
            "data_format": data_format,
        })

    return pool_out


def adaptive_max_pool1d(x, output_size, return_indices=False, name=None):
    """
    This API implements adaptive max pooling 1d operation.
    See more details in :ref:`api_nn_pooling_AdaptiveMaxPool1d` .

    Args:
        x (Tensor): The input tensor of pooling operator, which is a 3-D tensor
                              with shape [N, C, L].  The format of input tensor is NCL,
                              where N is batch size, C is the number of channels, L is the
                              length of the feature. The data type is float32 or float64.
        output_size (int): The pool kernel size. The value should be an integer.
        return_indices (bool): If true, the index of max pooling point will be returned along
                with outputs. It cannot be set in average pooling type. Default False.
        name(str, optional): For detailed information, please refer
                                 to :ref:`api_guide_Name`. Usually name is no need to set and
                                 None by default.
    Returns:
            Tensor: The output tensor of adaptive pooling result. The data type is same
                      as input tensor.
    Raises:
            ValueError: 'output_size' should be an integer.
    Examples:
        .. code-block:: python

              # max adaptive pool1d
              # suppose input data in shape of [N, C, L], `output_size` is m or [m],
              # output shape is [N, C, m], adaptive pool divide L dimension
              # of input data into m grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         lstart = floor(i * L / m)
              #         lend = ceil((i + 1) * L / m)
              #         output[:, :, i] = max(input[:, :, lstart: lend])
              #
              import paddle
              import paddle.nn.functional as F
              paddle.disable_static()
              data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
              pool_out = F.adaptive_max_pool1d(data, output_size=16)
              # pool_out shape: [1, 3, 16])
              pool_out, indices = F.adaptive_max_pool1d(data, output_size=16, return_indices=True)
              # pool_out shape: [1, 3, 16] indices  shape: [1, 3, 16]
    """
    pool_type = 'max'
    check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                             'adaptive_max_pool1d')
    _check_input(x, 3)
    check_type(output_size, 'pool_size', int, 'adaptive_max_pool1d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool1d')

    pool_size = [1] + utils.convert_to_list(output_size, 1, 'pool_size')

    l_type = 'max_pool2d_with_index'

    x = unsqueeze(x, [2])
    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'pooling_type', pool_type, 'ksize', pool_size, 'adaptive', True)
        return (squeeze(pool_out[0], [2]), squeeze(
            pool_out[1], [2])) if return_indices else squeeze(pool_out[0], [2])

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "adaptive": True,
        })

    return (squeeze(pool_out, [2]),
            squeeze(mask, [2])) if return_indices else squeeze(pool_out, [2])


def adaptive_max_pool2d(x, output_size, return_indices=False, name=None):
    """
        This operation applies a 2D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool2d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool2d operator, which is a 4-D tensor. The data type can be float16, float32, float64, int32 or int64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain two elements, (H, W). H and W can be either a int, or None which means the size will be the same as that of the input.
            return_indices (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool2d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # max adaptive pool2d
              # suppose input data in the shape of [N, C, H, W], `output_size` is [m, n]
              # output shape is [N, C, m, n], adaptive pool divide H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(m):
              #         for j in range(n):
              #             hstart = floor(i * H / m)
              #             hend = ceil((i + 1) * H / m)
              #             wstart = floor(i * W / n)
              #             wend = ceil((i + 1) * W / n)
              #             output[:, :, i, j] = max(input[:, :, hstart: hend, wstart: wend])
              #
              import paddle
              import numpy as np
              paddle.disable_static()
              input_data = np.random.rand(2, 3, 32, 32)
              x = paddle.to_tensor(input_data)
              # x.shape is [2, 3, 32, 32]
              out = paddle.nn.functional.adaptive_max_pool2d(
                            x = x,
                            output_size=[3, 3])
              # out.shape is [2, 3, 3, 3]
    """
    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_max_pool2d')
    _check_input(x, 4)
    #check_type(output_size, 'pool_size', (int), 'adaptive_max_pool2d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool2d')

    in_h, in_w = x.shape[2:4]
    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 2, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_h
        if output_size[1] == None:
            output_size[1] = in_w

    if in_dygraph_mode():
        pool_out = core.ops.max_pool2d_with_index(
            x, 'pooling_type', 'max', 'ksize', output_size, 'adaptive', True)
        return pool_out if return_indices else pool_out[0]

    l_type = 'max_pool2d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })
    #return (pool_out, mask) if return_indices else pool_out
    return pool_out


def adaptive_max_pool3d(x, output_size, return_indices=False, name=None):
    """
        This operation applies a 3D adaptive max pooling on input tensor.
        See more details in :ref:`api_nn_pooling_AdaptiveMaxPool3d` .

        Args:
            x (Tensor): The input tensor of adaptive max pool3d operator, which is a 5-D tensor. The data type can be float32, float64.
            output_size (int|list|tuple): The pool kernel size. If pool kernel size is a tuple or list, it must contain three elements, (D, H, W). D, H and W can be either a int, or None which means the size will be the same as that of the input.
            return_indices (bool): If true, the index of max pooling point will be returned along with outputs. Default False.
            name(str, optional): For detailed information, please refer to :ref:`api_guide_Name`. Usually name is no need to set and None by default.

        Returns:
            Tensor: The output tensor of adaptive max pool3d result. The data type is same as input tensor.

        Examples:
            .. code-block:: python

              # adaptive max pool3d
              # suppose input data in the shape of [N, C, D, H, W], `output_size` is [l, m, n]
              # output shape is [N, C, l, m, n], adaptive pool divide D, H and W dimensions
              # of input data into m*n grids averagely and performs poolings in each
              # grid to get output.
              # adaptive max pool performs calculations as follow:
              #
              #     for i in range(l):
              #         for j in range(m):
              #             for k in range(n):
              #                 dstart = floor(i * D / l)
              #                 dend = ceil((i + 1) * D / l)
              #                 hstart = floor(i * H / m)
              #                 hend = ceil((i + 1) * H / m)
              #                 wstart = floor(i * W / n)
              #                 wend = ceil((i + 1) * W / n)
              #             output[:, :, i, j, k] = max(input[:, :, dstart: dend, hstart: hend, wstart: wend])
              #
              import paddle
              import numpy as np
              paddle.disable_static()
              input_data = np.random.rand(2, 3, 8, 32, 32)
              x = paddle.to_tensor(input_data)
              # x.shape is [2, 3, 8, 32, 32]
              out = paddle.nn.functional.adaptive_max_pool3d(
                            x = x,
                            output_size=[3, 3, 3])
              # out.shape is [2, 3, 3, 3, 3]
    """

    if not in_dygraph_mode():
        check_variable_and_dtype(x, 'x', ['float32', 'float64'],
                                 'adaptive_max_pool3d')
    _check_input(x, 5)
    #check_type(output_size, 'pool_size', (int), 'adaptive_max_pool3d')
    check_type(return_indices, 'return_indices', bool, 'adaptive_max_pool3d')

    in_l, in_h, in_w = x.shape[2:5]
    if isinstance(output_size, int):
        output_size = utils.convert_to_list(output_size, 3, 'output_size')
    else:
        output_size = list(output_size)
        if output_size[0] == None:
            output_size[0] = in_l
        if output_size[1] == None:
            output_size[1] = in_h
        if output_size[2] == None:
            output_size[2] = in_w

    if in_dygraph_mode():
        pool_out = core.ops.max_pool3d_with_index(
            x, 'pooling_type', 'max', 'ksize', output_size, 'adaptive', True)
        return pool_out if return_indices else pool_out[0]

    l_type = 'max_pool3d_with_index'

    helper = LayerHelper(l_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)

    mask = helper.create_variable_for_type_inference(dtype)
    outputs = {"Out": pool_out, "Mask": mask}

    helper.append_op(
        type=l_type,
        inputs={"X": x},
        outputs=outputs,
        attrs={
            "pooling_type": 'max',
            "ksize": output_size,
            "adaptive": True,
        })

    return (pool_out, mask) if return_indices else pool_out
