import warnings
import operator
import numpy
from collections.abc import Iterable

def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None):
    ft_inplace = isinstance(indices, numpy.ndarray)
    dt_inplace = isinstance(distances, numpy.ndarray)
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    # calculate the feature transform
    input = numpy.atleast_1d(numpy.where(input, 1, 0).astype(numpy.int8))
    if sampling is not None:
        sampling = _normalize_sequence(sampling, input.ndim)
        sampling = numpy.asarray(sampling, dtype=numpy.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != numpy.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = numpy.zeros((input.ndim,) + input.shape, dtype=numpy.int32)

    # _nd_image.euclidean_feature_transform(input, sampling, ft)
    # if requested, calculate the distance transform
    if return_distances:
        dt = ft - numpy.indices(input.shape, dtype=ft.dtype)
        dt = dt.astype(numpy.float64)
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        numpy.multiply(dt, dt, dt)
        if dt_inplace:
            dt = numpy.add.reduce(dt, axis=0)
            if distances.shape != dt.shape:
                raise RuntimeError('distances array has wrong shape')
            if distances.dtype.type != numpy.float64:
                raise RuntimeError('distances array must be float64')
            numpy.sqrt(dt, distances)
        else:
            dt = numpy.add.reduce(dt, axis=0)
            dt = numpy.sqrt(dt)

    # construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None


def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            'at least one of return_distances/return_indices must be True')
    if distances_out and not return_distances:
        error_msgs.append(
            'return_distances must be True if distances is supplied'
        )
    if indices_out and not return_indices:
        error_msgs.append('return_indices must be True if indices is supplied')
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))


def _normalize_sequence(input, rank):
    """If input is a scalar, create a sequence of length equal to the
    rank by duplicating the input. If input is a sequence,
    check if its length is equal to the length of array.
    """
    is_str = isinstance(input, str)
    if not is_str and isinstance(input, Iterable):
        normalized = list(input)
        if len(normalized) != rank:
            err = "sequence argument must have length equal to input rank"
            raise RuntimeError(err)
    else:
        normalized = [input] * rank
    return normalized