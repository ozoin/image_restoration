import numpy as np
from itertools import chain


def MaskedImage(image_array: np.ndarray, offset: tuple, spacing: tuple):
    input_array = []
    known_array = []
    target_array = []

    if type(image_array) != np.ndarray:
        raise TypeError('Wrong array type')
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise NotImplementedError('Wrong dimentions')
    for elem in spacing:
        try:
            int(elem)
        except:
            raise ValueError('Not convertible')
        if not 2 <= elem <= 8:
            raise ValueError('Dimensions are not in [2,8]')
    for elem in offset:
        try:
            int(elem)
        except:
            raise ValueError('Not convertible')
        if not 0 <= elem <= 32:
            raise ValueError('Dimensions are not in [0,32]')

    image_array = image_array.copy()
    image_array = np.transpose(image_array, (2, 0, 1))

    known_array = np.zeros(
        (image_array.shape[1], image_array.shape[2]), dtype=image_array.dtype)

    for row_idx in range(offset[0], len(known_array[0, :]), spacing[0]):
        impute_idcs = np.arange(offset[1], len(
            known_array[:, row_idx]), spacing[1])
        known_array[impute_idcs, row_idx] = 1

    unknown_row, unknown_col = np.where(known_array == 0)
    known_array = np.repeat(np.expand_dims(known_array, 0), 3, 0)

    target_array = [channel_value for channel_value in chain(
        image_array[0, unknown_row, unknown_col], image_array[1, unknown_row, unknown_col], image_array[2, unknown_row, unknown_col])]

    input_array = image_array * known_array

    known_pixels = len(np.where(known_array[0] == 1)[0])

    if known_pixels < 144:
        raise ValueError(
            f'The number of known pixels after removing must be at least 144 but is {known_pixels}')

    return (input_array, known_array, np.asarray(target_array, dtype=image_array.dtype))
