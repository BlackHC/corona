import contextlib


@contextlib.contextmanager
def copy_view(arr, inds):
    # create copy from fancy inds
    arr_copy = arr[inds]

    yield arr_copy

    # after context, save modified data
    arr[inds] = arr_copy