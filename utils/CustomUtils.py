
def total_params(model):
    total_params = sum(p.numel() for p in model.parameters())  # 그냥 묶어서 사용하자.
    return total_params


def make_file_list(path, size):
    r"""
    path -> file list with directory name
    size : size to include (오름차순 정렬된 상태 기준)
    """
    import os
    import natsort

    files: list = os.listdir(path)
    sorted_files = natsort.natsorted(files)

    if size == 0:
        return [path + '/' + file for file in sorted_files]

    else:
        return [path + '/' + file for file in sorted_files[:size]]


def pair_two_lists(list1, list2):
    r"""
    list1 = [x1, x2, x3], list2 = [y1, y2, y3]
    returns [(x1, y1), (x2, y2), (x3, y3)]
    """
    assert len(list1) == len(list2), 'list1 and list2 sizes are not same'

    return [(list1[i], list2[i]) for i in range(len(list1))]


def split_pairs(pairs):
    assert len(pairs[0]) == 2, 'pairs must be combined by two objects'

    result1 = []
    result2 = []

    for elem1, elem2 in pairs:
        result1.append(elem1)
        result2.append(elem2)

    return result1, result2


def random_split_by_ratio(pairs, train_test_ratio: list):
    r"""
    random_split_by_ratio(pairs : torch.Dataset, train_test_ratio = [train, test])
    ex) random_split_by_ratio(pairs, [0.8, 0.2]) # sum must be 1

    return : (train_images, test_images)
    """
    from torch.utils.data import random_split

    assert len(train_test_ratio) == 2, 'ratio list must be with two elements'
    assert train_test_ratio[0] + train_test_ratio[1] == 1, 'sum of ratio must be 1'

    train_size = int(len(pairs) * train_ratio)
    val_size = len(pairs) - train_size

    print(f"Train set size is {train_size}, Validation set size is {val_size}")

    return random_split(pairs, [train_size, val_size])


def normalize_ndarray_256(arr):
    min = arr.min()
    range = arr.max() - min

    if range > 0:
        normed_arr = 255 * (arr - min) / range
    else:
        normed_arr = arr

    return normed_arr