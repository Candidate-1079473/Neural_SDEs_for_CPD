"""Operations on Arrays"""
import torch


def moving_average(array, window_size):
    """Compute the moving average of an array"""
    assert window_size % 2 == 1

    array_length = len(array)
    true_start = window_size // 2
    start = array[0] * torch.ones((true_start,), device=array.device)
    end = array[-1] * torch.ones((true_start,), device=array.device)

    array = torch.cat([start, array, end])

    summed_up = sum(array[i : i + array_length] for i in range(window_size))
    averaged = summed_up / window_size
    return averaged