import numpy as np
from numpy.polynomial import polynomial
from numpy.lib.stride_tricks import sliding_window_view


def moving_average(a, n=3):
    """
    Function that computes the moving average of an array following axis 0
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def fit_trend(forecast, mask, time_range, trend_period=12):
    """
    Function that computes trend on a time series of SSH values

    forecast: array (time, lon, lat)
    """
    lon = forecast.shape[1]
    lat = forecast.shape[2]

    x = list(time_range)  # the exact years in the year_range doesn't matter
    # the number of years does; as they will all give the same "x" calculated below
    mid_x = np.mean(x)
    x = np.asarray([i - mid_x for i in x])

    # to compute the trend, first step is to get an yearly (averaged) time series
    # from the original monthly series
    y = []
    for i in range(0, forecast.shape[0], trend_period):
        annual_pred = np.mean(forecast[i : i + trend_period, :, :], axis=0)
        y.append(annual_pred)
    y = np.stack(y)

    # fitted_coeff is a 2d array: it is the slope for the time series at every
    # grid point
    fitted_coeff = np.full((lon, lat), np.nan)
    for i in range(lon):
        for j in range(lat):
            y_i_j = y[:, i, j]
            mask_i_j = mask[:, i, j]
            if np.all(mask_i_j):
                fitted_all_coeffs = polynomial.polyfit(x, y_i_j, 1)
                # print(fitted_all_coeffs)
                fitted_coeff[i, j] = fitted_all_coeffs[1]

    return fitted_coeff


def sliding_moving_average(arr, window_size=3):
    """
    Calculate the sliding moving average of an array.

    Parameters:
    arr (numpy.ndarray): Input array, of shape (A,B,C,D....)
    window_size (int): Size of the sliding window. Default is 3.

    Returns:
    numpy.ndarray: Array containing the sliding moving average values.
    """
    window = sliding_window_view(arr, window_size, axis=0)
    return np.mean(window, axis=-1)



