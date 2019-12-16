import numpy as np
import os


def load_positions(directory: str, split_left: str = "", split_right: str = ""):
    """
    File names have the following format: 2019-05-13T15-04-01.648017-4c_30ms_0-DIV.txt
    """
    x_positions = []  # empty list to save positions

    files = os.listdir(directory)
    files.sort()
    files = sorted(files)
    if split_left != "":
        split_left = split_left
    else:
        split_left = "knife_"
    if split_right != "":
        split_right = split_right
    else:
        split_right = "div"

    for file in files:
        if file.endswith('.txt'):
            x_position = file.split(split_left)[1].split(split_right)[0]
            x_positions.append(int(x_position))

    return x_positions


def extract_data(signal: np.array, range: tuple, option: int):
    signal = signal - np.mean(signal)  # remove offset
    signal = signal[range[0]:range[1]]  # choose one pulse
    if option == 0:
        return np.max(signal)  # electric field strength
    if option == 1:
        return np.max(signal) - np.min(signal)  # electric field strength
    if option == 2:
        return (np.max(signal) - np.min(signal)) ** 2  # intensity
    if option == 3:
        return np.trapz(signal ** 2)  # intensity


def determine_y(directory: str, range: tuple, option: int):
    ys = []  # amplitude or intensity of the signal

    os.chdir(directory)  # change directory
    files = os.listdir('.')
    files.sort()
    files = sorted(files)

    for file in files:
        if file.endswith('.txt'):
            amplitude_data = np.loadtxt(file)[:, 1]
            y = extract_data(amplitude_data, range, option)
            ys.append(y)

    return ys


def derivative(x, y):
    deriv = []
    for i in range(1, len(y) - 1):
        deriv.append(1 / 2 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) + (y[i] - y[i - 1]) / (x[i] - x[i - 1])))
    return deriv


def gaus(x,a,x0,sigma):
    return a * np.exp(-2*(x-x0)**2/(sigma**2))
