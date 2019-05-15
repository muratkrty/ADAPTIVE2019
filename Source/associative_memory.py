import cv2
import random as rd
import numpy as np

# to be compatible among different opencv versions
cv2.CV_LOAD_IMAGE_GRAYSCALE = 0

def name_pattern(path, ind):
    """ Name a pattern for scene construction."""
    return path + str(ind) + '.png'


def bipolarize(num):
    """ Bipolarize a given pixel value. """
    return 2 * (num - 0.5)


def format_patterns(pattern, inp_height):
    """ Display patterns in according to input height.
        (Adapted and modified from NuPIC repo)
    """
    s = ''
    for c in range(len(pattern)):
        if c > 0 and c % inp_height == 0:
            s += ' \n'
        s += str(pattern[c])
    s += ' '

    return s


def convert_pattern_to_image(pattern, size):
    """ Convert converged pattern from numpy array form to pixel matrix."""
    width, height = size[0], size[1]
    nonz = pattern.nonzero()
    pattern[nonz] = 255

    return np.reshape(pattern, (width, height))


def binarize_pattern(pattern, resize_wh):
    """ Original pixel mats use 255 for 1s. This function converts 255 to 0.
        This will be used for a single image/pattern.
    """
    gray_img = cv2.imread(pattern, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    resize_img = cv2.resize(gray_img, resize_wh)
    bin_img = cv2.threshold(resize_img, 20, 255, cv2.THRESH_BINARY)[1]

    nonz = bin_img.nonzero()
    bin_img[nonz] = 1

    return bin_img.flatten()


def perform_bin_image_processing(patterns, resize_wh):
    """ Generate binarized vectors of training patterns."""
    width, height = resize_wh[0], resize_wh[1]
    binaries = np.zeros((len(patterns), width * height), dtype='int64')

    for i in range(len(patterns)):
        gray_img = cv2.imread(patterns[i], cv2.CV_LOAD_IMAGE_GRAYSCALE)
        resize_img = cv2.resize(gray_img, resize_wh)
        bin_img = cv2.threshold(resize_img, 20, 255, cv2.THRESH_BINARY)[1]

        temp_bin = np.zeros((width, height), dtype='int64')
        temp_bin[:] = bin_img[:]
        nonz = temp_bin.nonzero()
        temp_bin[nonz] = 1
        binaries[i, :] = temp_bin.flatten()

    return binaries


def construct_weight_mat(binaries, num_of_neurons):
    """ Extract individual weight matrix for each trained pattern. Then
        sum them to construct HN's weight matrix. Note that main diagonal
        must consists of 0s.
    """
    tot_weights = np.zeros((num_of_neurons, num_of_neurons), dtype='int64')
    pattern_weight = np.zeros((num_of_neurons, num_of_neurons), dtype='int64')

    for p in range(len(binaries)):
        for i in range(num_of_neurons):
            for j in range(num_of_neurons):
                if i == j:
                    pattern_weight[i, i] = 0
                else:
                    tmp = bipolarize(binaries[p, i]) * bipolarize(binaries[p, j])
                    pattern_weight[i, j] = tmp
                    pattern_weight[j, i] = tmp
        tot_weights = tot_weights + pattern_weight
        pattern_weight.fill(0)

    return tot_weights


def get_conrate(test, original, num_of_neurons):
    """ Extract contamination rate for converged pattern. The high rate for convergence
        rate indicates how a pattern successfully recalled by HN dynamics. conrate --> convergence rate
    """

    return np.sum(test == original) * 100 / num_of_neurons


def execute_hopfield(testid, weight_mat):
    """Run Hopfield Network dynamics to obtain necessary network information.
       Convergence rate, num_of_flipped_bit, etc.
    """
    nrow, ncols = 20, 20
    size = (nrow, ncols)
    num_of_neurons = nrow * ncols

    pspath = 'scene/patterns/'  # the patterns for creating scene
    testp = name_pattern(pspath, testid)

    test_bin = binarize_pattern(testp, size)
    prev_test = np.zeros((1, num_of_neurons), dtype='int64')
    np.copyto(prev_test, test_bin)

    state_counter, unchanged_states, max_iteration = 0, 1000, 10000
    iteration, flipped_bits = 0, 0

    while state_counter < unchanged_states and iteration < max_iteration:

        rand_ind = rd.randint(0, num_of_neurons - 1)
        sum_units = 0

        for i in range(num_of_neurons):
            if rand_ind != i:
                sum_units += test_bin[i] * weight_mat[i, rand_ind]

        if sum_units > 0:
            test_bin[rand_ind] = 1
        else:
            test_bin[rand_ind] = 0

        # log number of chaged bits
        if test_bin[rand_ind] != prev_test[0, rand_ind]:
            flipped_bits += 1

        # observe network state
        state_flag = False

        for i in range(len(test_bin)):
            if test_bin[i] != prev_test[0, i]:
                state_flag = True
                break
        if state_flag:
            state_counter += 1
        else:
            state_counter = 0
        iteration += 1
    # return (test_bin, flipped_bits, state_counter)
    # hot fix for ploting
    return test_bin, flipped_bits, state_counter


def save_convscene(test_bin, testid, cspath, size, original_size):
    """ Save converged patterns """
    cimage = convert_pattern_to_image(test_bin, size)
    iname = cspath + str(testid) + '.png'

    rimage = cv2.resize(cimage, original_size)
    cv2.imwrite(iname, rimage)


def extract_reward(srew, arew):
    """ Extract reward value as a function of energy """
    if srew >= arew:
        reward = 1
    else:
        reward = -1
    return reward