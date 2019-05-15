import scipy.io as sio
import numpy as np
import seaborn as sns
import os


def extract_mat_content(path, header):
    """ Extract the .mat file content """

    content = sio.loadmat(path)

    return content[header]


def display_energy_policy(q_mat, avg_energy, svisit=True):
    """ Display final energy state with policy. Ideally, the final policy
        should lead the agent to move from high to low. Otherwise, display
        FALSE to indicate wrong action.
    """

    policy = 'policy.txt'
    state_size = 20
    avgen_arr = np.reshape(avg_energy, (1, state_size))[0]
    state_action = []
    for i in range(state_size):
        dst = np.argmax(q_mat[i, :])
        if svisit is False and i == dst:
            dst = np.argpartition(q_mat[i, :], -2)[-2:]

        if avgen_arr[i] >= avgen_arr[dst]:
            sign = 'True'
        else:
            sign = 'False'

        state_action.append((i, dst, sign))
        log = "echo " + "state: " + str(i) + " next state " + str(dst) + " High2Low: " + sign + " >> " + policy
        os.system(log)

    return state_action


def display_high2low_policy(st_action):
    """ Display state action pair as heatmap and check whether the agent follow high2low policy. """
    state_mgs, state_labels, l2h_index, scene_size = [], np.zeros(20), 1, (4, 5)
    for i in range(len(st_action)):
        sa = str(st_action[i][0]) + ' --> ' + str(st_action[i][1])
        if st_action[i][2] == 'False':
            h2l_msg = '\n LOW to HIGH'
            state_labels[i] = l2h_index
        else:
            h2l_msg = '\n HIGH to LOW'
        state_mgs.append(sa + h2l_msg)

    labels = np.reshape(np.asarray(state_mgs), scene_size)
    sns.plt.figure("Policy")
    sns.heatmap(np.reshape(state_labels, scene_size), annot=labels, fmt='', linewidths=0.7, cbar=False)
    sns.plt.title("State-action pairs labels", fontweight='bold')

    axis = sns.plt.gca()
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    sns.plt.savefig("agent_policy.png")
    sns.plt.show(block=True)


def display_plots(results_mat):
    """ Display subplots which contains experiment results (in full screen). """

    sns.plt.figure("Agent Plots")
    sns.plt.subplot(231)
    sns.heatmap(results_mat[1], cmap="YlGnBu")
    sns.plt.title("Q Matrix Heatmap", fontweight='bold')
    sns.plt.ylabel("States")
    sns.plt.xlabel("Next states")

    sns.plt.subplot(232)
    sns.heatmap(results_mat[3], annot=True, fmt="d", cmap="YlGnBu")
    sns.plt.title("s,a pairs visit Heatmap", fontweight='bold')
    sns.plt.ylabel("States")
    sns.plt.xlabel("Next states")

    sns.plt.subplot(233)
    # instead of color put number of visit
    sns.heatmap(np.reshape(results_mat[5], (4, 5)), annot=True, fmt="d", cmap="YlGnBu", linewidths=1)
    sns.plt.title("s,a pairs visit Heatmap", fontweight='bold')

    axis = sns.plt.gca()
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)

    sns.plt.subplot(234)
    # instead of color put number of visits
    sns.heatmap(np.reshape(results_mat[5], (4, 5)), annot=True, fmt="d", cmap="YlGnBu", linewidths=1)
    sns.plt.title("Total energy per state", fontweight='bold')

    axis = sns.plt.gca()
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)

    sns.plt.subplot(235)
    sns.heatmap(results_mat[4], annot=True, fmt="d", cmap="YlGnBu", linewidths=1)
    sns.plt.title("Average energy per state", fontweight='bold')

    axis = sns.plt.gca()
    axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)

    sns.plt.subplot(236)
    sns.plt.plot(results_mat[2])
    sns.plt.title("Cumulative reward", fontweight='bold')
    sns.plt.xlabel("Iteration step")
    sns.plt.ylabel("Cumulative reward")

    sns.plt.subplots_adjust(hspace=.3)

    manager = sns.plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    sns.plt.savefig("results_full_size.png", figsize=(1640, 860))

    sns.plt.show(block=False)


def construct_avg_reward(path, iterations):
    """ Average cumulative discounted reward """

    num_of_exps = 10
    sum = np.zeros((1, iterations), dtype=np.int)
    rpath = path+str(iterations)+'/'

    for i in range(num_of_exps):
        rew = extract_mat_content(rpath + 'sreward' + str(i + 1) + '.mat', 'sreward')
        sum = np.add(sum, rew[0])

    return sum[0] * 0.1


def display_iteration_plots(iteration):
    """ Display the cumulative reward plots for different iterations."""

    sns.plt.plot(iteration[0], label='200 iterations')
    sns.plt.hold(True)

    sns.plt.plot(iteration[1], label='300 iterations')
    sns.plt.hold(True)

    sns.plt.plot(iteration[2], label='400 iterations')
    sns.plt.hold(True)

    sns.plt.plot(iteration[3], label='500 iterations')
    sns.plt.hold(True)

    sns.plt.plot(iteration[4], label='600 iterations')
    sns.plt.hold(True)

    sns.plt.plot(iteration[5], label='1000 iterations')
    sns.plt.hold(True)

    sns.plt.legend(loc='upper left', shadow=True)

    sns.plt.ylabel("Average cumulative rewards", fontweight='bold')
    sns.plt.xlabel("Iteration steps", fontweight='bold')
    sns.plt.ylim(-10, 500)
    sns.plt.show(block=True)


def running_average(x, window_size, mode='valid'):
    """ Adopted from openAI gym """
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


def construct_avg_td_error(tdpath, iterations, window_size):
    """ Get the average Temporal Difference errors for 10 runs."""
    num_of_exps = 10
    sum = np.zeros((1, iterations), dtype=np.int)
    for i in range(num_of_exps):
        td = extract_mat_content(tdpath + str(iterations) + '/qconv' + str(i + 1) + '.mat', 'concv')
        sum = np.add(sum, td[0])

    avg_td = sum[0] * 0.1
    return running_average(avg_td, window_size)


def display_td_error_plots(rtd):
    """ Display average TD errors for different runs. rtd --> run td error. """

    lw, window_size = 3.0, 100

    sns.plt.plot(np.arange(100, 201, 1).reshape(rtd[0].shape), rtd[0], label='200 iterations', linewidth=lw)
    sns.plt.hold(True)

    sns.plt.plot(np.arange(100, 301, 1).reshape(rtd[1].shape), rtd[1], label='300 iterations', linewidth=lw)
    sns.plt.hold(True)
    #
    sns.plt.plot(np.arange(100, 401, 1).reshape(rtd[2].shape),rtd[2], label='400 iterations', linewidth=lw)
    sns.plt.hold(True)
    #
    sns.plt.plot(np.arange(100, 501, 1).reshape(rtd[3].shape), rtd[3], label='500 iterations', linewidth=lw)
    sns.plt.hold(True)
    #
    sns.plt.plot(np.arange(100, 601, 1).reshape(rtd[4].shape), rtd[4], label='600 iterations', linewidth=lw)
    sns.plt.hold(True)
    #
    sns.plt.plot(np.arange(100, 1001, 1).reshape(rtd[5].shape), rtd[5], label='1000 iterations', linewidth=lw)
    sns.plt.hold(True)

    sns.plt.legend(loc='upper right', shadow=True)
    sns.plt.xlabel("Iteration steps", fontweight='bold')
    sns.plt.ylabel("Temporal Difference Error ( window size: "+str(window_size)+" )", fontweight='bold')
    sns.plt.ylim(0.0, 0.2)

    sns.plt.show(block=True)