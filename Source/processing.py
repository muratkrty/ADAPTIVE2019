import os
import cv2
import numpy as np
import re
import scipy


def save_result_mats(results_mat):
    """ Save extracted result mats for experiments. """
    scipy.io.savemat('toten.mat', mdict={'sc_arr_energy': results_mat[0]})
    scipy.io.savemat('q.mat', mdict={'q_mat': results_mat[1]})
    scipy.io.savemat('sreward.mat', mdict={'sreward': results_mat[2]})
    scipy.io.savemat('visits.mat', mdict={'visits': results_mat[3]})
    scipy.io.savemat('avgen.mat', mdict={'avg_energy': results_mat[4]})
    scipy.io.savemat('stcounts.mat', mdict={'sc_arr_cntr': results_mat[5]})



def move_result_patterns(icounter):
    """ Move result patterns to a folder with same name """
    resdir = 'mkdir res/' + str(icounter)
    os.system(resdir)
    mvres = 'mv *png *mat *txt ' + 'res/' + str(icounter)
    os.system(mvres)


def display_policy_sa(icounter, q_mat):
    """ Display final policy for self visiting allowed experiments"""
    policy = 'policy_' + str(icounter) + '.txt'
    state_size = 20
    for i in range(state_size):
        tmp = np.argmax(q_mat[i, :])
        log = "echo " + "state: " + str(i) + " next state " + str(tmp) + " >> " + policy
        os.system(log)


def display_policy_sp(icounter, q_mat):
    """ Display final policy for self visiting prohibited experiments"""
    policy = 'policy_' + str(icounter) + '.txt'
    for i in range(20):
        tmp = np.argmax(q_mat[i, :])
        if i == tmp:
            max2ind = np.argpartition(q_mat[i, :], -2)[-2:]
            log = "echo " + "state: " + str(i) + " next state " + str(max2ind) + " >> " + policy
        else:
            log = "echo " + "state: " + str(i) + " next state " + str(tmp) + " >> " + policy
        os.system(log)


def collect_scene_patterns():
    """ Collect both scene patterns and training patterns """
    cmd = 'cp -r training/* scene/patterns/'
    os.system(cmd)


def contaminate_pattern(pattern, rate, ind):
    """ Contaminate colored pattern acc. to given rate """
    cont_path = 'scene/patterns/'
    im_name = cont_path + str(ind) + '.png'

    img = cv2.imread(pattern)
    simg = img.size / 3  # work on 2D
    c_rate = int(rate * simg)

    row, col = img.shape[0], img.shape[1]

    for i in range(c_rate):
        rw = np.random.randint(row)
        cl = np.random.randint(col)
        if img[rw, cl][0] != 0:
            img[rw, cl][0] = 0
        else:
            img[rw, cl][0] = 255

    cv2.imwrite(im_name, img)


def create_scene_patterns(tpatterns, noise1, noise2):
    """ Create contaminated patterns from training patterns
        and save them acc. to given index.
    """
    for i in range(len(tpatterns)):
        contaminate_pattern(tpatterns[i], noise1, i + 10)
        contaminate_pattern(tpatterns[i], noise2, i + 15)


def display_pattern(ppath):
    """ Read pattern path and display it till ESC pressed """
    img = cv2.imread(ppath)
    print img.size

    cv2.imshow(ppath, img)
    cv2.waitKey(0)


def create_scene(sfiles, pspath):
    """Combine scene patterns and create scene in random fashion."""
    srow, scol = 4, 5

    # perm = np.random.permutation(len(sfiles)).reshape((4, 5))
    perm = np.asarray([[0, 1, 12, 17, 15],
                       [10, 16, 6, 7, 4],
                       [14, 3, 11, 2, 5],
                       [19, 18, 8, 13, 9]])
    tmps = []
    # TODO: modularize it!
    for i in range(srow):
        img = cv2.imread(pspath + str(perm[i][0]) + '.png')
        img2 = cv2.imread(pspath + str(perm[i][1]) + '.png')
        img3 = cv2.imread(pspath + str(perm[i][2]) + '.png')
        img4 = cv2.imread(pspath + str(perm[i][3]) + '.png')
        img5 = cv2.imread(pspath + str(perm[i][4]) + '.png')

        conc = np.concatenate((img, img2, img3, img4, img5), axis=1)
        tmps.append(conc)
        conc = []

    scene = tmps[0]
    for i in range(1, srow):
        scene = np.concatenate((scene, tmps[i]), axis=0)
    cv2.imwrite("scene.png", scene)

    return perm


def create_file_names(data_path):
    """ Walk trough DATA directory and create a list
        that contains file names.
    """
    inputs = []
    for dir_name, sub_dir, file_names in os.walk(data_path):
        file_names.sort()
        for f_name in file_names:
            f_path = os.path.join(dir_name, f_name)
            inputs.append(f_path)
    return inputs


def create_bordered_imgs(patterns, rpatterns):
    """ Create red-bordered version of scene patterns
        Later, they will be used for creating switching scene
    """

    sps = create_file_names(patterns)
    h, w = 200, 200  # rimg.shape[0], rimg.shape[1]

    for i in range(len(sps)):
        img = sps[i]
        tmp_img = img[27:len(img)]
        rimg = cv2.imread(img)
        bimg = rimg
        cv2.rectangle(bimg, (0, 0), (w, h), (0, 0, 255), 15)

        cv2.imwrite(rpatterns + tmp_img, bimg)


def create_state_folders(scpatterns, rpatterns):
    """ cp one red bordered state pattern with
        19 original pattern
    """
    sps = create_file_names(scpatterns)
    rsps = create_file_names(rpatterns)

    for k in range(20):
        cmkdir = 'mkdir scene/state_scene/states/s' + str(k)
        os.system(cmkdir)

    for i in range(20):
        rex = re.findall(r'\d+', rsps[i])
        # print rex[0]
        folder = ' scene/state_scene/states/' + 's' + str(rex[0]) + '/'
        os.system(folder)
        cmd = 'cp ' + rsps[i] + folder
        os.system(cmd)

        for ii in range(20):
            if i != ii:
                cmd = 'cp ' + sps[ii] + folder
                os.system(cmd)


def create_final_states_folders(ftpatterns):
    """ Use previously created folder then draw green rectangle around them"""
    data_path = ftpatterns
    h, w = 200, 200
    for dir_name, sub_dir, file_names in os.walk(data_path):
        file_names.sort()
        for subi in range(len(sub_dir)):
            dname = sub_dir[subi]
            rex = re.findall(r'\d+', dname)
            img = data_path + dname + '/' + rex[0] + '.png'
            rimg = cv2.imread(img)
            cv2.rectangle(rimg, (0, 0), (w, h), (0, 255, 0), 15)
            cv2.imwrite(img, rimg)


def create_scene_switch(stpatterns, cstpatterns, perm):
    """ Combine green/red-bordered and unbordered patterns """

    data_path = stpatterns
    for dir_name, sub_dir, file_names in os.walk(data_path):
        file_names.sort()
        for subi in range(len(sub_dir)):
            fname = sub_dir[subi]
            com_path = stpatterns + sub_dir[subi] + "/"
            tmp_files = create_file_names(com_path)
            combscene = create_scene(com_path, perm)
            cv2.imwrite(cstpatterns + sub_dir[subi] + '.png', combscene)
