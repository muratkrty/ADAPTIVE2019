import numpy as np
import associative_memory as hn
import processing as sc
import os
import visualization as vis
import scipy


def main():

    # folder paths for the visual assets
    tpath, pspath = 'training/', 'scene/patterns/'
    scpatterns, rpatterns = 'scene/state_scene/patterns/', 'scene/state_scene/rpatterns/'
    stpatterns, cstpatterns = 'scene/state_scene/states/', 'scene/state_scene/convstates/'
    ftpatterns, fcstpatterns = 'scene/state_scene/fstates/', 'scene/state_scene/finconvstates/'

    nu_states = 20
    sfiles, sc_pats = sc.create_file_names(pspath), sc.create_scene(sfiles, pspath)

    sc_arr = np.reshape(sc_pats, nu_states)
    sc_arr_cntr = np.zeros((1, nu_states), dtype=int)
    sc_arr_energy = np.zeros((1, nu_states), dtype=int)

    # create asset folders to facilitate the simulation and robot experiment
    os.system('rm -fr stpatterns')
    sc.create_bordered_imgs(scpatterns, rpatterns)
    sc.create_state_folders(scpatterns, rpatterns)
    sc.create_scene_switch(stpatterns, cstpatterns, sc_pats)

    cp_states = 'cp -r scene/state_scene/states/* scene/state_scene/fstates/'
    os.system(cp_states)

    sc.create_final_states_folders(ftpatterns)
    sc.create_scene_switch(ftpatterns, fcstpatterns, sc_pats)

    tfiles = sc.create_file_names(tpath)
    nrow, ncols = 20, 20
    size, neurons = (nrow, ncols), nrow * ncols

    binaries = hn.perform_bin_image_processing(tfiles, size)
    total_w = hn.construct_weight_mat(binaries, neurons)

    mu, gamma, epsilon, nits, nactions, nstates = 0.7, 0.4, 0.3, 0, 20, 20

    links = np.ones((nstates, nstates), dtype=int)
    q_mat = np.random.rand(nactions, nstates) * 0.1 - 0.5

    visits = np.zeros((nstates, nactions), dtype=int)

    # ocounter is not necessary but used for humanoids2016 paper
    ocounter, icounter = 1, 1000
    sreward, results_mats, qconv = [], [], []

    while nits < ocounter:

        s = np.random.randint(nstates)
        st_pat, eps = sc_arr[s], np.random.rand()  # epsilon threshold

        if eps < epsilon:
            # extract non-zero elements of links
            nonz = np.where(links[s, :] != 0)
            rindx = np.random.randint(np.shape(nonz)[1])
            a = nonz[0][rindx]
        else:
            a = np.argmax(q_mat[s, :])

        ac_pat = sc_arr[a]

        test_bins, sen, sctrs = hn.execute_hopfield(st_pat, total_w)
        test_bina, aen, sctra = hn.execute_hopfield(ac_pat, total_w)

        # log visiting count
        sc_arr_cntr[0][s] += 1
        sc_arr_cntr[0][a] += 1

        # log energy
        sc_arr_energy[0][s] += sen
        sc_arr_energy[0][a] += aen

        visits[s, a] += 1

        nits2, rew_stp = 0, 0

        while nits2 < icounter:
            rew = hn.extract_reward(sen, aen)
            rew_stp += rew
            sreward.append(rew_stp)

            sdot = a
            eps2 = np.random.rand()

            if eps2 < epsilon:
                nonz2 = np.where(links[sdot, :] != 0)
                rindx2 = np.random.randint(np.shape(nonz2)[1])
                adot = nonz2[0][rindx2]
            else:
                adot = np.argmax(q_mat[sdot, :])

            q_mat[s, a] += mu * (rew + gamma * q_mat[sdot, adot] - q_mat[s, a])
            qconv.append((rew + gamma * q_mat[sdot, adot] - q_mat[s, a]))

            s, a = sdot, adot
            st_pat, ac_pat = sc_arr[s], sc_arr[a]

            test_bins, sen, sctrs = hn.execute_hopfield(st_pat, total_w)
            test_bina, aen, sctra = hn.execute_hopfield(ac_pat, total_w)

            sc_arr_cntr[0][s] += 1
            sc_arr_cntr[0][a] += 1

            # log energy
            sc_arr_energy[0][s] += sen
            sc_arr_energy[0][a] += aen

            # log number of visits to extract valuable s,a pairs
            visits[s, a] += 1

            nits2 += 1
        nits += 1

    scipy.io.savemat('qconv.mat', mdict={'concv': qconv})

    tmp_energy = sc_arr_energy / sc_arr_cntr
    avg_energy = np.reshape(tmp_energy, (4, 5))

    # Append generated .mat files for visualization
    results_mats.append(sc_arr_energy)
    results_mats.append(q_mat)
    results_mats.append(sreward)
    results_mats.append(visits)
    results_mats.append(avg_energy)
    results_mats.append(sc_arr_cntr)

    sc.save_result_mats(results_mats)

    # Display and save generated plots
    vis.display_plots(results_mats)

    # Display final policy and save it as policy_icounter.txt
    sc.display_policy_sa(icounter, q_mat)

    state_action = vis.display_energy_policy(q_mat, avg_energy)
    vis.display_high2low_policy(state_action)

    # move the generated figures/mats/txt to the res folder
    sc.move_result_patterns(icounter)


if __name__ == '__main__':
    main()
