import statistics
import matplotlib.pyplot as plt # type: ignore


data_mpc_s1 = [96, 101, 110, 123, 136, 145, 149, 167, 199, 160] # fail: 0
data_ddpg_s1 = [151, 128, 135, 170, 153, 150, 143, 125, 128, 178] # fail: 0
data_dwa_s1 = [79, 79]
data_gpdf_s1 = [84, 84] # fail: 0

data_mpc_s2 = [154, 173, 88, 144, 147, 107, 139, 164, 88, 145] # fail: 2
data_ddpg_s2 = [120, 126, 117, 118, 116, 117, 116, 120, 114, 120] # fail: 0
data_dwa_s2 = [91, 91]
data_gpdf_s2 = [64, 64] # fail: 0

n_methods = 4
n_bar_per_category = 2
bar_width = 0.5
yerr_width = 5
color_list = ['#2878b5', '#9ac9db', '#f8ac8c', '#f9d4b7']

fig, ax = plt.subplots()

mpc_index = [i*(n_methods+1)*bar_width for i in range(n_bar_per_category)]
ddpg_index = [bar_width + i*(n_methods+1)*bar_width for i in range(n_bar_per_category)]
dwa_index = [2*bar_width + i*(n_methods+1)*bar_width for i in range(n_bar_per_category)]
gpdf_index = [3*bar_width + i*(n_methods+1)*bar_width for i in range(n_bar_per_category)]

# add the mean and error bars
bar_mpc = ax.bar(mpc_index, 
       [statistics.mean(data_mpc_s1), statistics.mean(data_mpc_s2)], bar_width, color=color_list[0],
       yerr=[statistics.stdev(data_mpc_s1), statistics.stdev(data_mpc_s2)], capsize=yerr_width)
bar_ddpg = ax.bar(ddpg_index,
       [statistics.mean(data_ddpg_s1), statistics.mean(data_ddpg_s2)], bar_width, color=color_list[1],
       yerr=[statistics.stdev(data_ddpg_s1), statistics.stdev(data_ddpg_s2)], capsize=yerr_width)
bar_dwa = ax.bar(dwa_index,
       [statistics.mean(data_dwa_s1), statistics.mean(data_dwa_s2)], bar_width, color=color_list[2],
       yerr=[statistics.stdev(data_dwa_s1), statistics.stdev(data_dwa_s2)], capsize=yerr_width)
bar_gpdf = ax.bar(gpdf_index,
       [statistics.mean(data_gpdf_s1), statistics.mean(data_gpdf_s2)], bar_width, color=color_list[3],
       yerr=[statistics.stdev(data_gpdf_s1), statistics.stdev(data_gpdf_s2)], capsize=yerr_width)

ax.plot([mpc_index[0]]*len(data_mpc_s1), data_mpc_s1, '.', color='black')
ax.plot([ddpg_index[0]]*len(data_ddpg_s1), data_ddpg_s1, '.', color='black')
ax.plot([dwa_index[0]]*len(data_dwa_s1), data_dwa_s1, '.', color='black')
ax.plot([gpdf_index[0]]*len(data_gpdf_s1), data_gpdf_s1, '.', color='black')

ax.plot([mpc_index[1]]*len(data_mpc_s2), data_mpc_s2, '.', color='black')
ax.plot([ddpg_index[1]]*len(data_ddpg_s2), data_ddpg_s2, '.', color='black')
ax.plot([dwa_index[1]]*len(data_dwa_s2), data_dwa_s2, '.', color='black')
ax.plot([gpdf_index[1]]*len(data_gpdf_s2), data_gpdf_s2, '.', color='black')

ax.set_xticks([x + bar_width/2 for x in ddpg_index])
ax.set_xticklabels(['Multi-robot test 1', 'Multi-robot test 2'], fontsize=16)
ax.set_ylabel('Final step', fontsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.legend((bar_mpc, bar_ddpg, bar_dwa, bar_gpdf), ('MPC', 'DDPG', 'DWA', 'GF-DWA'), prop={'size': 16}, ncols=4)

ax.set_ylim(0, 220)
# plt.title('Final step comparison')
plt.show()