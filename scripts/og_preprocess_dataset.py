from os import listdir

# rootfolder = '/project/learningphysics/dataset'
# subfolders = ['20210812','20210812_interventions','20210826_bags','20210828','20210902','20210903','20210910_bags','gascola_laps']

# outtxt = open('trajlist.txt','w')
# for k, subfolder in enumerate(subfolders):
#     subdir = rootfolder + '/' + subfolder
#     bagfiles = listdir(subdir)
#     bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
#     bagfiles.sort()
#     print('find {} bagfiles in {}'.format(len(bagfiles), subdir))

#     for bag in bagfiles:
#         outtxt.write(subdir + '/' + bag)
#         outtxt.write(' ')
#         if k<2:
#             outtxt.write(subfolder+'_'+bag.split('.bag')[0])
#         else:
#             outtxt.write(bag.split('.bag')[0])
#         outtxt.write('\n')
# outtxt.close()


# rootfolder = '/project/learningphysics/2022-06-13'
# subfolders = [
#     'exp2_rough_rider_down_run1',
#     'exp2_turnpike_3ms_1',
#     'exp3_mppi_baseline_turnpike_pole_1',
#     'exp3_mppi_baseline_turnpike_pole_2',
#     'exp3_mppi_baseline_turnpike_pole_3',
#     'exp3_mppi_baseline_warehouse_1',
#     'exp3_mppi_baseline_warehouse_2',
#     'exp3_mppi_baseline_warehouse_3',
#     'exp3_mppi_baseline_warehouse_4',
#     'exp3_mppi_baseline_warehouse_5',
#     'exp3_mppi_both_warehouse_1',
#     'exp3_mppi_both_warehouse_2',
#     'exp3_mppi_both_warehouse_3',
#     'exp3_mppi_both_warehouse_4',
#     'exp3_mppi_both_warehouse_5',
#     'exp3_mppi_fused_turnpike_pole_2',
#     'exp3_mppi_fused_turnpike_pole_3',
#     'exp3_mppi_learned_turnpike_pole_1',
#     'mppi_test_run',
# ]

'''
rootfolder = '/project/learningphysics/corl_sara/2022-06-19'
subfolders = [
    'exp1_warehouse_baseline_1',
    'exp1_warehouse_baseline_2',
    'exp1_warehouse_baseline_3',
    'exp1_warehouse_both_warehouse_1',
    'exp1_warehouse_both_warehouse_2',
    'exp2_figure8_both',
    'exp2_figure8_both_part2',
]
'''

rootfolder = '/project/learningphysics/parv_dataset/affix_sys_id'
subfolders = [
    'train_tartan_bags',
    # 'exp1_warehouse_baseline_2',
    # 'exp1_warehouse_baseline_3',
    # 'exp1_warehouse_both_warehouse_1',
    # 'exp1_warehouse_both_warehouse_2',
    # 'exp2_figure8_both',
    # 'exp2_figure8_both_part2',
]
outtxt = open('trajlist_tartan_train.txt','w')
for k, subfolder in enumerate(subfolders):
    subdir = rootfolder + '/' + subfolder
    bagfiles = listdir(subdir)
    bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
    bagfiles.sort()
    print('find {} bagfiles in {}'.format(len(bagfiles), subdir))

    for k,bag in enumerate(bagfiles):
        outtxt.write(subdir + '/' + bag)
        outtxt.write(' ')
        # outtxt.write(subfolder+'_'+ str(k))
        outtxt.write(bag[:-4])
        outtxt.write('\n')
outtxt.close()
