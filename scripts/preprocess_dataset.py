from os import listdir
from os.path import join, isdir

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

# rootfolder = '/project/learningphysics/corl_sara/2022-06-19'
# subfolders = [
#     'exp1_warehouse_baseline_1',
#     'exp1_warehouse_baseline_2',
#     'exp1_warehouse_baseline_3',
#     'exp1_warehouse_both_warehouse_1',
#     'exp1_warehouse_both_warehouse_2',
#     'exp2_figure8_both',
#     'exp2_figure8_both_part2',
# ]

# outtxt = open('trajlist_corl2.txt','w')
# for k, subfolder in enumerate(subfolders):
#     subdir = rootfolder + '/' + subfolder
#     bagfiles = listdir(subdir)
#     bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
#     bagfiles.sort()
#     print('find {} bagfiles in {}'.format(len(bagfiles), subdir))

#     for k,bag in enumerate(bagfiles):
#         outtxt.write(subdir + '/' + bag)
#         outtxt.write(' ')
#         outtxt.write(subfolder+'_'+ str(k))
#         outtxt.write('\n')
# outtxt.close()

# rootfolder = '/project/learningphysics/2022-06-28/aggresive'
# subfolders = ['run1', 'run10', 'run11', 'run12', 'run13', 'run14', 'run15', 'run16', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8', 'run9']
# rootfolder2 = '/project/learningphysics/2022-06-28/sara-obs'

# outtxt = open('trajlist_062822.txt','w')
# for k, subfolder in enumerate(subfolders):
#     subdir = rootfolder + '/' + subfolder
#     bagfiles = listdir(subdir)
#     bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
#     bagfiles.sort()
#     print('find {} bagfiles in {}'.format(len(bagfiles), subdir))

#     for k,bag in enumerate(bagfiles):
#         outtxt.write(subdir + '/' + bag)
#         outtxt.write(' ')
#         outtxt.write(subfolder)
#         outtxt.write('\n')

# bagfiles = listdir(rootfolder2)
# bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
# bagfiles.sort()
# print('find {} bagfiles in {}'.format(len(bagfiles), subdir))
# for k,bag in enumerate(bagfiles):
#     outtxt.write(rootfolder2 + '/' + bag)
#     outtxt.write(' ')
#     outtxt.write('sara_obs_'+str(k))
#     outtxt.write('\n')

# outtxt.close()

def listfolder(folderdir):
    print(folderdir)
    baglist = []
    files = listdir(folderdir)
    bagfiles = [bb for bb in files if bb.endswith('.bag')]
    subfolders = [ff for ff in files if isdir(join(folderdir,ff))]
    # import ipdb;ipdb.set_trace()
    if len(bagfiles) > 0:
        bagfiles.sort()
        for bag in bagfiles:
            baglist.append(join(folderdir, bag))

    if len(subfolders) > 0:
        subfolders.sort()
        for subfolder in subfolders:
            subfolderdir = join(folderdir, subfolder)
            subbags = listfolder(subfolderdir)
            baglist.extend(subbags)

    return baglist

# organize 2022 data
rootfolder = '/home/offroad/parv_code/data/test_baseline_extract'
subfolders = [
            # '2022-04-15',
            # '2022-05-02',
            # '2022-05-05',
            # '2022-05-31',
            # '2022-06-04',
            # '2022-06-12',
            # '2022-06-13',
            # '2022-06-16',
            # '2022-06-28',
            # '2022-06-30',
            # '2022-07-13',
            # '2022-07-20',
            # '2022-07-22',
            # '2022-07-27',
            # '2022-07-31',
            # '2022-08-16',
            # '2022-09-08',
            # '2022-09-27',
            # '2022-10-04',
            'bagfiles']
# rootfolder = '/cairo/arl_bag_files'
# subfolders = ['SARA', 'racer']

outtxt = open('trajlist_test.txt','w')
for k, subfolder in enumerate(subfolders):
    subdir = join(rootfolder, subfolder)
    bagfiles = listfolder(subdir)
    # import ipdb;ipdb.set_trace()
    for k,bag in enumerate(bagfiles):
        outtxt.write(bag)
        outtxt.write(' ')
        outtxt.write(bag.split('/')[-1].split('.bag')[0])
        outtxt.write('\n')
outtxt.close()