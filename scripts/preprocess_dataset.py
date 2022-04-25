from os import listdir

rootfolder = '/project/learningphysics/dataset'
subfolders = ['20210812','20210812_interventions','20210826_bags','20210828','20210902','20210903','20210910_bags','gascola_laps']

outtxt = open('trajlist.txt','w')
for k, subfolder in enumerate(subfolders):
    subdir = rootfolder + '/' + subfolder
    bagfiles = listdir(subdir)
    bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
    bagfiles.sort()
    print('find {} bagfiles in {}'.format(len(bagfiles), subdir))

    for bag in bagfiles:
        outtxt.write(subdir + '/' + bag)
        outtxt.write(' ')
        if k<2:
            outtxt.write(subfolder+'_'+bag.split('.bag')[0])
        else:
            outtxt.write(bag.split('.bag')[0])
        outtxt.write('\n')
outtxt.close()
