from convert_folder_to_torch import *
from torch.utils.data import Dataset, DataLoader
class MapDataset(Dataset):

    def __init__(self,rgb_map_fp_list,hmap_fp_list,mask_fp_list,output_res):
        self.rgb_map_fp_list = rgb_map_fp_list
        self.hmap_fp_list = hmap_fp_list
        self.mask_fp_list = mask_fp_list
        self.output_res = output_res
    
    def __len__(self):
        return len(self.rgb_map_fp_list)
    
    def __getitem__(self,idx):
        rgb = cv2.cvtColor(cv2.imread(self.rgb_map_fp_list[idx]), cv2.COLOR_BGR2RGB)/255. # of type np.uint8
        height = np.fromfile(self.hmap_fp_list[idx], dtype=np.float32).reshape(600, 600) 
        mask = cv2.imread(self.mask_fp_list[idx], cv2.IMREAD_GRAYSCALE) # 2D mask of type np.uint8 {0, 255}
        mask = (mask/255.0)
        if self.output_res != '':
            height = cv2.resize(height, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)

        height = torch.from_numpy(height).unsqueeze(-1)
        mask = torch.from_numpy(mask).unsqueeze(-1)

        height = torch.cat((height,mask),dim=-1).permute(2,0,1)
        rgb = torch.from_numpy(rgb).permute(2,0,1)

        return [height,rgb]


def overwrite_traj(traj,new_x,traj_key):
    if traj_key in traj['observation'].keys():
        traj['observation'][traj_key] = x
        traj['next_observation'][traj_key] = new_x[1:]


def use_new_maps(traj_source_fp,root_maps_source_fp,maps_source_fp_txt,save_fp,output_res,map_idx_txt_file):
    traj = torch.load(traj_source_fp)
    del traj['observation']['heightmap']
    del traj['next_observation']['heightmap']
    del traj['observation']['rgbmap']
    del traj['next_observation']['rgbmap']
    with open(maps_source_fp_txt) as f:
        f_list = f.readlines()
        rgb_map_fp_list = [x.strip().split(',')[map_idx_txt_file] for x in f_list]
        hmap_fp_list = [x.strip().split(',')[map_idx_txt_file+1] for x in f_list]
        mask_fp_list = [x.strip().split(',')[map_idx_txt_file+2] for x in f_list]
        rgb_map_fp_list.sort()
        hmap_fp_list.sort()
        mask_fp_list.sort()

        rgb_map_fp_list = [join(root_maps_source_fp,x) for x in rgb_map_fp_list]
        hmap_fp_list = [join(root_maps_source_fp,x) for x in hmap_fp_list]
        mask_fp_list = [join(root_maps_source_fp,x) for x in mask_fp_list]

    h_mask_list = []
    rgb_list = []
    
    # for i in range(len(traj['action'])+1): 
    #     rgb = cv2.cvtColor(cv2.imread(rgb_map_fp_list[i]), cv2.COLOR_BGR2RGB)/255. # of type np.uint8
    #     height = np.fromfile(hmap_fp_list[i], dtype=np.float32).reshape(600, 600) 
    #     mask = cv2.imread(mask_fp_list[i], cv2.IMREAD_GRAYSCALE) # 2D mask of type np.uint8 {0, 255}
    #     mask = (mask/255.0)
    #     if output_res != '':
    #         height = cv2.resize(height, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
    #         mask = cv2.resize(mask, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
    #         rgb = cv2.resize(rgb, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)

    #     height = torch.from_numpy(height).unsqueeze(-1)
    #     mask = torch.from_numpy(mask).unsqueeze(-1)

    dataloader = iter(DataLoader(MapDataset(rgb_map_fp_list,hmap_fp_list,mask_fp_list,output_res),shuffle=False,batch_size=1))
    for i in range(len(traj['action'])+1):
        batch = next(dataloader)
        overwrite_traj(traj,batch[0],'heightmap')
        overwrite_traj(traj,batch[1],'rgbmap')
        # h_mask_list.append(batch[0])
        # rgb_list.append(batch[1])
    # traj = overwrite_traj(traj,torch.cat(h_mask_list,dim=0),'heightmap')
    # traj = overwrite_traj(traj,torch.cat(rgb_list,dim=0),'rgbmap')

    torch.save(traj,save_fp)


if __name__ =='__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_source_fp', type=str, required=True, help='Path to the traj_source directory')
    parser.add_argument('--maps_source_fp', type=str, required=True, help='Path to the maps_source directory')
    parser.add_argument('--save_fp', type=str, required=True, help='Path to the destination trajectory')
    parser.add_argument('--traj_file', type=str, required=False,default = "None", help='Path to the traj list. If None then entire source folder is converted')
    parser.add_argument('--output_res', type=str, required=False,default='', help='Output Resolution')
    parser.add_argument('--txt_file_idx', type=int, required=False, default=0,help='Index of the txt file coloumns to be used for maps - i , i+1, i+2')


    args = parser.parse_args()

    root_source_fp = args.root_source_fp
    map_source_fp = args.maps_source_fp
    root_save_fp = args.save_fp
    traj_file = args.traj_file
    if args.output_res != '':
        output_res = [int(x) for x in args.output_res.split(',')]
    else:
        output_res=''
    now = datetime.now()
    program_time = f'{now.strftime("%m-%d-%Y,%H-%M-%S")}'
    root_save_fp = join(root_save_fp,program_time)

    maybe_mkdirs(root_save_fp, force=True)


    if traj_file == "None":
        traj_name = [x for x in listdir(root_source_fp) if x.endswith('.pt')]
    else:
        traj_file = open(traj_file)
        traj_list = traj_file.read()
        traj_name = traj_list.split(', ')
    
    already_extracted_traj_name = [x[:-3] for x in listdir(root_save_fp) if x.endswith('.pt')]
    print("According to Traj List ", len(traj_name))

    traj_name = [x for x in traj_name if x not in set(already_extracted_traj_name)]

    traj_name.sort()

    print(already_extracted_traj_name)
    print("Remaining after ignoring already extracted ", len(traj_name))

    for i in tqdm(range(len(traj_name))):
        x = traj_name[i]
        traj_source_fp = join(root_source_fp,f'{x}.pt')
        save_fp = join(root_save_fp,f'{x}.pt')
        map_source_fp_txt = join(map_source_fp,'traj_frame_lists',f'{x}.txt')
        use_new_maps(traj_source_fp,map_source_fp,map_source_fp_txt,save_fp,output_res,args.txt_file_idx)

    # num_proc = 30
    # i=0
    # while i < len(traj_name):
    #     print(i)
    #     pool = multiprocessing.Pool(processes=num_proc)
    #     for j in range(num_proc):
    #         if (i+j) >=len(traj_name):
    #             break
    #         x = traj_name[i+j]

    #         traj_source_fp = join(root_source_fp,f'{x}.pt')
    #         save_fp = join(root_save_fp,f'{x}.pt')
    #         map_source_fp_txt = join(map_source_fp,'traj_frame_lists',f'{x}.txt')
    #         try:

    #             pool.apply_async(use_new_maps,args=(traj_source_fp,map_source_fp,map_source_fp_txt,save_fp,output_res,args.txt_file_idx))
    #         except Exception as e:
    #             print(e)
    #             print(x)

            
    #     pool.close()
    #     pool.join()
    #     i+=num_proc