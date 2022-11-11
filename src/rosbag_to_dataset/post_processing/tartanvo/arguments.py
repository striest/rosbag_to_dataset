# arguments for tartandrive
common_args = {}
common_args['image_width']      = 1024 # the width of the image from the sensor
common_args['image_height']     = 544 # the height of the image from the sensor
common_args['focal_x']          = 477.6049499511719 # The focal length of the camera
common_args['focal_y']          = 477.6049499511719 # The focal length of the camera
common_args['center_x']         = 499.5 # The optical center of the camera
common_args['center_y']         = 252.0 # The optical center of the camera
common_args['focal_x_baseline'] = 100.14994812011719 # focal lengh multiplied by baseline
common_args['batch_size']       = 128 # The batch size
common_args['worker_num']       = 4 # The number of workers in the dataloader
common_args['image_compressed'] = False 
common_args['stereo_maps']      = '' 

vo_args = {}
vo_args['model_name']           = '43_6_2_vonet_30000.pkl' # 'The name of the pretrained model, located in the model folder. 
vo_args['network_type']         = 1 # Load different architecture of vonet
vo_args['image_crop_w']         = 102 # crop out the image because of vignette effect
vo_args['image_crop_h']         = 0 # crop out the image because of vignette effect
vo_args['image_resize_w']       = 844 # the width of the input image into the model
vo_args['image_resize_h']       = 448 # the height of the input image into the model
vo_args['image_input_w']        = 640 # the width of the input image into the model
vo_args['image_input_h']        = 448 # the height of the input image into the model
vo_args['visualize']            = False # help='visualize the depth estimation (default: False)

stereo_args = {}
stereo_args['model_name']       = '6_3_2_stereo_60000.pkl' # The name of the pretrained model, located in the model folder. 
stereo_args['image_crop_w']     = 64 # crop out the image because of vignette effect
# stereo_args['image_crop_h']     = 32 # crop out the image because of vignette effect
stereo_args['image_crop_h_low'] = 32 
stereo_args['image_crop_h_high']= 32 
stereo_args['image_input_w']    = 512 # the width of the input image into the model
stereo_args['image_input_h']    = 256 # the height of the input image into the model
stereo_args['visualize_depth']  = False # visualize the depth estimation (default: False)
stereo_args['pc_min_dist']      = 2.5 # Minimum distance of the points, filter out close points on the vehicle
stereo_args['pc_max_dist']      = 15.0 # Maximum distance of the points
stereo_args['pc_max_height']    = 2.0 # Maximum height of the points
stereo_args['uncertainty_thresh']    = -2.5 
stereo_args['mask_file']        = 'atvmask_tartandrive.npy' 
stereo_args['colored_folder']   = 'image_left_color' 

