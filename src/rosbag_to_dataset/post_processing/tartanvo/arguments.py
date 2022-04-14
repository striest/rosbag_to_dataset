import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Deep learning stereo and vo')

    parser.add_argument('--model-name', default='5_5_4_stereo_30000.pkl', # 4_3_3_stereo_60000.pkl', # 
                        help='The name of the pretrained model, located in the model folder. ')

    parser.add_argument('--network-type', type=int, default=0,
                        help='Load different architecture of vonet')

    parser.add_argument('--image-width', type=int, default=1024,
                        help='the width of the image from the sensor')

    parser.add_argument('--image-height', type=int, default=544,
                        help='the height of the image from the sensor')

    parser.add_argument('--focal-x', type=float, default=477.6049499511719,
                        help='The focal length of the camera')

    parser.add_argument('--focal-y', type=float, default=477.6049499511719,
                        help='The focal length of the camera')

    parser.add_argument('--center-x', type=float, default=499.5,
                        help='The optical center of the camera')

    parser.add_argument('--center-y', type=float, default=252.0,
                        help='The optical center of the camera')

    parser.add_argument('--focal-x-baseline', type=float, default=100.14994812011719,
                        help='focal lengh multiplied by baseline')

    parser.add_argument('--image-crop-w', type=int, default=64,
                        help='crop out the image because of vignette effect')

    parser.add_argument('--image-crop-h', type=int, default=32,
                        help='crop out the image because of vignette effect')

    parser.add_argument('--image-input-w', type=int, default=512,
                        help='the width of the input image into the model')

    parser.add_argument('--image-input-h', type=int, default=256,
                        help='the height of the input image into the model')

    parser.add_argument('--visualize-depth', action='store_true', default=False,
                        help='visualize the depth estimation (default: False)')

    parser.add_argument('--pc-min-dist', type=float, default=2.5,
                        help='Minimum distance of the points, filter out close points on the vehicle')

    parser.add_argument('--pc-max-dist', type=float, default=15.0,
                        help='Maximum distance of the points')

    parser.add_argument('--pc-max-height', type=float, default=2.0,
                        help='Maximum height of the points')

    parser.add_argument('--transform-ground', action='store_true', default=False,
                        help='Transform the point cloud (default: False)')

    parser.add_argument('--batch-size', type=int, default=8,
                        help='The batch size')

    parser.add_argument('--worker-num', type=int, default=4,
                        help='The number of workers in the dataloader')

    args = parser.parse_args()

    return args
