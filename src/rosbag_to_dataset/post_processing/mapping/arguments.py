import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Deep learning stereo and vo')

    parser.add_argument('--resolution', type=float, default=0.02,
                        help='The resolution of the map')

    parser.add_argument('--min-x', type=float, default=-2.0,
                        help='The minimum range of the map')

    parser.add_argument('--max-x', type=float, default=10.0,
                        help='The maximum range of the map')

    parser.add_argument('--min-y', type=float, default=-6.0,
                        help='The minimum range of the map')

    parser.add_argument('--max-y', type=float, default=6.0,
                        help='The maximum range of the map')


    parser.add_argument('--max-points-num', type=int, default=3000000,
                        help='maximum number of the points')

    parser.add_argument('--visualize-maps', action='store_true', default=False,
                        help='Visualize the map')

    args = parser.parse_args()

    return args
