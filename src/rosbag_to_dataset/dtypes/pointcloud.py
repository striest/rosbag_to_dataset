import numpy as np
import ros_numpy

from sensor_msgs.msg import PointCloud2, PointField

from rosbag_to_dataset.dtypes.base import Dtype
import copy
import numpy.linalg as LA
import struct

ply_header_color = '''ply
format %(pt_format)s 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

ply_header = '''ply
format %(pt_format)s 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''
# pt_format
# binary_little_endian 1.0
# ascii 1.0

PLY_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#c0392b",\
    ]

PLY_COLOR_LEVELS = 20

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
    ''' [255,255,255] -> "#FFFFFF" '''
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])

def color_dict(gradient):
    ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
    return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
        "r":[RGB[0] for RGB in gradient],
        "g":[RGB[1] for RGB in gradient],
        "b":[RGB[2] for RGB in gradient]}

def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
            for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return color_dict(RGB_list)

def polylinear_gradient(colors, n):
    ''' returns a list of colors forming linear gradients between
        all sequential pairs of colors. "n" specifies the total
        number of desired output colors '''
    # The number of colors per individual linear gradient
    n_out = int(float(n) / (len(colors) - 1))
    # returns dictionary defined by color_dict()
    gradient_dict = linear_gradient(colors[0], colors[1], n_out)

    if len(colors) > 1:
        for col in range(1, len(colors) - 1):
            next = linear_gradient(colors[col], colors[col+1], n_out)
            for k in ("hex", "r", "g", "b"):
                # Exclude first point to avoid duplicates
                gradient_dict[k] += next[k][1:]

    return gradient_dict

def color_map(data, colors, nLevels):
    # Get the color gradient dict.
    gradientDict = polylinear_gradient(colors, nLevels)

    # Get the actual levels generated.
    n = len( gradientDict["hex"] )

    # Level step.
    dMin = data.min()
    dMax = data.max()
    step = ( dMax - dMin ) / (n-1)

    stepIdx = ( data - dMin ) / step
    stepIdx = stepIdx.astype(np.int32)

    rArray = np.array( gradientDict["r"] )
    gArray = np.array( gradientDict["g"] )
    bArray = np.array( gradientDict["b"] )

    r = rArray[ stepIdx ]
    g = gArray[ stepIdx ]
    b = bArray[ stepIdx ]

    return r, g, b

def write_ply(fn, verts, colors=None, pt_format='binary_little_endian'):
    '''
    pt_format: text: ascii
               binary: 'binary_little_endian'
    '''
    verts  = verts.reshape(-1, 3)
    if colors is None:
        fmtstr = '%f %f %f'
        headerstr = (ply_header % dict(vert_num=len(verts), pt_format=pt_format)).encode('utf-8')
    else:
        fmtstr = '%f %f %f %d %d %d'
        headerstr = (ply_header_color % dict(vert_num=len(verts), pt_format=pt_format)).encode('utf-8')
        colors = colors.reshape(-1, 3)
        verts  = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write(headerstr)
        # np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        if pt_format == 'ascii':
            np.savetxt(f, verts, fmt=fmtstr)
        elif pt_format == 'binary_little_endian':
            for i in range(verts.shape[0]):
                if colors is None:
                    f.write(bytearray(struct.pack("fff",verts[i,0],verts[i,1],verts[i,2])))
                else:
                    f.write(bytearray(struct.pack("fffccc",verts[i,0],verts[i,1],verts[i,2],
                                colors[i,0].tobytes(), colors[i,1].tobytes(), colors[i,2].tobytes())))
        else:
            print("ERROR: Unknow PLY format {}".format(pt_format))

def output_to_ply(fn, X, imageSize, rLimit, origin, format, use_rgb=False):
    # Check the input X.
    if ( X.max() <= X.min() ):
        raise Exception("X.max() = %f, X.min() = %f." % ( X.max(), X.min() ) )
    
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = np.float32)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))
    rv = copy.deepcopy(vertices)
    rv[:, 0] = vertices[:, 0] - origin[0, 0]
    rv[:, 1] = vertices[:, 1] - origin[1, 0]
    rv[:, 2] = vertices[:, 2] - origin[2, 0]

    r = LA.norm(rv, axis=1).reshape((-1,1))
    mask = r < rLimit
    mask = mask.reshape(( mask.size ))
    # import ipdb; ipdb.set_trace()
    r = r[ mask ]

    if use_rgb:

        cr, cg, cb = color_map(r, PLY_COLORS, PLY_COLOR_LEVELS)

        colors = np.zeros( (r.size, 3), dtype = np.uint8 )

        colors[:, 0] = cr.reshape( cr.size )
        colors[:, 1] = cg.reshape( cr.size )
        colors[:, 2] = cb.reshape( cr.size )
    else:
        colors = None

    write_ply(fn, vertices[mask, :], colors, format)

class PointCloudConvert(Dtype):
    """
    Convert a pointcloud message to numpy
    Note that this one in particular needs some hacks to work.
    """
    def __init__(self, fields, out_format='npy'):
        """
        Args:
            fields: List of fields in the pointcloud
            out_format: npy, ply_binary, ply_text, plt_binary_rgb, plt_text_rgb
        """
        self.fields = fields
        # self.max_num_points = max_num_points
        # self.fill_value = fill_value

        # output file format
        self.gen_rgb = False
        self.pt_format = 'npy'
        if out_format == 'npy':
            pass
        elif out_format == 'ply_binary':
            self.pt_format = 'binary_little_endian'
        elif out_format == 'ply_text':
            self.pt_format = 'ascii'
        elif out_format == 'plt_binary_rgb':
            self.pt_format = 'binary_little_endian'
            self.gen_rgb = True
        elif out_format == 'plt_text_rgb':
            self.pt_format = 'ascii'
            self.gen_rgb = True

    def N(self):
        return [len(self.fields)]

    def rosmsg_type(self):
        return PointCloud2

    def ros_to_numpy(self, msg):
        #BIG HACK TO GET TO WORK WITH ROS_NUMPY
        #The issue is that the rosbag package uses a different class for messages, so we need to convert back to PointCloud2
        msg2 = PointCloud2()
        msg2.header = msg.header
        msg2.height = msg.height
        msg2.width = msg.width
        msg2.fields = msg.fields
        msg2.is_bigendian = msg.is_bigendian
        msg2.point_step = msg.point_step
        msg2.row_step = msg.row_step
        msg2.data = msg.data
        msg2.is_dense = msg.is_dense

        pts = ros_numpy.numpify(msg2)
        pts = np.stack([pts[f].flatten() for f in self.fields], axis=-1)

        # out = np.ones([self.max_num_points, len(self.fields)]) * self.fill_value
        # out[:pts.shape[0]] = pts

        return pts

    def save_file_one_msg(self, data, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data is stored frame by frame like image or point cloud
        """
        data = self.ros_to_numpy(data)

        if ( 2 != len(data.shape) or data.shape[1] != 3 ):
            raise Exception("xyz.shape = {}. ".format(data.shape))

        maxDistColor = 200 # hard code
        if self.pt_format == 'npy':
            np.save(filename + '.npy', data)
        else:
            data = data.transpose(1,0)
            output_to_ply( filename + '.ply', data, [ 1, data.shape[1]], 
                            maxDistColor, np.array([0, 0, 0]).reshape((-1,1)), 
                            self.pt_format, self.gen_rgb )

    def save_file(self, data, filename):
        pass


if __name__ == "__main__":
    points = np.random.random((3,1600*16)).astype(np.float32)
    fn = '/home/amigo/tmp/test'
    output_to_ply(fn + '1.ply', points, [16, 1600], 300, np.array([0,0,0]), format='ascii', use_rgb=False)
    output_to_ply(fn + '2.ply', points, [16, 1600], 300, np.array([0,0,0]), format='ascii', use_rgb=True)
    output_to_ply(fn + '3.ply', points, [16, 1600], 300, np.array([0,0,0]), format='binary_little_endian', use_rgb=False)
    output_to_ply(fn + '4.ply', points, [16, 1600], 300, np.array([0,0,0]), format='binary_little_endian', use_rgb=True)
    np.save(fn + '.npy', points)
    np.savetxt(fn + '.txt', points)