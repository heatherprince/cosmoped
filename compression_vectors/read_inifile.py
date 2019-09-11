import numpy as np

def read_settings_file(filename):
    # read in and split rows that aren't comments
    f = open(filename, 'r+')
    data = [line.strip().split('=') for line in f.readlines()
                if not (line.startswith('#') or line.startswith('\n'))]
    f.close()

    # remove spaces before or after =
    data = [[item.strip() for item in line] for line in data]

    # remove comments at the end of rows
    for row in data:
        for i, val in enumerate(row):
            if '#' in val:
                row=row[:i]
                break

    names = [row[0] for row in data]
    vals =  [row[1] for row in data]

    dict={}
    for i, n in enumerate(names):
        dict[n]=vals[i]
    # deal with which are floats/ints when initialising sampling variables

    year=int(dict['year'])
    compression_inifile=dict['compression_inifile']
    data_dir=dict['data_dir']
    save_dir=dict['save_dir']

    return year, compression_inifile, data_dir, save_dir

def read_compression_file(ini_filename='../inifiles/LambdaCDM.ini'):
    f = open(ini_filename, 'r+')
    ini_data = [line.strip().split() for line in f.readlines()
                if not (line.startswith('#') or line.startswith('\n'))]
    f.close()

    param_names=np.array([row[0] for row in ini_data])
    param_fiducial_values=np.array([float(row[1]) for row in ini_data])
    use_param=np.array([int(row[2]) if len(row)>2 else 0 for row in ini_data ], dtype=bool)
    param_order=np.array([int(row[3]) if len(row)>3 else -1 for row in ini_data ], dtype=int)

    #dictionary with fiducial values for all parameters
    #(including some we might not want to compute weighting vectors for)
    param_dict={}
    for i, n in enumerate(param_names):
        param_dict[n]=param_fiducial_values[i]

    param_names_make_vectors=param_names[use_param]
    param_order=param_order[use_param]

    #assume that param order has values in some random order with max n and -1 for the rest
    #want to make the rest n+1, n+2 ... in order
    if np.amin(param_order)<0:
        max=np.amax(param_order)
        for i, order in enumerate(param_order):
            if order<0:
                #user did not provide a specific order, go in order params were read
                param_order[i]=max+1+i

    inds=param_order.argsort()
    ordered_param_names_make_vectors=param_names_make_vectors[inds]

    return param_dict, ordered_param_names_make_vectors
