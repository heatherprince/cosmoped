import numpy as np

def read_file(ini_filename='../input_datafiles/LambdaCDM.ini'):
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

if __name__=='__main__':
    param_dict, param_names=read_file()
    print('The weighting vectors to create in the order they will be created:')
    for n in param_names:
        print(n, param_dict[n])

    print('The fiducial parameters to be used are:')
    for n in param_dict:
        print(n, param_dict[n])
