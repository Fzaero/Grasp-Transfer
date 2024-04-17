import numpy as np

def find_inverse_RT_3x4(ref_RT):
    RT= np.eye(4)
    RT[:3,:] = ref_RT
    return find_inverse_RT_4x4(RT)

def find_inverse_RT_4x4(RT):
    R_,T_ = RT[:3,:3], RT[:3,3]
    RT_inv = np.eye(4)
    RT_inv[:3,:3] = np.linalg.inv(R_)
    RT_inv[:3,3]  =- RT_inv[:3,:3]@ T_ 
    return RT_inv