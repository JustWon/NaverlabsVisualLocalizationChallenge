import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def intrinsic_params(camera_model):
    if (camera_model == '22970285' or 
        camera_model == '22970286' or 
        camera_model == '22970288' or 
        camera_model == '22970289' or 
        camera_model == '22970290' or 
        camera_model == '22970291'):
        
        A = np.array([[1725.842032333, 0.0, 1024.0],
                      [0.0, 1725.842032333, 768.0],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (0,0,0,0,0)
        image_size = np.array([2048, 1536])
        
    elif (camera_model == 'AC01324954' or
          camera_model == 'AC01324955' or
          camera_model == 'AC01324968' or
          camera_model == 'AC01324969'):
        
        A = np.array([[2192.6345, 0.0, 1080.0],
                      [0.0, 2192.6345, 1440.0],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (0.309526634993, -1.68546591669, 0.000516016398484, 0.000304875649237, 2.77731885597)
        image_size = np.array([2160, 2880])
    
    elif (camera_model == '40027089'):
        A = np.array([[1716.28280, 0.0, 1299.03953],
                      [0.0, 1713.79148, 1013.52990],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.2032717, 0.2402597, -0.0005943499, -0.002147060, -0.1755963)
        image_size = np.array([2592, 2048])
    elif (camera_model == '40029628'):
        A = np.array([[1722.61355, 0.0, 1275.30243],
                      [0.0, 1721.63577, 992.13280],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.1676073, 0.09076241, -0.0008216913, -0.0006427209, -0.001445933)
        image_size = np.array([2592, 2048])
    elif (camera_model == '40030065'):
        A = np.array([[1705.26708, 0.0, 1279.42131],
                      [0.0, 1705.66273, 1043.08606],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.2035782, 0.2375905, -0.0004143076, -0.001001452, -0.1529464)
        image_size = np.array([2592, 2048])
    elif (camera_model == '40031951'):
        A = np.array([[1705.96608, 0.0, 1293.23000],
                      [0.0, 1704.38140, 1003.98063],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.2277732, 0.3183032, -0.002417126, -0.001141689, -0.2496837)
        image_size = np.array([2592, 2048])
    elif (camera_model == '40033113'):
        A = np.array([[1706.13584, 0.0, 1311.65649],
                      [0.0, 1705.82533, 1005.88848],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.1755889, 0.1322496 -0.001250907, -0.0007547798, -0.03799974)
        image_size = np.array([2592, 2048])
    elif (camera_model == '40033116'):
        A = np.array([[1735.26550, 0.0, 1264.83282],
                      [0.0, 1733.47093, 988.15107],
                      [0.0, 0.0, 1.0]])
        dist_coeff = (-0.1884311, 0.1090200, -0.00006484913, -0.0007721923, 0.009144638)
        image_size = np.array([2592, 2048])
        
    return image_size, A, dist_coeff

def placeRecognitionTopFive(dataset, predictions, query_idx, mode='test', viz=False):
    
    query_image_full_path = dataset.dbStruct.q_image[query_idx]
    query_img = Image.open(query_image_full_path)
    
    if (mode=='val' or mode=='train'):
        query_pose = dataset.dbStruct.q_full_pose[query_idx]
    else:
        query_pose = None
    
    if (viz):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,5,1)
        ax1.title.set_text('query image')
        ax1.imshow(query_img)
    
    pred_list = []
    for rank in range(5):
        pred_image_full_path = dataset.dbStruct.db_image[predictions[query_idx][rank]]
        pred_img = Image.open(pred_image_full_path) 
        pred_pose = dataset.dbStruct.db_full_pose[predictions[query_idx][rank]]

        if (viz):
            ax = fig.add_subplot(2,5,6 + rank)
            ax.title.set_text('rank %d reference image' % (rank))
            ax.imshow(pred_img)
        
        pred_list.append([pred_image_full_path, pred_img, pred_pose])
    
    if (viz):
        plt.show()
    
    query_item = [query_image_full_path, query_img, query_pose]
    
    return query_item, pred_list


def placeRecognitionAll(dataset, predictions, query_idx, mode='test'):
    
    query_image_full_path = dataset.dbStruct.q_image[query_idx]
    query_img = Image.open(query_image_full_path)
    
    if (mode=='val' or mode=='train'):
        query_pose = dataset.dbStruct.q_full_pose[query_idx]
    else:
        query_pose = None
    
    pred_list = []
    for rank in range(20):
        pred_image_full_path = dataset.dbStruct.db_image[predictions[query_idx][rank]]
        pred_img = Image.open(pred_image_full_path) 
        pred_pose = dataset.dbStruct.db_full_pose[predictions[query_idx][rank]]
        
        pred_list.append([pred_image_full_path, pred_img, pred_pose])
    
    query_item = [query_image_full_path, query_img, query_pose]
    
    return query_item, pred_list

def projection(img, _points, _A, _Rt, thickness=1):

    projected_img = np.ones_like(img,dtype=float)*np.inf
    agumented_points = np.c_[_points,np.ones(_points.shape[0])]
    transformed_points = np.linalg.inv(_Rt)@np.transpose(agumented_points)
    projected_points = _A@transformed_points[:3,:]
    hnormalized_points = projected_points/projected_points[2,:]
    
    # transpose
    hnormalized_points = np.transpose(hnormalized_points.astype(int))
    transformed_points = np.transpose(transformed_points)
    
    # in front of cameras only
    condition1 = transformed_points[:,2] > 0
    hnormalized_points = hnormalized_points[condition1]
    transformed_points = transformed_points[condition1]
    
    # inside of the image region
    condition2 = ((0 <= hnormalized_points[:,0]) & (hnormalized_points[:,0] < img.size[0]) & (0 <= hnormalized_points[:,1]) & (hnormalized_points[:,1] < img.size[1]))
    hnormalized_points = hnormalized_points[condition2]
    transformed_points = transformed_points[condition2]

    for i, hnormalized_point in enumerate(hnormalized_points):
        t_y = hnormalized_point[1]
        t_x = hnormalized_point[0]

        new_val = transformed_points[i]
        pre_val = projected_img[t_y, t_x]
        if (new_val[2] < pre_val[2]):
            projected_img[t_y-thickness:t_y+thickness, t_x-thickness:t_x+thickness] = transformed_points[i,:3]

    return projected_img


import pyquaternion as pyq
import math
from scipy.spatial.transform import Rotation as R

def pose_eval(query_rot, query_trans, GT_pose):
    
    GT_Rt = np.eye(4)
    GT_Rt[:3,3] = GT_pose[:3]
    (GT_qw, GT_qx, GT_qy, GT_qz) = GT_pose[3:]
    GT_Rt[:3,:3] = R.from_quat([GT_qx,GT_qy,GT_qz,GT_qw]).as_matrix()
    
    query_quaternion = R.from_matrix(query_rot).as_quat()
    query_quaternion = pyq.Quaternion(query_quaternion[3],query_quaternion[0],query_quaternion[1],query_quaternion[2])
    GT_quaternion = pyq.Quaternion(GT_qw,GT_qx,GT_qy,GT_qz)

    # Get the 3D difference between these two orientations
    qd = GT_quaternion.conjugate * query_quaternion

    # Calculate Euler angles from this difference quaternion
    phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
    theta = math.asin ( 2 * (qd.w * qd.y - qd.z * qd.x) )
    psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )

    print('Difference')
    print('phi \t: ', phi* 180 / 3.141592)
    print('theta \t: ', theta* 180 / 3.141592)
    print('psi \t: ', psi* 180 / 3.141592)
    print('trans\t: ',np.linalg.norm(query_trans- GT_pose[:3]))