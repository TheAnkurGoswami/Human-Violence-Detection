import numpy as np
import shelve
import tqdm
#Function to calculate angle between two vectors
def angle_bw_vectors(x,y):
    x_mag=np.sqrt(np.square(x).sum())
    y_mag=np.sqrt(np.square(y).sum())
    dot_prod=np.dot(x,y)
    if x_mag==0 or y_mag==0:
        return 0.0

    if dot_prod/(x_mag*y_mag)>0:
        angle=np.arccos(min(1,dot_prod/(x_mag*y_mag)))
    else:
        angle=np.arccos(max(-1,dot_prod/(x_mag*y_mag)))
        
    return np.degrees(angle)

#Function to generate angles
def generate_angles(person):

    if tuple(person['left_elbow'])==(-1,-1) or tuple(person['left_wrist'])==(-1,-1) or tuple(person['left_shoulder'])==(-1,-1):
        left_elbow_ang=0.0
    else:
        left_elbow_ang=angle_bw_vectors(person['left_elbow']-person['left_wrist'],person['left_elbow']-person['left_shoulder'])

    if tuple(person['right_elbow'])==(-1,-1) or tuple(person['right_wrist'])==(-1,-1) or tuple(person['right_shoulder'])==(-1,-1):
        right_elbow_ang=0.0
    else:
        right_elbow_ang=angle_bw_vectors(person['right_elbow']-person['right_wrist'],person['right_elbow']-person['right_shoulder'])

    if tuple(person['left_knee'])==(-1,-1) or tuple(person['left_ankle'])==(-1,-1) or tuple(person['left_hip'])==(-1,-1):
        left_knee_ang=0.0
    else:
        left_knee_ang=angle_bw_vectors(person['left_knee']-person['left_ankle'],person['left_knee']-person['left_hip'])

    if tuple(person['right_knee'])==(-1,-1) or tuple(person['right_ankle'])==(-1,-1) or tuple(person['right_hip'])==(-1,-1):
        right_knee_ang=0.0
    else:
        right_knee_ang=angle_bw_vectors(person['right_knee']-person['right_ankle'],person['right_knee']-person['right_hip'])

    if tuple(person['left_hip'])==(-1,-1) or tuple(person['left_knee'])==(-1,-1) or tuple(person['left_shoulder'])==(-1,-1):
        left_hip_ang=0.0
    else:
        left_hip_ang=angle_bw_vectors(person['left_hip']-person['left_knee'],person['left_hip']-person['left_shoulder'])

    if tuple(person['right_hip'])==(-1,-1) or tuple(person['right_knee'])==(-1,-1) or tuple(person['right_shoulder'])==(-1,-1):
        right_hip_ang=0.0
    else:
        right_hip_ang=angle_bw_vectors(person['right_hip']-person['right_knee'],person['right_hip']-person['right_shoulder'])

    if tuple(person['left_shoulder'])==(-1,-1) or tuple(person['left_elbow'])==(-1,-1) or tuple(person['neck'])==(-1,-1):
        left_shoulder_ang=0.0
    else:
        left_shoulder_ang=angle_bw_vectors(person['left_shoulder']-person['left_elbow'],person['left_shoulder']-person['neck'])

    if tuple(person['right_shoulder'])==(-1,-1) or tuple(person['right_elbow'])==(-1,-1) or tuple(person['neck'])==(-1,-1):
        right_shoulder_ang=0.0
    else:
        right_shoulder_ang=angle_bw_vectors(person['right_shoulder']-person['right_elbow'],person['right_shoulder']-person['neck'])


    angles=[left_elbow_ang,right_elbow_ang,left_knee_ang,right_knee_ang,left_hip_ang,right_hip_ang,left_shoulder_ang,right_shoulder_ang]

    
    return angles



# Function to generate batches(of sequences) with 10 T-States and jump of 5 step.
def batch(array,t_states=10,strides=5):
    temp=[]
    if len(array)%strides==0:
        for k in range(len(array)//strides-1):
            temp.append(array[k*strides:k*strides+t_states])
    else: 
        for k in range(len(array)//strides-1):
            temp.append(array[k*strides:k*strides+t_states])
        temp.append(array[-1*t_states:])
        
    return np.array(temp)

# Function to convert shelve file to numpy array
def shelve_to_nparray(path):
    file=shelve.open(path)
    angles=[]
    n=len(f['data'])
    for i in tqdm(range(len(file['data']))):
        temp=[]
        for j in range(len(file['data'][i])):
            temp.append(generate_angles(file['data'][i][j]))
        angles.append(temp)
        
    for i in tqdm(range(len(angles))):
        if len(angles[i])>=10:
            if i==0:
                data=batch(angles[i])
            else:
                data=np.concatenate((data,batch(angles[i])))
    np.save(path+'.npy',data)
    print('Array saved at '+path+'.npy')
