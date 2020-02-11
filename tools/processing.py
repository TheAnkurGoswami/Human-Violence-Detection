import math
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2

from tools import utils

COCO_BODY_PARTS=['nose','neck',
                 'right_shoulder','right_elbow','right_wrist',
                 'left_shoulder','left_elbow','left_wrist',
                 'right_hip','right_knee','right_ankle',
                 'left_hip','left_knee','left_ankle',
                 'right_eye','left_eye','right_ear','left_ear','background'
                   ]


def extract_parts(input_image,params,model,model_params):
    multiplier=[x*model_params['boxsize']/input_image.shape[0] for x in params['scale_search']]

    # Body parts location heatmap, one per part (19)
    heatmap_avg=np.zeros((input_image.shape[0],input_image.shape[1],19))
    # Part affinities, one per limb (38)
    paf_avg=np.zeros((input_image.shape[0],input_image.shape[1],38))
    # start=time.time()

    for scale in multiplier:
        image_to_test=cv2.resize(input_image,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        image_to_test_padded,pad=utils.pad_right_down_corner(image_to_test,model_params['stride'],
                                                               model_params['padValue'])

        # required shape (1, width, height, channels)
        input_img=np.transpose(np.float32(image_to_test_padded[:,:,:,np.newaxis]),(3,0,1,2))
        # start1=time.time()
        output_blobs=model.predict(input_img)
        # extract outputs, resize, and remove padding

        heatmap=np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap=cv2.resize(heatmap,(0,0),fx=model_params['stride'],fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap=heatmap[:image_to_test_padded.shape[0]-pad[2],:image_to_test_padded.shape[1]-pad[3],:]
        heatmap=cv2.resize(heatmap,(input_image.shape[1],input_image.shape[0]),interpolation=cv2.INTER_CUBIC)

        paf=np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf=cv2.resize(paf,(0,0),fx=model_params['stride'],fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf=paf[:image_to_test_padded.shape[0]-pad[2],:image_to_test_padded.shape[1]-pad[3],:]
        paf=cv2.resize(paf,(input_image.shape[1],input_image.shape[0]),interpolation=cv2.INTER_CUBIC)
        
        heatmap_avg=heatmap_avg+heatmap
        paf_avg=paf_avg+paf
        # print('Net took {} seconds'.format(time.time()-start1))

    # 'Loop 1 took {} seconds'.format(time.time()-start))
    heatmap_avg=heatmap_avg/len(multiplier)
    paf_avg=paf_avg/len(multiplier)
    

    all_peaks=[]
    peak_counter=0
    # start=time.time()
    for part in range(18):
        hmap_ori=heatmap_avg[:,:,part]
        hmap=gaussian_filter(hmap_ori,sigma=3)

        # Find the pixel that has maximum value compared to those around it
        hmap_left=np.zeros(hmap.shape)
        hmap_left[1:,:]=hmap[:-1,:]
        hmap_right=np.zeros(hmap.shape)
        hmap_right[:-1,:]=hmap[1:,:]
        hmap_up=np.zeros(hmap.shape)
        hmap_up[:,1:]=hmap[:,:-1]
        hmap_down=np.zeros(hmap.shape)
        hmap_down[:,:-1]=hmap[:,1:]

        # reduce needed because there are > 2 arguments
        peaks_binary=np.logical_and.reduce(
            (hmap>=hmap_left,hmap>=hmap_right,hmap>=hmap_up,hmap>=hmap_down,hmap>params['thre1']))
        peaks=list(zip(np.nonzero(peaks_binary)[1],np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score=[x+(hmap_ori[x[1],x[0]],) for x in peaks]  # add a third element to tuple with score
        idx=range(peak_counter,peak_counter+len(peaks))
        peaks_with_score_and_id=[peaks_with_score[i]+(idx[i],) for i in range(len(idx))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter+=len(peaks)


    # print('Loop 2 took {} seconds'.format(time.time()-start))

    connection_all=[]
    special_k=[]
    mid_num=10

    # start=time.time()
    for k in range(len(utils.hmapIdx)):
        score_mid=paf_avg[:,:,[x-19 for x in utils.hmapIdx[k]]]
        cand_a=all_peaks[utils.limbSeq[k][0]-1]
        cand_b=all_peaks[utils.limbSeq[k][1]-1]
        n_a=len(cand_a)
        n_b=len(cand_b)
        if n_a!=0 and n_b!=0:
            connection_candidate=[]
            for i in range(n_a):
                for j in range(n_b):
                    vec=np.subtract(cand_b[j][:2],cand_a[i][:2])
                    norm=math.sqrt(vec[0]*vec[0]+vec[1]*vec[1])
                    # failure case when 2 body parts overlaps
                    if norm==0:
                        continue
                    vec=np.divide(vec,norm)

                    startend=list(zip(np.linspace(cand_a[i][0],cand_b[j][0],num=mid_num),
                                        np.linspace(cand_a[i][1], cand_b[j][1],num=mid_num)))

                    vec_x=np.array(
                        [score_mid[int(round(startend[I][1])),int(round(startend[I][0])),0]
                         for I in range(len(startend))])
                    vec_y=np.array(
                        [score_mid[int(round(startend[I][1])),int(round(startend[I][0])),1]
                         for I in range(len(startend))])

                    score_midpts=np.multiply(vec_x,vec[0])+np.multiply(vec_y,vec[1])
                    score_with_dist_prior=sum(score_midpts)/len(score_midpts)+ min(0.5*input_image.shape[0]/norm-1,0)

                    criterion1=len(np.nonzero(score_midpts>params['thre2'])[0])>0.8*len(score_midpts)
                    criterion2=score_with_dist_prior>0

                    if criterion1 and criterion2:
                        connection_candidate.append([i,j,score_with_dist_prior,
                                                     score_with_dist_prior+cand_a[i][2]+cand_b[j][2]])

            connection_candidate=sorted(connection_candidate,key=lambda x: x[2],reverse=True)
            connection=np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s=connection_candidate[c][0:3]
                if i not in connection[:,3] and j not in connection[:,4]:
                    connection=np.vstack([connection,[cand_a[i][3],cand_b[j][3],s,i,j]])
                    if len(connection)>=min(n_a,n_b):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    # print('Loop 3 took {} seconds'.format(time.time()-start))
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset=np.empty((0,20))
    candidate=np.array([item for sublist in all_peaks for item in sublist])
    # start=time.time()
    for k in range(len(utils.hmapIdx)):
        if k not in special_k:
            part_as=connection_all[k][:,0]
            part_bs=connection_all[k][:,1]
            index_a,index_b=np.array(utils.limbSeq[k])-1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found=0
                subset_idx=[-1,-1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][index_a]==part_as[i] or subset[j][index_b]==part_bs[i]:
                        subset_idx[found]=j
                        found+=1

                if found==1:
                    j=subset_idx[0]
                    if subset[j][index_b]!=part_bs[i]:
                        subset[j][index_b]=part_bs[i]
                        subset[j][-1]+=1
                        subset[j][-2]+=candidate[part_bs[i].astype(int),2]+connection_all[k][i][2]
                elif found==2:  # if found 2 and disjoint, merge them
                    j1,j2=subset_idx
                    membership=((subset[j1]>=0).astype(int)+(subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership==2)[0])==0:  # merge
                        subset[j1][:-2]+=(subset[j2][:-2]+1)
                        subset[j1][-2:]+=subset[j2][-2:]
                        subset[j1][-2]+=connection_all[k][i][2]
                        subset=np.delete(subset,j2, 0)
                    else:  # as like found == 1
                        subset[j1][index_b]=part_bs[i]
                        subset[j1][-1]+=1
                        subset[j1][-2]+=candidate[part_bs[i].astype(int),2]+connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k<17:
                    row=-1*np.ones(20)
                    row[index_a]=part_as[i]
                    row[index_b]=part_bs[i]
                    row[-1]=2
                    row[-2]=sum(candidate[connection_all[k][i,:2].astype(int),2])+connection_all[k][i][2]
                    subset=np.vstack([subset,row])
    # print('Loop 4 took {} seconds'.format(time.time()-start))
    # delete some rows of subset which has few parts occur
    delete_idx=[]
    for i in range(len(subset)):
        if subset[i][-1]<4 or subset[i][-2]/subset[i][-1]<0.4:
            delete_idx.append(i)
    subset=np.delete(subset,delete_idx,axis=0)


    coord_id=[]
    for i in all_peaks:
        for j in i:
            coord_id.append(j)
    coord_id.append((-1,-1,-1,-1))
    coord_id=np.array(coord_id,'int64')[:,[0,1,3]]

    temp=coord_id[np.array(subset[:,:18],'int64'),:2]
    person_dict={}
    for i in range(temp.shape[0]):
        for j in range(18):
            if 'person'+str(i+1) not in person_dict:
                person_dict['person'+str(i+1)]={}

            person_dict['person'+str(i+1)][COCO_BODY_PARTS[j]]=temp[i,j,:]

    return person_dict

def non_max_suppression(boxes,max_bbox_overlap,scores=None):
 
    if len(boxes)==0:
        return []

    boxes=boxes.astype(np.float)
    pick=[]

    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]+boxes[:,0]
    y2=boxes[:,3]+boxes[:,1]

    area=(x2-x1+1)*(y2-y1+1)
    if scores is not None:
        idxs=np.argsort(scores)
    else:
        idxs=np.argsort(y2)

    while len(idxs)>0:
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)

        xx1=np.maximum(x1[i],x1[idxs[:last]])
        yy1=np.maximum(y1[i],y1[idxs[:last]])
        xx2=np.minimum(x2[i],x2[idxs[:last]])
        yy2=np.minimum(y2[i],y2[idxs[:last]])

        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)

        overlap=(w*h)/area[idxs[:last]]

        idxs=np.delete(
            idxs,np.concatenate(
                ([last],np.where(overlap>max_bbox_overlap)[0])))

    return pick