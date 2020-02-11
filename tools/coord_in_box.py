import numpy as np

def coordinates_in_box(box,coord):     #box & values of ith-person dict
	count=0
	total=len(coord)
	if total>0:
		for i in coord:
			# print(box,i)
			if i[0]==-1 and i[1]==-1:
				total-=1
				continue
			
			if box[0]<i[0]<box[2] and box[1]<i[1]<box[3]:
				count+=1
		if count/total>0.7:
			return True
		
	return False


def bbox_to_fig_ratio(bbox,coord):
    coord=np.array(coord)
    fig_max_y=coord.max(axis=0)[1]
    fig_min_y=coord.min(axis=0)[1]
    dst_fig=fig_max_y-fig_min_y
    
    bbox_max_y=bbox[3]
    bbox_min_y=bbox[1]
    dst_bbox=bbox_max_y-bbox_min_y
    
    if dst_fig/dst_bbox>0.7:
        return True
    
    return False
    