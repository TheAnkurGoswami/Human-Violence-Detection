from __future__ import absolute_import
import numpy as np
from deepsort import linear_assignment


def iou(bbox,candidates):
    #Computer intersection over union.

    #Top left, Bottom right coordinates of bounding box.
    bbox_tl,bbox_br=bbox[:2],bbox[:2]+bbox[2:] 
     #Top left, Bottom right coordinates of Candidate boxes.
    candidates_tl,candidates_br=candidates[:,:2],candidates[:,:2]+candidates[:,2:]

    tl=np.c_[np.maximum(bbox_tl[0],candidates_tl[:,0])[:,np.newaxis],
             np.maximum(bbox_tl[1],candidates_tl[:,1])[:,np.newaxis]]
    br=np.c_[np.minimum(bbox_br[0],candidates_br[:,0])[:,np.newaxis],
             np.minimum(bbox_br[1],candidates_br[:,1])[:,np.newaxis]]
    wh=np.maximum(0.,br-tl)

    area_intersection=wh.prod(axis=1)     #Area of intersection b/w bounding box and a candidate box.
    area_bbox=bbox[2:].prod()     #Area of bounding box.
    area_candidates=candidates[:,2:].prod(axis=1)       #Area of candidate boxes.
    return area_intersection/(area_bbox+area_candidates-area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
