"""
LaneNet model post process
"""

import cv2
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closing


def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """
        """
        self.dbscan_eps = dbscan_eps
        self.postprocess_min_samples = postprocess_min_samples

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        from sklearn.cluster import MeanShift

        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.postprocess_min_samples)
        # db = MeanShift()
        try:
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err:
            # print(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels)

        num_clusters = len(unique_labels)
        # cluster_centers = db.components_

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels,
            # 'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255)
        lane_embedding_feats = instance_seg_ret[idx]
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0]

        ret = {
            'lane_embedding_feats': lane_embedding_feats,
            'lane_coordinates': lane_coordinate
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )


        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1]], dtype=np.int)
        coords = get_lane_embedding_feats_result['lane_coordinates']
        
        for coord in coords:
            label = instance_seg_result[coord[1], coord[0]]
            mask[coord[1], coord[0]] = label + 1

        return mask, coords

class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, dbscan_eps=0.35, postprocess_min_samples=200):
        """

        :param ipm_remap_file_path: ipm generate file path
        """

        self._cluster = _LaneNetCluster(dbscan_eps, postprocess_min_samples)

    def postprocess(self, seg_result, min_area_threshold=100):
        """
        :param seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        for i in range(seg_result.shape[0]):
            seg_result[i,:,:] *= i
        instance_seg_result = np.zeros((seg_result.shape[1:])).astype('uint8') 
        for i in range(1,seg_result.shape[0]):
            instance_seg_result += seg_result[i,:,:].astype('uint8') 

        binary_seg_result = instance_seg_result!=0
        # convert binary_seg_result
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1]
        stats = connect_components_analysis_ret[2]
        for index, stat in enumerate(stats):
            if stat[4] <= min_area_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        # apply embedding features cluster
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )
        return mask_image, lane_coords