# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import os

from google.protobuf import text_format

all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    try:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, True)
    except RuntimeError as e:
        print(e)

from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2


object_type_to_id = {
    'TYPE_UNSET': 0,
    'TYPE_VEHICLE': 1,
    'TYPE_PEDESTRIAN': 2,
    'TYPE_CYCLIST': 3,
    'TYPE_OTHER': 4
}

def _default_metrics_config_sliding_window(eval_second, num_modes_for_eval=6, measurement_step=5,frams_to_track=30 ):
    assert eval_second in [1, 3, 4, 5, 8]
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 10
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    """
    # step_configurations {
    # measurement_step: 5
    # lateral_miss_threshold: 1.0
    # longitudinal_miss_threshold: 2.0
    # }
    config_text += f"""
    max_predictions: {num_modes_for_eval}
    """
    # putting the measurement_step equals to the last time step gives segmentation fault
    # this is only use for miss rate calculation
    if eval_second == 1:
        config_text += """
        track_future_samples: 10
        step_configurations {
        measurement_step: 3
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        """

    elif eval_second == 3:
        config_text += f"""
        track_future_samples: {frams_to_track}
        step_configurations {{
            measurement_step: {measurement_step}
            lateral_miss_threshold: 1.0
            longitudinal_miss_threshold: 2.0
        }}
        """
        # config_text += """
        # track_future_samples: 30
        # step_configurations {
        # measurement_step: 0
        # lateral_miss_threshold: 1.0
        # longitudinal_miss_threshold: 2.0
        # }
        # """
    elif eval_second == 4:
        config_text += """
        track_future_samples: 40
        step_configurations {
        measurement_step: 25
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        """
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 25
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 25
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        step_configurations {
        measurement_step: 45
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 75
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config


def _default_metrics_config(eval_second, num_modes_for_eval=6):
    assert eval_second in [1, 3, 4, 5, 8]
    config = motion_metrics_pb2.MotionMetricsConfig()
    config_text = """
    track_steps_per_second: 10
    prediction_steps_per_second: 2
    track_history_samples: 10
    speed_lower_bound: 1.4
    speed_upper_bound: 11.0
    speed_scale_lower: 0.5
    speed_scale_upper: 1.0
    """
    # step_configurations {
    # measurement_step: 5
    # lateral_miss_threshold: 1.0
    # longitudinal_miss_threshold: 2.0
    # }
    config_text += f"""
    max_predictions: {num_modes_for_eval}
    """
    # putting the measurement_step equals to the last time step gives segmentation fault
    # this is only use for miss rate calculation
    if eval_second == 1:
        config_text += """
        track_future_samples: 10
        step_configurations {
        measurement_step: 1
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        """

    elif eval_second == 3:
        config_text += """
        track_future_samples: 30
        step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        """
    elif eval_second == 4:
        config_text += """
        track_future_samples: 40
        step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        """
    elif eval_second == 5:
        config_text += """
        track_future_samples: 50
        step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        """
    else:
        config_text += """
        track_future_samples: 80
        step_configurations {
        measurement_step: 5
        lateral_miss_threshold: 1.0
        longitudinal_miss_threshold: 2.0
        }
        step_configurations {
        measurement_step: 9
        lateral_miss_threshold: 1.8
        longitudinal_miss_threshold: 3.6
        }
        step_configurations {
        measurement_step: 15
        lateral_miss_threshold: 3.0
        longitudinal_miss_threshold: 6.0
        }
        """

    text_format.Parse(config_text, config)
    return config


def transform_preds_to_waymo_format_custom(pred_dicts, top_k_for_eval=-1, eval_second=8):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f'Total number for evaluation (after processed): {len(pred_dicts)}')

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if  cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    print('number of future frames:', num_future_frames)

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    if num_future_frames in [10, 30, 50, 70]:
        sampled_interval = 1
        # sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    num_frame_to_eval = num_future_frames // sampled_interval

    if eval_second == 1:
        num_frames_in_total = 31
        num_frame_to_eval = 10

    elif eval_second == 3:
        num_frames_in_total = 51
        num_frame_to_eval = 30

    elif eval_second == 4:
        num_frames_in_total = 61
        num_frame_to_eval = 40

    elif eval_second == 5:
        num_frames_in_total = 71
        num_frame_to_eval = 50

    else:
        num_frames_in_total = 91
        num_frame_to_eval = 70
   
    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total-10, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total-10), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)


    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]

            cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()
            # batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 0::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][10:num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][10:num_frames_in_total, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict


def transform_preds_to_waymo_format(pred_dicts, top_k_for_eval=-1, eval_second=8):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f'Total number for evaluation (after processed): {len(pred_dicts)}')

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if  cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    print('number of future frames:', num_future_frames)

    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    if num_future_frames in [10, 30, 50, 80]:
        sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    num_frame_to_eval = num_future_frames // sampled_interval

     
    if eval_second == 1:
        num_frames_in_total = 21
        num_frame_to_eval = 2

    elif eval_second == 3:
        num_frames_in_total = 41
        num_frame_to_eval = 6

    elif eval_second == 4:
        num_frames_in_total = 51
        num_frame_to_eval = 8

    elif eval_second == 5:
        num_frames_in_total = 61
        num_frame_to_eval = 10

    else:
        num_frames_in_total = 91
        num_frame_to_eval = 16

    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)

    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]

            cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()
            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][:num_frames_in_total, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict

def waymo_evaluation(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6):

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = transform_preds_to_waymo_format(
        pred_dicts, top_k_for_eval=top_k, eval_second=eval_second,
    )

    eval_config = _default_metrics_config(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval)


    pred_score = tf.convert_to_tensor(pred_score, np.float32)
    pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
    gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
    gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], np.bool)
    pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], np.bool)
    object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)

    

    metric_results = py_metrics_ops.motion_metrics(
        config=eval_config.SerializeToString(),
        prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
        prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
        ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
        ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
        prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
        prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, num_agents_per_group)
        object_type=object_type  # (batch_size, num_total_agents)
    )

    metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
    # print('metric_names:', metric_names)

    result_dict = {}
    avg_results = {}
    for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']):
        avg_results.update({
            f'{m} - VEHICLE': [0.0, 0], f'{m} - PEDESTRIAN': [0.0, 0], f'{m} - CYCLIST': [0.0, 0]
        })
        for j, n in enumerate(metric_names):
            cur_name = n.split('_')[1]
            avg_results[f'{m} - {cur_name}'][0] += float(metric_results[i][j])
            avg_results[f'{m} - {cur_name}'][1] += 1
            result_dict[f'{m} - {n}\t'] = float(metric_results[i][j])

    for key in avg_results:
        avg_results[key] = avg_results[key][0] / avg_results[key][1]

    result_dict['-------------------------------------------------------------'] = 0
    result_dict.update(avg_results)

    final_avg_results = {}
    result_format_list = [
        ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', '\n'],
        ['VEHICLE', None, None, None, None, '\n'],
        ['PEDESTRIAN', None, None, None, None, '\n'],
        ['CYCLIST', None, None, None, None, '\n'],
        ['Avg', None, None, None, None, '\n'],
    ]
    name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
    name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4}

    for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP']:
        final_avg_results[cur_metric_name] = 0
        for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
            final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name} - {cur_name}']

            result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = '%.4f,' % avg_results[f'{cur_metric_name} - {cur_name}']

        final_avg_results[cur_metric_name] /= 3
        result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

    result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])

    result_dict['--------------------------------------------------------------'] = 0
    result_dict.update(final_avg_results)
    result_dict['---------------------------------------------------------------'] = 0
    result_dict.update(object_type_cnt_dict)
    result_dict['-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'] = 0

    return result_dict, result_format_str



def waymo_evaluation_custom(pred_dicts, top_k=-1, eval_second=8, num_modes_for_eval=6):

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict = transform_preds_to_waymo_format_custom(
        pred_dicts, top_k_for_eval=top_k, eval_second=eval_second,
    )

    minADE=[]
    minFDE=[]
    mAP=[]
    missRate=[]
    for i in range(0,30):

        eval_config = _default_metrics_config_custom(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval, measurement_step=i)

        pred_score = tf.convert_to_tensor(pred_score, np.float32)
        pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
        gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
        gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], np.bool)
        pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
        pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], np.bool)
        object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)
        # print(pred_trajs)
        # print(gt_trajs[:,:,21:,:2])

        # exit()

        metric_results = py_metrics_ops.motion_metrics(
            config=eval_config.SerializeToString(),
            prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
            prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
            ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
            ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
            prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
            prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, num_agents_per_group)
            object_type=object_type  # (batch_size, num_total_agents)
        )

        metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
        # print('metric_names:', metric_names)

        result_dict = {}
        avg_results = {}
        for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']):
            avg_results.update({
                f'{m} - VEHICLE': [0.0, 0], f'{m} - PEDESTRIAN': [0.0, 0], f'{m} - CYCLIST': [0.0, 0]
            })
            for j, n in enumerate(metric_names):
                cur_name = n.split('_')[1]
                avg_results[f'{m} - {cur_name}'][0] += float(metric_results[i][j])
                avg_results[f'{m} - {cur_name}'][1] += 1
                result_dict[f'{m} - {n}\t'] = float(metric_results[i][j])

        for key in avg_results:
            avg_results[key] = avg_results[key][0] / avg_results[key][1]

        
        minADE.append(avg_results['minADE - VEHICLE'])
        minFDE.append(avg_results['minFDE - VEHICLE'])
        mAP.append(avg_results['mAP - VEHICLE'])
        missRate.append(avg_results['MissRate - VEHICLE'])

        # print('avg_results:', avg_results['minADE - VEHICLE'])
        # print('avg_results:', avg_results['minFDE - VEHICLE'])
        # print('avg_results:', avg_results['MissRate - VEHICLE'])
        # print('avg_results:', avg_results['OverlapRate - VEHICLE'])
        # print('avg_results:', avg_results['mAP - VEHICLE'])
        # result_dict['-------------------------------------------------------------'] = 0
        # result_dict.update(avg_results)

        # final_avg_results = {}
        # result_format_list = [
        #     ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', '\n'],
        #     ['VEHICLE', None, None, None, None, '\n'],
        #     ['PEDESTRIAN', None, None, None, None, '\n'],
        #     ['CYCLIST', None, None, None, None, '\n'],
        #     ['Avg', None, None, None, None, '\n'],
        # ]
        # name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
        # name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4}

        # for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP']:
        #     final_avg_results[cur_metric_name] = 0
        #     for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
        #         final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name} - {cur_name}']

        #         result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = '%.4f,' % avg_results[f'{cur_metric_name} - {cur_name}']

        #     final_avg_results[cur_metric_name] /= 3
        #     result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

        # result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])

        # result_dict['--------------------------------------------------------------'] = 0
        # result_dict.update(final_avg_results)
        # result_dict['---------------------------------------------------------------'] = 0
        # result_dict.update(object_type_cnt_dict)
        # result_dict['-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'] = 0

        # return result_dict, result_format_str
    return mAP, minADE, minFDE, missRate



def compute_stats_per_timestep(fde_min, best_scores, final_mask):
    """
    Compute statistics (mean, median, std, quartiles) per timestep for valid entries.
    
    Args:
        fde_min: Tensor of shape (B, N, T) — minimum FDE values per batch, object, time
        final_mask: Boolean Tensor (B, N, T) — mask of valid entries to include
        
    Returns:
        Dict of Tensors each shape (T,): 'mean', 'median', 'std', 'q1', 'q3'
    """
    B, N, T = fde_min.shape

    mean_list = []
    mean_list_scores =[]
    median_list = []
    std_list = []
    q1_list = []
    q3_list = []

    for t in range(T):
        # Masked values at timestep t
        fde_t = tf.boolean_mask(fde_min[:, :, t], final_mask[:, :, t])  # shape (?,)
        best_scores_t = tf.boolean_mask(best_scores[:, :, t], final_mask[:, :, t])
        # fde_t = fde_min[:, :, t]
        if tf.size(fde_t) > 0:
            mean_list.append(tf.reduce_mean(fde_t))
            mean_list_scores.append(tf.reduce_mean(best_scores_t))
            median_list.append(tfp.stats.percentile(fde_t, 50.0))
            std_list.append(tf.math.reduce_std(fde_t))
            q1_list.append(tfp.stats.percentile(fde_t, 25.0))
            q3_list.append(tfp.stats.percentile(fde_t, 75.0))
        else:
            # No valid data at this timestep, use NaN as placeholder
            mean_list.append(tf.constant(float('nan'), dtype=fde_min.dtype))
            mean_list_scores.append(tf.constant(float('nan'), dtype=fde_min.dtype))
            median_list.append(tf.constant(float('nan'), dtype=fde_min.dtype))
            std_list.append(tf.constant(float('nan'), dtype=fde_min.dtype))
            q1_list.append(tf.constant(float('nan'), dtype=fde_min.dtype))
            q3_list.append(tf.constant(float('nan'), dtype=fde_min.dtype))

    return {
        'mean_fde': tf.stack(mean_list),     # shape (T,)
        'mean_scores': tf.stack(mean_list_scores),     # shape (T,)
        'median_fde': tf.stack(median_list), # shape (T,)
        'std_fde': tf.stack(std_list),       # shape (T,)
        'q1_fde': tf.stack(q1_list),         # shape (T,)
        'q3_fde': tf.stack(q3_list)          # shape (T,)
    }


def convert_tensors(obj):
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist() if obj.shape else obj.numpy().item()
    if isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_tensors(x) for x in obj]
    return obj

def compute_fde_statistics_tf(pred_trajs, gt_trajs, gt_valid, pred_scores, object_types):
    """
    Args:
        pred_trajs: (B, N, M, 1, T, D)  — predicted trajectories
        gt_trajs: (B, N, T, 7)          — ground truth trajectories (last dim includes x, y)
        gt_valid: (B, N, T)             — ground truth validity mask (1 = valid, 0 = invalid)
        pred_scores: (B, N, M)          — prediction scores for each mode
        object_types: (B, N)            — type of each object
    Returns:
        stats_by_type: Dict of stats for each type: {'mean': [...], 'median': [...], etc.}
    """

    B, N, M, _, T, D = pred_trajs.shape

    # Only x, y from gt
    gt_xy = gt_trajs[:, :, 11:, :2]  # (B, N, T, 2)
    gt_valid = gt_valid[:,:,11:]

    # Expand ground truth for broadcasting
    gt_xy_exp = tf.expand_dims(gt_xy, axis=2)  # (B, N, 1, T, 2)
    # print("pred_trajs:", pred_trajs[..., 0, :, :].shape)  # Expect (B, N, M, T, 2)
    # print("gt_xy_exp:", gt_xy_exp.shape) 

    # Compute L2 distance
    dist = tf.norm(pred_trajs[..., 0, :, :] - gt_xy_exp, axis=-1)  # (B, N, M, T)

    # Broadcast gt_valid
    gt_valid_exp = tf.expand_dims(gt_valid, axis=2)  # (B, N, 1, T)

    # Mask invalid timestamps
    # dist = tf.where(tf.equal(gt_valid_exp, 1), dist, tf.constant(float('inf'), dtype=dist.dtype))
    # dist = tf.where(gt_valid_exp, dist, tf.constant(float('inf'), dtype=dist.dtype))
    dist = tf.where(gt_valid_exp, dist, tf.constant(0.0, dtype=dist.dtype))


    # Minimum over modes
    fde_min = tf.reduce_min(dist, axis=2)  # (B, N, T)
    best_mode = tf.argmin(dist, axis=2)    # (B, N, T)

    # print(pred_scores)
    # print(best_mode)

    # print(fde_min)
    # exit()

    # Get score for best mode
    pred_scores_exp = tf.expand_dims(pred_scores, axis=-1)  # (B, N, M, 1)
    pred_scores_tiled = tf.tile(pred_scores_exp, [1, 1, 1, T])  # (B, N, M, T)

    # Gather scores for best mode
    batch_indices = tf.range(B)[:, None, None]
    obj_indices = tf.range(N)[None, :, None]
    time_indices = tf.range(T)[None, None, :]

    batch_indices = tf.tile(batch_indices, [1, N, T])  # (B, N, T)
    obj_indices = tf.tile(obj_indices, [B, 1, T])
    time_indices = tf.tile(time_indices, [B, N, 1])

    # to stack all must be same unit
    batch_indices = tf.cast(batch_indices, tf.int32)
    obj_indices = tf.cast(obj_indices, tf.int32)
    best_mode = tf.cast(best_mode, tf.int32)
    time_indices = tf.cast(time_indices, tf.int32)

    indices = tf.stack([batch_indices, obj_indices, best_mode, time_indices], axis=-1)
    best_scores = tf.gather_nd(pred_scores_tiled, indices)  # (B, N, T)

    # print (best_scores)

    # Mask for valid fde
    valid_mask = tf.logical_and(gt_valid ,tf.expand_dims(object_types > 0, axis=-1) )  # (B, N, T)
    
    # Compute stats by object type
    stats_by_type = {}
    # for obj_type in [1, 2, 3, 4]:

    for obj_type in [1]: #1 == vehicle
        # Mask for current type
        # Create a mask where object type > 0 → (B, N, 1)
        type_mask = tf.expand_dims(object_types==obj_type, axis=-1)  # (B, N, 1)
        # Broadcast it to match gt_valid shape → (B, N, T)
        type_mask = tf.broadcast_to(type_mask, tf.shape(gt_valid))  # (B, N, T)
        # Final mask = valid points AND correct object type
        final_mask = tf.logical_and(gt_valid, type_mask)  # (B, N, T)

        # Filter values
        # fde_vals = tf.boolean_mask(fde_min, final_mask)
        # fde_vals = tf.boolean_mask(fde_min, valid_mask)

        stats = compute_stats_per_timestep(fde_min, best_scores, final_mask)
        stats_by_type[obj_type] = stats

        # print("Stats by type:", stats )
        # exit()

    return stats_by_type


def transform_preds_to_waymo_format_sliding_window(pred_dicts, top_k_for_eval=-1, eval_second=8, current_time_stamp=10):
    print(f'Total number for evaluation (intput): {len(pred_dicts)}')
    temp_pred_dicts = []
    for k in range(len(pred_dicts)):
        if isinstance(pred_dicts[k], list):
            temp_pred_dicts.extend(pred_dicts[k])
        else:
            temp_pred_dicts.append(pred_dicts[k])
    pred_dicts = temp_pred_dicts
    print(f'Total number for evaluation (after processed): {len(pred_dicts)}')

    scene2preds = {}
    num_max_objs_per_scene = 0
    for k in range(len(pred_dicts)):
        cur_scenario_id = pred_dicts[k]['scenario_id']
        if  cur_scenario_id not in scene2preds:
            scene2preds[cur_scenario_id] = []
        scene2preds[cur_scenario_id].append(pred_dicts[k])
        num_max_objs_per_scene = max(num_max_objs_per_scene, len(scene2preds[cur_scenario_id]))
    num_scenario = len(scene2preds)
    topK, num_future_frames, _ = pred_dicts[0]['pred_trajs'].shape

    # for key, value_list in pred_dicts[0].items():
    #     print(f"{key}: {value_list.shape}")

    valid_gt = pred_dicts[0]['valid_gt']

    # print (valid_gt)

    # exit()




    if top_k_for_eval != -1:
        topK = min(top_k_for_eval, topK)

    sampled_interval = 1
    # if num_future_frames in [10, 30, 50, 70, 80]:
    #     sampled_interval = 1
        # sampled_interval = 5
    assert num_future_frames % sampled_interval == 0, f'num_future_frames={num_future_frames}'
    # num_frame_to_eval = num_future_frames // sampled_interval

    # if eval_second == 1:
    #     num_frames_in_total = 31
    #     num_frame_to_eval = 10

    # elif eval_second == 3:
    #     num_frames_in_total = 51
    #     num_frame_to_eval = 30

    # elif eval_second == 4:
    #     num_frames_in_total = 61
    #     num_frame_to_eval = 40

    # elif eval_second == 5:
    #     num_frames_in_total = 71
    #     num_frame_to_eval = 50

    # else:
    #     num_frames_in_total = 91
    #     num_frame_to_eval = 70

    if num_future_frames==30:
        num_frames_in_total = valid_gt+11
        num_frame_to_eval = valid_gt
        if current_time_stamp+31 <= 91:
            max_gt_frames = current_time_stamp + 31
        else:
            max_gt_frames = 91

    if num_future_frames==80:
        num_frames_in_total = valid_gt+11
        num_frame_to_eval = valid_gt
        max_gt_frames=91

    elif num_future_frames==70:
        if current_time_stamp >=20:
            num_frames_in_total = valid_gt+11
            num_frame_to_eval = valid_gt
            max_gt_frames=91
        else:
            # remove the gt above 70 as prediction are only 70
            valid_gt = 70
    
            num_frames_in_total = valid_gt+11
            num_frame_to_eval = valid_gt
            max_gt_frames=91 - (20 - current_time_stamp)

            # print('start index', max_gt_frames-num_frames_in_total)
            # print('end index: ', max_gt_frames)

            # exit()

   
    batch_pred_trajs = np.zeros((num_scenario, num_max_objs_per_scene, topK, 1, num_frame_to_eval, 2))
    batch_pred_scores = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    batch_pred_scores_wo_normalization = np.zeros((num_scenario, num_max_objs_per_scene, topK))
    gt_trajs = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total, 7))
    gt_is_valid = np.zeros((num_scenario, num_max_objs_per_scene, num_frames_in_total), dtype=np.int)
    pred_gt_idxs = np.zeros((num_scenario, num_max_objs_per_scene, 1))
    pred_gt_idx_valid_mask = np.zeros((num_scenario, num_max_objs_per_scene, 1), dtype=np.int)
    object_type = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.object)
    object_id = np.zeros((num_scenario, num_max_objs_per_scene), dtype=np.int)
    scenario_id = np.zeros((num_scenario), dtype=np.object)


    


    object_type_cnt_dict = {}
    for key in object_type_to_id.keys():
        object_type_cnt_dict[key] = 0

    for scene_idx, val in enumerate(scene2preds.items()):
        cur_scenario_id, preds_per_scene = val
        scenario_id[scene_idx] = cur_scenario_id
        for obj_idx, cur_pred in enumerate(preds_per_scene):
            # print(cur_pred['pred_scores'])   

            sort_idxs = cur_pred['pred_scores'].argsort()[::-1]
            cur_pred['pred_scores'] = cur_pred['pred_scores'][sort_idxs]
            # print('after sort:', cur_pred['pred_scores'])
            cur_pred['pred_trajs'] = cur_pred['pred_trajs'][sort_idxs]

            batch_pred_scores_wo_normalization[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]

            cur_pred['pred_scores'] = cur_pred['pred_scores'] / cur_pred['pred_scores'].sum()
            # batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 4::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_trajs[scene_idx, obj_idx] = cur_pred['pred_trajs'][:topK, np.newaxis, 0::sampled_interval, :][:, :, :num_frame_to_eval, :]
            batch_pred_scores[scene_idx, obj_idx] = cur_pred['pred_scores'][:topK]
            gt_trajs[scene_idx, obj_idx] = cur_pred['gt_trajs'][current_time_stamp-10:max_gt_frames, [0, 1, 3, 4, 6, 7, 8]]  # (num_timestamps_in_total, 10), [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
            gt_is_valid[scene_idx, obj_idx] = cur_pred['gt_trajs'][current_time_stamp-10:max_gt_frames, -1]
            pred_gt_idxs[scene_idx, obj_idx, 0] = obj_idx
            pred_gt_idx_valid_mask[scene_idx, obj_idx, 0] = 1
            object_type[scene_idx, obj_idx] = object_type_to_id[cur_pred['object_type']]
            object_id[scene_idx, obj_idx] = cur_pred['object_id']

            object_type_cnt_dict[cur_pred['object_type']] += 1

            # print(cur_pred['pred_trajs'].shape)
            # print(gt_trajs[:,:, 11:, :2].shape)
            # print(batch_pred_trajs.shape)
            # print(cur_pred['pred_scores'])   
            # print(batch_pred_scores)
            # exit()
            # print(cur_pred['pred_trajs'][:,-1,:])
            # print(cur_pred['gt_trajs'][ -1, :2])
            # print(batch_pred_trajs)


        # break

    gt_infos = {
        'scenario_id': scenario_id.tolist(),
        'object_id': object_id.tolist(),
        'object_type': object_type.tolist(),
        'gt_is_valid': gt_is_valid,
        'gt_trajectory': gt_trajs,
        'pred_gt_indices': pred_gt_idxs,
        'pred_gt_indices_mask': pred_gt_idx_valid_mask
    }
    return batch_pred_scores, batch_pred_trajs, gt_infos, object_type_cnt_dict ,num_future_frames, batch_pred_scores_wo_normalization


def waymo_evaluation_sliding_window(pred_dicts, top_k=-1, eval_second=8, current_time_stamp=10, num_modes_for_eval=6):

    # print("here")

    pred_score, pred_trajectory, gt_infos, object_type_cnt_dict, num_future_frames, pred_scores_wo_normalization  = transform_preds_to_waymo_format_sliding_window(
        pred_dicts, top_k_for_eval=top_k, eval_second=eval_second, current_time_stamp=current_time_stamp
    )

    # print("here2")

    minADE=[]
    minFDE=[]
    mAP=[]
    missRate=[]
    confidenece=[]


    pred_score = tf.convert_to_tensor(pred_score, np.float32)
    pred_trajs = tf.convert_to_tensor(pred_trajectory, np.float32)
    gt_trajs = tf.convert_to_tensor(gt_infos['gt_trajectory'], np.float32)
    gt_is_valid = tf.convert_to_tensor(gt_infos['gt_is_valid'], np.bool)
    pred_gt_indices = tf.convert_to_tensor(gt_infos['pred_gt_indices'], tf.int64)
    pred_gt_indices_mask = tf.convert_to_tensor(gt_infos['pred_gt_indices_mask'], np.bool)
    object_type = tf.convert_to_tensor(gt_infos['object_type'], tf.int64)


    # non_zero_mean = pred_scores_wo_normalization[pred_scores_wo_normalization != 0].mean()
    # confidenece.append(non_zero_mean)

    # start = 0 
    # end = 90
    # if num_future_frames==30:
    #     if end - current_time_stamp > 30:
    #         end = 30+current_time_stamp
        

    # if num_future_frames==70 and current_time_stamp <20:
    #     end = 70+current_time_stamp


    # for i in range(start,   end-current_time_stamp):

    #     # break
    #     if num_future_frames==70 and current_time_stamp <20:
    #         frames_to_track = end
    #     else:
    #         frames_to_track = end-current_time_stamp


    #     eval_config = _default_metrics_config_sliding_window(eval_second=eval_second, num_modes_for_eval=num_modes_for_eval, measurement_step=i, frams_to_track=end-current_time_stamp)

    #     metric_results = py_metrics_ops.motion_metrics(
    #         config=eval_config.SerializeToString(),
    #         prediction_trajectory=pred_trajs,  # (batch_size, num_pred_groups, top_k, num_agents_per_group, num_pred_steps, )
    #         prediction_score=pred_score,  # (batch_size, num_pred_groups, top_k)
    #         ground_truth_trajectory=gt_trajs,  # (batch_size, num_total_agents, num_gt_steps, 7)
    #         ground_truth_is_valid=gt_is_valid,  # (batch_size, num_total_agents, num_gt_steps)
    #         prediction_ground_truth_indices=pred_gt_indices,  # (batch_size, num_pred_groups, num_agents_per_group)
    #         prediction_ground_truth_indices_mask=pred_gt_indices_mask,  # (batch_size, num_pred_groups, num_agents_per_group)
    #         object_type=object_type  # (batch_size, num_total_agents)
    #     )


    #     metric_names = config_util.get_breakdown_names_from_motion_config(eval_config)
    #     # print('metric_names:', metric_names)

    #     result_dict = {}
    #     avg_results = {}
    #     for i, m in enumerate(['minADE', 'minFDE', 'MissRate', 'OverlapRate', 'mAP']):
    #         avg_results.update({
    #             f'{m} - VEHICLE': [0.0, 0], f'{m} - PEDESTRIAN': [0.0, 0], f'{m} - CYCLIST': [0.0, 0]
    #         })
    #         for j, n in enumerate(metric_names):
    #             cur_name = n.split('_')[1]
    #             avg_results[f'{m} - {cur_name}'][0] += float(metric_results[i][j])
    #             avg_results[f'{m} - {cur_name}'][1] += 1
    #             result_dict[f'{m} - {n}\t'] = float(metric_results[i][j])

    #     for key in avg_results:
    #         avg_results[key] = avg_results[key][0] / avg_results[key][1]

        
    #     minADE.append(avg_results['minADE - VEHICLE'])
    #     minFDE.append(avg_results['minFDE - VEHICLE'])
    #     mAP.append(avg_results['mAP - VEHICLE'])
    #     missRate.append(avg_results['MissRate - VEHICLE'])
        

    #     # print('avg_results:', avg_results['minADE - VEHICLE'])
    #     # print('avg_results:', avg_results['minFDE - VEHICLE'])
    #     # print('avg_results:', avg_results['MissRate - VEHICLE'])
    #     # print('avg_results:', avg_results['OverlapRate - VEHICLE'])
    #     # print('avg_results:', avg_results['mAP - VEHICLE'])
    #     # result_dict['-------------------------------------------------------------'] = 0
    #     # result_dict.update(avg_results)

    #     # final_avg_results = {}
    #     # result_format_list = [
    #     #     ['Waymo', 'mAP', 'minADE', 'minFDE', 'MissRate', '\n'],
    #     #     ['VEHICLE', None, None, None, None, '\n'],
    #     #     ['PEDESTRIAN', None, None, None, None, '\n'],
    #     #     ['CYCLIST', None, None, None, None, '\n'],
    #     #     ['Avg', None, None, None, None, '\n'],
    #     # ]
    #     # name_to_row = {'VEHICLE': 1, 'PEDESTRIAN': 2, 'CYCLIST': 3, 'Avg': 4}
    #     # name_to_col = {'mAP': 1, 'minADE': 2, 'minFDE': 3, 'MissRate': 4}

    #     # for cur_metric_name in ['minADE', 'minFDE', 'MissRate', 'mAP']:
    #     #     final_avg_results[cur_metric_name] = 0
    #     #     for cur_name in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']:
    #     #         final_avg_results[cur_metric_name] += avg_results[f'{cur_metric_name} - {cur_name}']

    #     #         result_format_list[name_to_row[cur_name]][name_to_col[cur_metric_name]] = '%.4f,' % avg_results[f'{cur_metric_name} - {cur_name}']

    #     #     final_avg_results[cur_metric_name] /= 3
    #     #     result_format_list[4][name_to_col[cur_metric_name]] = '%.4f,' % final_avg_results[cur_metric_name]

    #     # result_format_str = ' '.join([x.rjust(12) for items in result_format_list for x in items])

    #     # result_dict['--------------------------------------------------------------'] = 0
    #     # result_dict.update(final_avg_results)
    #     # result_dict['---------------------------------------------------------------'] = 0
    #     # result_dict.update(object_type_cnt_dict)
    #     # result_dict['-----Note that this evaluation may have marginal differences with the official Waymo evaluation server-----'] = 0

    #     # return result_dict, result_format_str
        
    # print(minFDE)
    stats= compute_fde_statistics_tf(pred_trajectory, gt_trajs, gt_is_valid, pred_scores_wo_normalization, object_type)

    stats = convert_tensors(stats)  


    return mAP, minADE, minFDE, missRate, confidenece, stats


def main():
    import pickle
    import argparse
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file')
    parser.add_argument('--top_k', type=int, default=-1, help='')
    parser.add_argument('--eval_second', type=int, default=8, help='')
    parser.add_argument('--num_modes_for_eval', type=int, default=6, help='')

    args = parser.parse_args()
    print(args)

    assert args.eval_second in [3, 5, 8]
    pred_infos = pickle.load(open(args.pred_infos, 'rb'))

    result_format_str = ''
    print('Start to evaluate the waymo format results...')

    metric_results, result_format_str = waymo_evaluation(
        pred_dicts=pred_infos, top_k=args.top_k, eval_second=args.eval_second,
        num_modes_for_eval=args.num_modes_for_eval,
    )

    print(metric_results)
    metric_result_str = '\n'
    for key in metric_results:
        metric_results[key] = metric_results[key]
        metric_result_str += '%s: %.4f \n' % (key, metric_results[key])
    print(metric_result_str)
    print(result_format_str)






if __name__ == '__main__':
    main()