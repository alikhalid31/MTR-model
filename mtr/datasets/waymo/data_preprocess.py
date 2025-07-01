# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import sys, os
import numpy as np
import pickle
import tensorflow as tf
import multiprocessing
import glob
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type

# import pandas as pd
# from shapely.geometry import LineString
# from itertools import combinations

# function to count total intersections between all the segments of polylines
# def count_polyline_intersections(polylines):
#     """
#     Given a list of polylines (each is a list of (x, y) tuples),
#     count how many times the line segments intersect each other.

#     Args:
#         polylines: List of polylines, where each polyline is a list of (x, y) tuples.

#     Returns:
#         Total number of pairwise segment intersections.
#     """
#     segments = []

#     # Break each polyline into line segments
#     for polyline in polylines:
#         for i in range(len(polyline) - 1):
#             segment = LineString([polyline[i], polyline[i + 1]])
#             segments.append(segment)

#     # Count all unique intersections
#     intersection_count = 0
#     for seg1, seg2 in combinations(segments, 2):
#         if seg1.crosses(seg2) or seg1.intersects(seg2):
#             if seg1.intersection(seg2).geom_type in ['Point', 'MultiPoint', 'LineString']:
#                 intersection_count += 1

#     return intersection_count

# funtion to count intersections between polylines, but only once per pair of polylines
def count_polyline_intersections_unique(polylines):
    """
    Given a list of polylines (each is a list of (x, y) tuples),
    count how many times polylines intersect each other — only once per polyline pair.

    Args:
        polylines: List of polylines, where each polyline is a list of (x, y) tuples.

    Returns:
        Total number of unique polyline pair intersections.
    """
    # Convert each polyline into list of segments
    all_segments = []
    for polyline in polylines:
        segments = []
        for i in range(len(polyline) - 1):
            segment = LineString([polyline[i], polyline[i + 1]])
            segments.append(segment)
        all_segments.append(segments)

    intersected_pairs = set()

    # Check all unique pairs of polylines
    for i, j in combinations(range(len(polylines)), 2):
        segments_i = all_segments[i]
        segments_j = all_segments[j]

        # Check if any segment from polyline i intersects any segment from polyline j
        for seg1 in segments_i:
            for seg2 in segments_j:
                if seg1.crosses(seg2) or seg1.intersects(seg2):
                    if seg1.intersection(seg2).geom_type in ['Point', 'MultiPoint', 'LineString']:
                        intersected_pairs.add((i, j))
                        break  # Stop checking this pair once intersection is found
            else:
                continue
            break

    return len(intersected_pairs)

def populate_dataframe(info, track_infos, interactions_count):
    rows=[]
    scenario_id = info['scenario_id']
    current_time_index = info['current_time_index']

    for idx, track in enumerate(track_infos['trajs']):
        object_id = track_infos['object_id'][idx]
        object_type = track_infos['object_type'][idx]
        # Split into past and future trajectories
        past_traj = track[:current_time_index]
        future_traj = track[current_time_index:]
        curr_traj = track[current_time_index ]

        # The last column (index 9) is the `valid` flag
        valid_past = int(np.sum(past_traj[:, 9]))     # counts how many are valid in past
        valid_future = int(np.sum(future_traj[:, 9])) # counts how many are valid in future
        valid_current = int(np.sum(curr_traj[9])) # valid at current time index
        
        to_predict = False
        if idx in info['tracks_to_predict']['track_index']:
            to_predict = True

        rows.append([
            scenario_id, object_id, object_type, 
            valid_past, valid_future, valid_current, to_predict, interactions_count
        ])
    return rows


def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'trajs': []
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)

    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': []
    }
    polylines = []

    point_cnt = 0
    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = lane_type[cur_data.lane.type]  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': x.boundary_type  # roadline type
                } for x in cur_data.lane.left_boundaries
            ]
            cur_info['right_boundary'] = [{
                    'start_index': x.lane_start_index, 'end_index': x.lane_end_index,
                    'feature_id': x.boundary_feature_id,
                    'boundary_type': road_line_type[x.boundary_type]  # roadline type
                } for x in cur_data.lane.right_boundaries
            ]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = road_line_type[cur_data.road_line.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = road_edge_type[cur_data.road_edge.type]

            global_type = polyline_type[cur_info['type']]
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            print(cur_data)
            raise ValueError

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except:
        polylines = np.zeros((0, 7), dtype=np.float32)
        print('Empty polylines: ')
    map_infos['all_polylines'] = polylines
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos


def process_waymo_data_with_scenario_proto(data_file, output_path=None):
    dataset = tf.data.TFRecordDataset(data_file, compression_type='')
    ret_infos = []
    # df_rows=[]
    for cnt, data in enumerate(dataset):
        info = {}
        scenario = scenario_pb2.Scenario()
        # scenario.ParseFromString(bytearray(data.numpy()))
        scenario.ParseFromString(bytes(data.numpy()))

        # if condition to filter out the scenarios with no tracks to predict
        if len(scenario.tracks_to_predict) != 1:
            continue

        info['scenario_id'] = scenario.scenario_id
        info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
        info['current_time_index'] = scenario.current_time_index  # int, 10
        info['sdc_track_index'] = scenario.sdc_track_index  # int
        info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list

        track_infos = decode_tracks_from_proto(scenario.tracks)

        # filtering tracks to predict based on speed
        # track_index= [cur_pred.track_index for cur_pred in scenario.tracks_to_predict]

        # track_index_filter =[]
        # for index in track_index:
        #     single_track = track_infos['trajs'][index] 
        #     # 10 = timestamp, 7 = velocity_x, 8 = velocity_y
        #     speed = np.sqrt(single_track[10][7]**2 + single_track[10][8]**2)
        #     if speed >= 21 and speed <= 1000:
        #         track_index_filter.append(index)
        #         # print(speed)



        # modified this code to filter out the tracks wrt to speed
        # info['tracks_to_predict'] = {
        #     'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict if cur_pred.track_index in track_index_filter],
        #     'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict if cur_pred.track_index in track_index_filter]
        # }

        info['tracks_to_predict'] = {
            'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
        }  # for training: suggestion of objects to train on, for val/test: need to be predicted

        # if conditoin to filter scenerio based on number of agents (tracks) in the scene 
        #if not (len(scenario.tracks) >=100 and len(scenario.tracks) < 1000):
            # print(len(scenario.tracks))
            #continue    
        
        info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']]
        
        # decode map related data
        map_infos = decode_map_features_from_proto(scenario.map_features)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)


        # code to create a dataframe to find data stats
        # ego_polylines = [track_infos['trajs'][i][:,:2] for i in info['tracks_to_predict']['track_index']]
        # interactions_count = count_polyline_intersections_unique(ego_polylines)
        # df_rows += populate_dataframe(info, track_infos,interactions_count)

        save_infos = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        save_infos.update(info)

        output_file = os.path.join(output_path, f'sample_{scenario.scenario_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(save_infos, f)


        ret_infos.append(info)
    # return df_rows, ret_infos
    return  ret_infos


def get_infos_from_protos(data_path, output_path=None, num_workers=8):
    from functools import partial
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    func = partial(
        process_waymo_data_with_scenario_proto, output_path=output_path
    )

    src_files = glob.glob(os.path.join(data_path, '*.tfrecord*'))
    src_files.sort()


    # func(src_files[0])
    # all_df_rows= []
    # all_infos = []
    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap(func, src_files), total=len(src_files)))
        # for df_rows, data_infos in tqdm(p.imap(func, src_files), total=len(src_files)):
        #     all_df_rows.extend(df_rows)
        #     all_infos.extend(data_infos)

    all_infos = [item for infos in data_infos for item in infos]
    # return all_df_rows, all_infos

    return  all_infos


def create_infos_from_protos(raw_data_path, output_path, num_workers=1):
    # train_infos = get_infos_from_protos(
    #     data_path=os.path.join(raw_data_path, 'training'),
    #     output_path=os.path.join(output_path, 'processed_scenarios_training'),
    #     num_workers=num_workers
    # )
    # train_filename = os.path.join(output_path, 'processed_scenarios_training_infos.pkl')
    # with open(train_filename, 'wb') as f:
    #     pickle.dump(train_infos, f)
    # print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    # val_infos = get_infos_from_protos(
    #     data_path=os.path.join(raw_data_path, 'validation'),
    #     output_path=os.path.join(output_path, 'processed_scenarios_validation_speed_btw_21_inf'),
    #     num_workers=num_workers
    # )
    # val_filename = os.path.join(output_path, 'processed_scenarios_val_speed_btw_21_inf_infos.pkl')
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(val_infos, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)
    
    # df_rows, 
    debug_infos = get_infos_from_protos(
        data_path=os.path.join(raw_data_path, 'validation'),
        output_path=os.path.join(output_path, 'processed_scenarios_validation_ego_1'),
        num_workers=num_workers
    )
    debug_filename = os.path.join(output_path, 'processed_scenarios_val_ego_1_infos.pkl')
    # df = pd.DataFrame(df_rows, columns=[
    # 'scenario', 'agnet_id', 'object_type',
    # 'past_valid_stamps', 'future_valid_stamps',
    # 'current_valid_timestamp', 'to_predict', 'interactions_count'
    # ])
    # df.to_csv(os.path.join(output_path, 'processed_scenarios_validation.csv'), index=False) 
   
    with open(debug_filename, 'wb') as f:
        pickle.dump(debug_infos, f)
    print('----------------Waymo info debug file is saved to %s----------------' % debug_filename)


if __name__ == '__main__':
    create_infos_from_protos(
        raw_data_path=sys.argv[1],
        output_path=sys.argv[2]
    )
