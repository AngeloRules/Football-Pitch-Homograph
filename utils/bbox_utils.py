import math
def get_center_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2] - bbox[0]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2), int(y2)

def get_closest_keypoint_index(points, keypoints, keypoint_indicies,frame_num):
    #point = get_center_bbox(points['bbox'])
    original_key_points = convert_keypoints_list(keypoints,frame_num)
    closest_distance = float('inf')
    key_point_ind = keypoint_indicies[0]
    for keypoint_index in keypoint_indicies:
        #print(keypoint_index)
        keypoint = (original_key_points[((keypoint_index*2) - 2)], original_key_points[((keypoint_index*2)-1)])
        distance = math.sqrt(((points[1]-keypoint[1])**2) + ((points[0]-keypoint[0])**2))
        #distance = abs(points[1]-keypoint[1])
        if distance<closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_index
    return key_point_ind

def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]

def measure_distance(p1,p2):
    return(math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

def convert_keypoints_list(keypoints,frame_num):
    original_key_points = [0]*50

    one_point =   keypoints['1'][frame_num]
    two_point =   keypoints['2'][frame_num]
    three_point =   keypoints['3'][frame_num]
    four_point =   keypoints['4'][frame_num]
    five_point =   keypoints['5'][frame_num]
    six_point =   keypoints['6'][frame_num]
    seven_point =   keypoints['7'][frame_num]
    eight_point =   keypoints['8'][frame_num]
    nine_point =   keypoints['9'][frame_num]
    ten_point =   keypoints['10'][frame_num]
    eleven_point =   keypoints['11'][frame_num]
    twelve_point =   keypoints['12'][frame_num]
    thirt_point =   keypoints['13'][frame_num]
    fourt_point =   keypoints['14'][frame_num]
    fiftht_point =   keypoints['15'][frame_num]
    sixtht_point =   keypoints['16'][frame_num]
    seventht_point =   keypoints['17'][frame_num]
    eightt_point =   keypoints['18'][frame_num]
    ninet_point =   keypoints['19'][frame_num]
    twenty_point =   keypoints['20'][frame_num]
    A_point =   keypoints['A'][frame_num]
    B_point =   keypoints['B'][frame_num]
    C_point =   keypoints['C'][frame_num]
    D_point =   keypoints['D'][frame_num]
    E_point =   keypoints['E'][frame_num]

    for track_id, point in one_point.items():
        original_key_points[0], original_key_points[1] = get_center_bbox(point['bbox'])
    for track_id, point in two_point.items():
        original_key_points[2], original_key_points[3] = get_center_bbox(point['bbox'])
    for track_id, point in three_point.items():
        original_key_points[4], original_key_points[5] = get_center_bbox(point['bbox'])
    for track_id, point in four_point.items():
        original_key_points[6], original_key_points[7] = get_center_bbox(point['bbox'])
    for track_id, point in five_point.items():
        original_key_points[8], original_key_points[9] = get_center_bbox(point['bbox'])
    for track_id, point in six_point.items():
        original_key_points[10], original_key_points[11] = get_center_bbox(point['bbox'])
    for track_id, point in seven_point.items():
        original_key_points[12], original_key_points[13] = get_center_bbox(point['bbox'])
    for track_id, point in eight_point.items():
        original_key_points[14], original_key_points[15] = get_center_bbox(point['bbox'])
    for track_id, point in nine_point.items():
        original_key_points[16], original_key_points[17] = get_center_bbox(point['bbox'])
    for track_id, point in ten_point.items():
        original_key_points[18], original_key_points[19] = get_center_bbox(point['bbox'])
    for track_id, point in eleven_point.items():
        original_key_points[20], original_key_points[21] = get_center_bbox(point['bbox'])
    for track_id, point in twelve_point.items():
        original_key_points[22], original_key_points[23] = get_center_bbox(point['bbox'])
    for track_id, point in thirt_point.items():
        original_key_points[24], original_key_points[25] = get_center_bbox(point['bbox'])
    for track_id, point in fourt_point.items():
        original_key_points[26], original_key_points[27] = get_center_bbox(point['bbox'])
    for track_id, point in fiftht_point.items():
        original_key_points[28], original_key_points[29] = get_center_bbox(point['bbox'])
    for track_id, point in sixtht_point.items():
        original_key_points[30], original_key_points[31] = get_center_bbox(point['bbox'])
    for track_id, point in seventht_point.items():
        original_key_points[32], original_key_points[33] = get_center_bbox(point['bbox'])
    for track_id, point in eightt_point.items():
        original_key_points[34], original_key_points[35] = get_center_bbox(point['bbox'])
    for track_id, point in ninet_point.items():
        original_key_points[36], original_key_points[37] = get_center_bbox(point['bbox'])
    for track_id, point in twenty_point.items():
        original_key_points[38], original_key_points[39] = get_center_bbox(point['bbox'])
    for track_id, point in A_point.items():
        original_key_points[40], original_key_points[41] = get_center_bbox(point['bbox'])
    for track_id, point in B_point.items():
        original_key_points[42], original_key_points[43] = get_center_bbox(point['bbox'])
    for track_id, point in C_point.items():
        original_key_points[44], original_key_points[45] = get_center_bbox(point['bbox'])
    for track_id, point in D_point.items():
        original_key_points[46], original_key_points[47] = get_center_bbox(point['bbox'])
    for track_id, point in E_point.items():
        original_key_points[48], original_key_points[49] = get_center_bbox(point['bbox']) 
    return original_key_points