import cv2
import sys
import numpy as np

sys.path.append("../")
import constants
from utils import (convert_pixels_to_meters, 
                   convert_meters_to_pixels, 
                   get_foot_position, 
                   get_closest_keypoint_index, 
                   get_height_of_bbox, measure_xy_distance,convert_keypoints_list,get_center_bbox,measure_distance)

class MiniPitch():
    def __init__(self, frame):
        self.mini_pitch_width = 290
        self.mini_pitch_height = 485
        self.buffer = 50
        self.top_buffer = frame[0].shape[0] - (self.mini_pitch_width + 50)
        self.side_buffer =  int(frame[0].shape[1]/2) - int(self.mini_pitch_height/2) 
        self.padding = 10
        self._set_pitch_background(frame)
        self._set_mini_pitch()
        self._pitch_drawing_key_points()
        self._set_connecting_line()
    
    def _pitch_drawing_key_points(self):
        self.drawing_keypoints = [0]*50

        self.drawing_keypoints[0],self.drawing_keypoints[1] = int(self.mini_pitch_start_x), int(self.mini_pitch_end_y)
        self.drawing_keypoints[12], self.drawing_keypoints[13] = int(self.mini_pitch_start_x), int(self.mini_pitch_start_y)
        self.drawing_keypoints[14], self.drawing_keypoints[15] = int(self.mini_pitch_end_x), int(self.mini_pitch_end_y) 

        self.drawing_keypoints[2], self.drawing_keypoints[3] = int(self.mini_pitch_end_x), int(self.mini_pitch_start_y) 

        self.drawing_keypoints[4], self.drawing_keypoints[5] = int(self.mini_pitch_start_x) ,int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG_2,
                                                                                                 constants.TOTAL_FIELD_WIDTH,
                                                                                                 self.drawing_mini_pitch_width))
        
        self.drawing_keypoints[8], self.drawing_keypoints[9] =  int(self.mini_pitch_start_x), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG,
                                                                                                            constants.TOTAL_FIELD_WIDTH,
                                                                                                            self.drawing_mini_pitch_width))
        
        self.drawing_keypoints[16], self.drawing_keypoints[17] = int(self.mini_pitch_start_x + convert_meters_to_pixels(
            constants.PENALTY_BOX_LENGTH, constants.TOTAL_FIELD_WIDTH, self.drawing_mini_pitch_width
        )), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))

        self.drawing_keypoints[20], self.drawing_keypoints[21] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
       
        self.drawing_keypoints[10], self.drawing_keypoints[11] = int(self.mini_pitch_end_x), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        

        self.drawing_keypoints[6], self.drawing_keypoints[7] = int(self.mini_pitch_end_x), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_TO_FLAG,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))

        self.drawing_keypoints[22], self.drawing_keypoints[23] = int(self.mini_pitch_end_x - convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,
                                                                                                              constants.TOTAL_FIELD_WIDTH,
                                                                                                              self.drawing_mini_pitch_width)), int(self.mini_pitch_start_y + (convert_meters_to_pixels(constants.BOX_TO_FLAG_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)))

        self.drawing_keypoints[18], self.drawing_keypoints[19] =  int(self.mini_pitch_end_x - convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,
                                                                                                              constants.TOTAL_FIELD_WIDTH,
                                                                                                              self.drawing_mini_pitch_width)), int(self.mini_pitch_start_y + (convert_meters_to_pixels(constants.BOX_TO_FLAG,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)))

        self.drawing_keypoints[48], self.drawing_keypoints[49] =  int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.HALF_LENGTH,
                                                                                                                                                      constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)), int(self.mini_pitch_end_y)

        self.drawing_keypoints[40], self.drawing_keypoints[41] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.HALF_LENGTH,
                                                                                                                        constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)), int(self.mini_pitch_start_y)
        
        self.drawing_keypoints[44], self.drawing_keypoints[45] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.HALF_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.HALF_WIDTH,
                                                                                                                                                      constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[24], self.drawing_keypoints[25] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_EDGE_TO_POINT_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[26], self.drawing_keypoints[27] = int(self.mini_pitch_end_x - convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_EDGE_TO_POINT,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[28], self.drawing_keypoints[29] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_EDGE_TO_POINT,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[30], self.drawing_keypoints[31] = int(self.mini_pitch_end_x - convert_meters_to_pixels(constants.PENALTY_BOX_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.BOX_EDGE_TO_POINT_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[32], self.drawing_keypoints[33] = int(self.mini_pitch_start_x), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.POST_EDGE_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[34], self.drawing_keypoints[35] = int(self.mini_pitch_end_x),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.POST_EDGE,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[36], self.drawing_keypoints[37] = int(self.mini_pitch_start_x), int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.POST_EDGE,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[38], self.drawing_keypoints[39] = int(self.mini_pitch_end_x),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.POST_EDGE_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[42], self.drawing_keypoints[43] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.HALF_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.CENTER_CIRCLE_POINT,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))
        self.drawing_keypoints[46], self.drawing_keypoints[47] = int(self.mini_pitch_start_x + convert_meters_to_pixels(constants.HALF_LENGTH,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),int(self.mini_pitch_start_y + convert_meters_to_pixels(constants.CENTER_CIRCLE_POINT_2,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width))

    def _set_connecting_line(self):
        self.lines = [
                (1,7),
                (1,8),
                (8,2),
                (2,7),
                (3,9),
                (5,11),
                (9,11),
                (6,12),
                (4,10),
                (12,10), 
                (21,25)
            ]

        

    # def _set_mini_pitch(self):
    #     self.mini_pitch_start_x = self.start_x + self.padding
    #     self.mini_pitch_start_y = self.start_y + self.padding
    #     self.mini_pitch_end_x = self.end_x + self.padding
    #     self.mini_pitch_end_y = self.end_y + self.padding
    #     self.drawing_mini_pitch_width = self.mini_pitch_end_x - self.mini_pitch_start_x 
    
    def _set_mini_pitch(self):
        self.mini_pitch_start_x = self.start_x 
        self.mini_pitch_start_y = self.start_y 
        self.mini_pitch_end_x = self.end_x 
        self.mini_pitch_end_y = self.end_y 
        self.drawing_mini_pitch_width = self.mini_pitch_end_y - self.mini_pitch_start_y

    # def _set_pitch_background(self,frame):
    #     frame = frame.copy()
    #     self.end_x = frame[0].shape[1] - self.buffer
    #     self.end_y = self.buffer + self.mini_pitch_height
    #     self.start_x = self.end_x - self.drawing_mini_pitch_width
    #     self.start_y = self.end_y - self.mini_pitch_height
    
    def _set_pitch_background(self,frame):

        frame = frame.copy()
        self.end_x = frame[0].shape[1] - self.side_buffer
        self.end_y = self.top_buffer + self.mini_pitch_width
        self.start_x = self.end_x - self.mini_pitch_height
        self.start_y = self.end_y - self.mini_pitch_width


    def _draw_background_rectangle(self, frame):
        #shapes = np.zeros_like(frame, np.uint8)
        out_put = frame.copy()
        cv2.rectangle(out_put, (self.start_x,self.start_y), (self.end_x, self.end_y),(0,255,0), -1)
        #out_put = frame.copy()
        #ALPHA = 0.5
        #mask = shapes.astype(bool)
        #out_put[mask] = cv2.addWeighted(frame, ALPHA, shapes, 1-ALPHA, 0)[mask]
        return out_put
    
    def _draw_pitch(self, frame):
        origin_x = int(self.drawing_keypoints[44])
        origin_y = int(self.drawing_keypoints[45])
        cv2.circle(frame, (origin_x,origin_y),int(convert_meters_to_pixels(constants.CENTER_CIRCLE_RADIUS,
                                                                           constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)),(255,255,255),2)
       

        for i in range(0, len(self.drawing_keypoints), 2):
            x = int(self.drawing_keypoints[i])
            y = int(self.drawing_keypoints[i+1])
            cv2.circle(frame, (x,y), 5, (255,0,0),-1)

        for line in self.lines:
            if line[0]  == 1:
                start = (int(self.drawing_keypoints[line[0]-1]), int(self.drawing_keypoints[line[0]]))
            else:
                start =  (int(self.drawing_keypoints[(line[0]*2) - 2]), int(self.drawing_keypoints[(line[0]*2) - 1]))
            stop = (int(self.drawing_keypoints[(line[1]*2) - 2]), int(self.drawing_keypoints[(line[1]*2) - 1]))
            cv2.line(frame, start, stop, (255,255,255),2)
        return frame

    
    def draw_mini_pitch(self, frames):
        output_frames = []
        for frame in frames:
            frame =  self._draw_background_rectangle(frame)
            frame = self._draw_pitch(frame)
            output_frames.append(frame)

        return  output_frames
    
    def get_mini_pitch_start_point(self):
        return (self.mini_pitch_start_x, self.mini_pitch_start_y)
    
    def get_drawing_keypoints(self):
        return self.drawing_keypoints
    
    def get_mini_pitch_width(self):
        return self.drawing_mini_pitch_width
    
    def get_mini_court_coordinates(self, 
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index,
                                   player_height_in_pixels,
                                   player_height_in_meters):
        x_pixel_distance_from_keypoint, y_pixel_distance_from_keypoint = measure_xy_distance(object_position, closest_key_point)

        x_meter_distance_from_keypoint = convert_pixels_to_meters(x_pixel_distance_from_keypoint,
                                                                  player_height_in_meters,
                                                                  player_height_in_pixels)
        
        y_meter_distance_from_keypoint = convert_pixels_to_meters(y_pixel_distance_from_keypoint,
                                                                  player_height_in_meters,
                                                                  player_height_in_pixels)
        
        mini_court_x_pixel_distance = convert_meters_to_pixels(x_meter_distance_from_keypoint,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)
        mini_court_y_pixel_distance = convert_meters_to_pixels(y_meter_distance_from_keypoint,constants.TOTAL_FIELD_WIDTH,self.drawing_mini_pitch_width)
        closest_mini_court_keypoint = (self.drawing_keypoints[(closest_key_point_index*2) -2],
                                       self.drawing_keypoints[(closest_key_point_index*2)- 1])
        mini_court_player_positions = (
            closest_mini_court_keypoint[0]+mini_court_x_pixel_distance,
            closest_mini_court_keypoint[1]+mini_court_y_pixel_distance
        )
        
        return mini_court_player_positions

    def return_zero_keypoint(self):
        condition = lambda x: x==0
        indexes = [index for index, value in enumerate(self.drawing_keypoints) if condition(value)]
        return indexes


    def project_court_to_mini_court(self,player_bboxs,original_keypoints,video_frames):
    
        output_player_boxes = []
        output_ball_boxes = []
        for frame_num, _ in enumerate(video_frames):
            player_tracks = player_bboxs['players'][frame_num]
            ball_tracks = player_bboxs['ball'][frame_num][1]['bbox']
            ball_position = get_center_bbox(ball_tracks)
            closest_player_id_to_ball = min(player_tracks.keys(), default=0,key=lambda x: measure_distance(ball_position,get_foot_position(player_tracks[x]['bbox'])))

            output_player_bboxs_dict = {}

            for player_id, bbox in player_tracks.items():
                color = bbox.get('team_color',(0,0,255))
                foot_position = get_foot_position(bbox["bbox"])
                originalKeypoints = convert_keypoints_list(original_keypoints,frame_num)
                # condition = lambda x: x != 0
                # indexes = list(filter(lambda x: condition(originalKeypoints[x]), range(len(originalKeypoints))))
                # indexes_converted = list(set([int((x+2)/2) if x%2==0 else int((x+1)/2) for x in indexes]))
                closest_keypoint_index = get_closest_keypoint_index(foot_position, original_keypoints, [i for i in range(1,26)],frame_num)
                
                closest_key_point = (originalKeypoints[(closest_keypoint_index*2) -2],
                                     originalKeypoints[(closest_keypoint_index*2) -1])
                
    
                bbox_height_in_pixels = [get_height_of_bbox(bbox['bbox'])]
                

                mini_court_player_position = self.get_mini_court_coordinates(
                    foot_position,
                    closest_key_point,
                    closest_keypoint_index,
                    bbox_height_in_pixels[0],
                    constants.AVERAGE_PLAYER_HEIGHT
                )
                output_player_bboxs_dict[player_id] = [mini_court_player_position,color]

                if closest_player_id_to_ball == player_id:
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, original_keypoints, [i for i in range(1,26)],frame_num)
                
                    closest_key_point = (originalKeypoints[(closest_keypoint_index*2) -2],
                                     originalKeypoints[(closest_keypoint_index*2) -1])
                    
                    mini_court_ball_position = self.get_mini_court_coordinates(
                    ball_position,
                    closest_key_point,
                    closest_keypoint_index,
                    bbox_height_in_pixels[0],
                    constants.AVERAGE_PLAYER_HEIGHT
                )
                    output_ball_boxes.append({1:[mini_court_ball_position,(0,0,255)]})

            output_player_boxes.append(output_player_bboxs_dict)
        return output_player_boxes, output_ball_boxes

  
    def draw_points_on_minipitch(self, video_frames,positions):
        for frame_num, frame in enumerate(video_frames):
            for _, position in positions[frame_num].items():
                x,y = position[0]
                color = position[1]
                x = int(x)
                y = int(y)
                cv2.circle(frame,(x,y),5,color,-1)
        return video_frames
                





