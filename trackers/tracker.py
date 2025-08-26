from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_bbox, get_bbox_width
import cv2
import numpy as np
import pandas as pd

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def _detect_frames(self, frames):
        BATCH_SIZE = 1
        CONFIDENCE = 0.1

        detections = []
        for i in range(0, len(frames),BATCH_SIZE):
            detections_batch = self.model.predict(frames[i:i+BATCH_SIZE],conf=CONFIDENCE)
            detections += detections_batch
        return detections 
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:{'bbox':x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def get_object_tracks(self, frames,read_from_stubs=False, stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
            
        detections = self._detect_frames(frames)
        tracks = {
            'players':[],
            'referees':[],
            'ball':[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detections_supervision = sv.Detections.from_ultralytics(detection)
            for obj_index, class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detections_supervision.class_id[obj_index] = cls_names_inv['player']
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision)
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox':bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox':bbox}

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox':bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)            
        return tracks
    
    def _draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width),
                  ),angle=0.0, startAngle=-45,endAngle=225,
                  color=color,
                  thickness=2,
                  lineType=cv2.LINE_4)
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,(int(x1_rect),int(y1_rect)),(int(x2_rect),int(y2_rect)),color,-1)
            x1_text = x1_rect +12
            if track_id > 99:
                x1_rect -= 10
            cv2.putText(frame,f"{track_id}",(int(x1_text),int(y1_rect+15)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        return frame
    
    def _draw_triangle(self, frame, bbox, color):
        y_bottom = int(bbox[1])
        x,_ = get_center_bbox(bbox)

        trangle_points = np.array([
            [x,y_bottom],
            [x-10,y_bottom-20],
            [x+10,y_bottom-20]
        ])
        cv2.drawContours(frame, [trangle_points],0,color,-1)
        cv2.drawContours(frame, [trangle_points],0,(0,0,0),2)
        return frame
    
    def draw_annotations(self, video_frames,tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            player_tracks =   tracks['players'][frame_num]
            ball_tracks = tracks['ball'][frame_num]
            referee_tracks = tracks['referees'][frame_num]

            for track_id, player in player_tracks.items():
                color = player.get("team_color",(0,0,255))
                frame = self._draw_ellipse(frame, player['bbox'],color,track_id)

            for track_id, referee in referee_tracks.items():
                frame = self._draw_ellipse(frame, referee['bbox'],(0,255,255),track_id)

            for track_id, ball in ball_tracks.items():
                frame = self._draw_triangle(frame,ball['bbox'],(0,255,0))
            output_video_frames.append(frame)
        return output_video_frames
