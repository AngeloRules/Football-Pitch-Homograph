from ultralytics import YOLO
import pickle
import os
import sys
sys.path.append("../")
from utils import get_center_bbox, get_bbox_width
import cv2
import supervision as sv


class KeyPoints:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames,read_from_stubs=False,stub_path=None):
        if read_from_stubs and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                key_points = pickle.load(f)
            return key_points
        BATCH_SIZE = 1
        CONFIDENCE = 0.1

        detections = []

        key_points = {
            '1':[],
            '2':[],
            '3':[],
            '4':[],
            '5':[],
            '6':[],
            '7':[],
            '8':[],
            '9':[],
            '10':[],
            '11':[],
            '12':[],
            '13':[],
            '14':[],
            '15':[],
            '16':[],
            '17':[],
            '18':[],
            '19':[],
            '20':[],
            'A':[],
            'B':[],
            'C':[],
            'D':[],
            'E':[]
        }
        for i in range(0, len(frames),BATCH_SIZE):
            detections_batch = self.model.predict(frames[i:i+BATCH_SIZE],conf=CONFIDENCE)
            detections += detections_batch
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            detections_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision)
            key_points['1'].append({})
            key_points['2'].append({})
            key_points['3'].append({})
            key_points['4'].append({})
            key_points['5'].append({})
            key_points['6'].append({})
            key_points['7'].append({})
            key_points['8'].append({})
            key_points['9'].append({})
            key_points['10'].append({})
            key_points['11'].append({})
            key_points['12'].append({})
            key_points['13'].append({})
            key_points['14'].append({})
            key_points['15'].append({})
            key_points['16'].append({})
            key_points['17'].append({})
            key_points['18'].append({})
            key_points['19'].append({})
            key_points['20'].append({})
            key_points['A'].append({})
            key_points['B'].append({})
            key_points['C'].append({})
            key_points['D'].append({})
            key_points['E'].append({})
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if '1' in cls_names_inv.keys() and cls_id == cls_names_inv['1']:
                    key_points['1'][frame_num][track_id] = {'bbox':bbox}
                if '2' in cls_names_inv.keys() and cls_id == cls_names_inv['2']:
                    key_points['2'][frame_num][track_id] = {'bbox':bbox}
                if '3' in cls_names_inv.keys() and cls_id == cls_names_inv['3']:
                    key_points['3'][frame_num][track_id] = {'bbox':bbox}
                if '4' in cls_names_inv.keys() and cls_id == cls_names_inv['4']:
                    key_points['4'][frame_num][track_id] = {'bbox':bbox}
                if '5' in cls_names_inv.keys() and cls_id == cls_names_inv['5']:
                    key_points['5'][frame_num][track_id] = {'bbox':bbox}
                if '6' in cls_names_inv.keys() and cls_id == cls_names_inv['6']:
                    key_points['6'][frame_num][track_id] = {'bbox':bbox}
                if '7' in cls_names_inv.keys() and cls_id == cls_names_inv['7']:
                    key_points['7'][frame_num][track_id] = {'bbox':bbox}
                if '8' in cls_names_inv.keys() and cls_id == cls_names_inv['8']:
                    key_points['8'][frame_num][track_id] = {'bbox':bbox}
                if '9' in cls_names_inv.keys() and cls_id == cls_names_inv['9']:
                    key_points['9'][frame_num][track_id] = {'bbox':bbox}
                if '10' in cls_names_inv.keys() and cls_id == cls_names_inv['10']:
                    key_points['10'][frame_num][track_id] = {'bbox':bbox}
                if '11' in cls_names_inv.keys() and cls_id == cls_names_inv['11']:
                    key_points['11'][frame_num][track_id] = {'bbox':bbox}
                if '12' in cls_names_inv.keys() and cls_id == cls_names_inv['12']:
                    key_points['12'][frame_num][track_id] = {'bbox':bbox}
                if '13' in cls_names_inv.keys() and cls_id == cls_names_inv['13']:
                    key_points['13'][frame_num][track_id] = {'bbox':bbox}
                if '14' in cls_names_inv.keys() and cls_id == cls_names_inv['14']:
                    key_points['14'][frame_num][track_id] = {'bbox':bbox}
                if '15' in cls_names_inv.keys() and cls_id == cls_names_inv['15']:
                    key_points['15'][frame_num][track_id] = {'bbox':bbox}
                if '16' in cls_names_inv.keys() and cls_id == cls_names_inv['16']:
                    key_points['16'][frame_num][track_id] = {'bbox':bbox}
                if '17' in cls_names_inv.keys() and cls_id == cls_names_inv['17']:
                    key_points['17'][frame_num][track_id] = {'bbox':bbox}
                if '18' in cls_names_inv.keys() and cls_id == cls_names_inv['18']:
                    key_points['18'][frame_num][track_id] = {'bbox':bbox}
                if '19' in cls_names_inv.keys() and cls_id == cls_names_inv['19']:
                    key_points['19'][frame_num][track_id] = {'bbox':bbox}
                if '20' in cls_names_inv.keys() and cls_id == cls_names_inv['20']:
                    key_points['20'][frame_num][track_id] = {'bbox':bbox}
                if 'A' in cls_names_inv.keys() and cls_id == cls_names_inv['A']:
                    key_points['A'][frame_num][track_id] = {'bbox':bbox}
                if 'B' in cls_names_inv.keys() and cls_id == cls_names_inv['B']:
                    key_points['B'][frame_num][track_id] = {'bbox':bbox}
                if 'C' in cls_names_inv.keys() and cls_id == cls_names_inv['C']:
                    key_points['C'][frame_num][track_id] = {'bbox':bbox}
                if 'D' in cls_names_inv.keys() and cls_id == cls_names_inv['D']:
                    key_points['D'][frame_num][track_id] = {'bbox':bbox}
                if 'E' in cls_names_inv.keys() and cls_id == cls_names_inv['E']:
                    key_points['E'][frame_num][track_id] = {'bbox':bbox}
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(key_points, f)
        
        return key_points
    
    def _draw_point(self,frame,bbox,color,track_id,point):
        x_center, y_center = get_center_bbox(bbox)
        width = get_bbox_width(bbox)
         
        cv2.putText(frame,point,(x_center,y_center+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        cv2.circle(
            frame,
            center=(x_center,y_center),
            radius=int(width*0.1),
            color=color,
            thickness=-1,
            lineType=cv2.LINE_4)
        return frame
       
    
    def draw_annotations(self, video_frames,tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            one_point =   tracks['1'][frame_num]
            two_point =   tracks['2'][frame_num]
            three_point =   tracks['3'][frame_num]
            four_point =   tracks['4'][frame_num]
            five_point =   tracks['5'][frame_num]
            six_point =   tracks['6'][frame_num]
            seven_point =   tracks['7'][frame_num]
            eight_point =   tracks['8'][frame_num]
            nine_point =   tracks['9'][frame_num]
            ten_point =   tracks['10'][frame_num]
            eleven_point =   tracks['11'][frame_num]
            twelve_point =   tracks['12'][frame_num]
            thirt_point =   tracks['13'][frame_num]
            fourt_point =   tracks['14'][frame_num]
            fiftht_point =   tracks['15'][frame_num]
            sixtht_point =   tracks['16'][frame_num]
            seventht_point =   tracks['17'][frame_num]
            eightt_point =   tracks['18'][frame_num]
            ninet_point =   tracks['19'][frame_num]
            twenty_point =   tracks['20'][frame_num]
            A_point =   tracks['A'][frame_num]
            B_point =   tracks['B'][frame_num]
            C_point =   tracks['C'][frame_num]
            D_point =   tracks['D'][frame_num]
            E_point =   tracks['E'][frame_num]


            for track_id, point in one_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"1")
            for track_id, point in two_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"2")
            for track_id, point in three_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"3")
            for track_id, point in four_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"4")
            for track_id, point in five_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"5")
            for track_id, point in six_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"6")
            for track_id, point in seven_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"7")
            for track_id, point in eight_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"8")
            for track_id, point in nine_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"9")
            for track_id, point in ten_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"10")
            for track_id, point in eleven_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"11")
            for track_id, point in twelve_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"12")
            for track_id, point in thirt_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"13")
            for track_id, point in fourt_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"14")
            for track_id, point in fiftht_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"15")
            for track_id, point in sixtht_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"16")
            for track_id, point in seventht_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"17")
            for track_id, point in eightt_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"18")
            for track_id, point in ninet_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"19")
            for track_id, point in twenty_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"20")
            for track_id, point in A_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"A")
            for track_id, point in B_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"B")
            for track_id, point in C_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"C")
            for track_id, point in D_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"D")
            for track_id, point in E_point.items():
                frame = self._draw_point(frame, point['bbox'],(0,0,255),track_id,"E")        
                            
            output_video_frames.append(frame)
        return output_video_frames