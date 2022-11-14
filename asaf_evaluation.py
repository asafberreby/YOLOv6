from pathlib import Path
from asaf_yolov6_detector_class import Yolov6Detector
from tqdm import tqdm
import cv2
import numpy as np
import boto3
from pygit2 import Repository, Commit
from datetime import datetime
import mlflow
import mlflow.sklearn
import json
import time
from statistics import mean
import configparser
config = configparser.ConfigParser()
config.read((Path(__file__).parent / '.git' / 'config').as_posix())

session = boto3.Session(profile_name='lsports-dev')
credentials = session.get_credentials()
s3 = boto3.resource(
	service_name='s3',
	region_name='eu-west-1',
	aws_access_key_id=credentials.access_key,
	aws_secret_access_key=credentials.secret_key,
	aws_session_token=credentials.token
)

def draw_bbox_on_image(img, box, color=(0,0,255)):
	if box is None:
		return img
	img = cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]), color, 1)
	return img



def get_iou(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou





def evaluate(detector:Yolov6Detector, s3_bucket, model:str, date:str, sport:str, iou_threshold:float):
	times = []

	sorted_images = sorted([it for it in list(s3_bucket.objects.all()) if 'images' in it.key and date in it.key and model in it.key and 'val' in it.key and 'images' != Path(it.key).name], key=lambda x: Path(x.key).name.split('.')[0])
	sorted_txts = sorted([it for it in list(s3_bucket.objects.all()) if 'lables' in it.key and date in it.key and model in it.key and 'val' in it.key and 'lables' != Path(it.key).name ], key=lambda x: Path(x.key).name.split('.')[0])
	
	
	couples = list(zip(sorted_images,sorted_txts))
	TP, TN, FP, FN = 0,0,0,0
	for couple in tqdm(couples, total= len(couples)):
		img_in_bytes = s3_bucket.Object(couple[0].key).get().get('Body').read()
		decoded_image_data = bytearray(img_in_bytes)
		imgFromByteArr = np.frombuffer(decoded_image_data,np.uint8)
		imgDecoded = cv2.imdecode(imgFromByteArr,cv2.IMREAD_COLOR)
		img = imgDecoded
		if img.shape != (416,416,3):
			img = cv2.resize(img, (416,416))

		json_file_in_bytes = s3_bucket.Object(couple[1].key).get().get('Body').read()
		value = eval('['+str(json_file_in_bytes).replace('b', '').replace("'", "").replace(' ', ',')+']')
		annotations = np.array([value[1], value[2], value[3], value[4]]) if len(value)>3 else None
		tl = (annotations[0] - 0.5*annotations[2], annotations[1] - 0.5*annotations[3]) if len(value)>3 else None
		br = (annotations[0] + 0.5*annotations[2], annotations[1] + 0.5*annotations[3]) if len(value)>3 else None
		annotated_bbox = np.array([tl[0], tl[1], br[0], br[1]]) if len(value)>3 else None

		t1 = time.time()
		detections, relative_detections = detector.detect(img) # (x1, y1, x2, y2)
		times.append(time.time() - t1)

		detected_bbox= relative_detections.cpu().numpy() if relative_detections is not None else None

		if annotated_bbox is None and detected_bbox is None: #means both bboxes are none
			TN+=1
		elif annotated_bbox is not None and detected_bbox is None:
			FN+=1
		elif annotated_bbox is None and detected_bbox is not None:
			FP+=1
		else:
			iou = get_iou(detected_bbox,annotated_bbox)
			if iou > iou_threshold:
				TP+=1
			if iou < iou_threshold:
				FP+=1

	
		detected_bbox_for_drawing = [round(it) for it in detections.cpu().numpy()] if detections is not None else None
		annotated_bbox_for_drawing = [round(it*416) for it in annotated_bbox] if detections is not None else None
		img = draw_bbox_on_image(img, detected_bbox_for_drawing,(0,0,255)) # detected in red
		img = draw_bbox_on_image(img, annotated_bbox_for_drawing,(255,0,0)) # annotated in blue
		img_to_show = cv2.resize(img, (1280, 720), cv2.INTER_LINEAR)
		cv2.imshow('test', img_to_show)
		cv2.waitKey(1)




	accuracy = (TP + TN)/(TP + TN + FP + FN)
	precision = TP/(TP + FP)
	recall = TP/(TP + FN)
	f1_score = (2 * precision * recall)/(precision + recall)
	
	mlflow.set_experiment(experiment_name='scoreboard_detector', experiment_id= '1')
	mlflow.log_param("Repository", config['remote "origin"'].get('url'))
	mlflow.log_param("weights_location", detector.weights_location)
	mlflow.log_param("Created", str(datetime.now()))
	mlflow.log_param("Models", type(detector))
	mlflow.log_param("git branch", Repository('.').head.shorthand)
	mlflow.log_param("commit id", str(Repository('.').head.target))
	mlflow.log_param("test_data", "s3/"+s3_bucket.name+'/'+model+'/'+date+'/'+'val')
	mlflow.log_param("train_data", "should be uploaded")
	mlflow.log_param("Sport", sport)
	mlflow.log_param("RT per inrference", mean(times[1:]))
	mlflow.log_metric("accuracy", accuracy)
	mlflow.log_metric("precision", precision)
	mlflow.log_metric("recall", recall)
	mlflow.log_metric("f1_score", f1_score)
	mlflow.log_metric("iou_threshold", iou_threshold)





if __name__ == "__main__":
	weights=Path(r"R:\Asaf\best_models\scoreboard_detection\Yolov6_SMALL_with_heavy_augmentations_14_11_2022\exp2 (with heavy augmentation)\weights\best_stop_aug_ckpt.pt").as_posix()
	yaml = Path(r"A:\YOLOv6\data\dataset.yaml").as_posix()
	detector = Yolov6Detector(weights,yaml)

	bucket = s3.Bucket('ocr-train-test-data')
	model = 'scoreboard_detector'
	date = '14_11_2022'
	sport = 'tennis+tabletennis'
	iou_threshold = 0.97

	
	evaluate(detector,bucket,model,date,sport, iou_threshold)