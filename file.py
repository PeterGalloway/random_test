import glob
import random
import os
import shutil

files = glob.glob("PATH_TO_FILES/*/*.jpg")
if os.path.isdir('SAVE_DIR'):
	shutil.rmtree('SAVE_DIR') 

os.makedirs('SAVE_DIR')
random.seed(35)
random.shuffle(files)


from PIL import Image, ImageFile, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
from tqdm import tqdm
import time
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

def draw_image(img, file_name, bbox_face, bbox_person, label, idx, color):
	draw = ImageDraw.Draw(img)
	if bbox_face:
		x_min, y_min, x_max, y_max = bbox_face
		shapely_face = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
		draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)
		draw.text((x_min, y_min), f'Label: {label}')
	if bbox_person:
		x_min, y_min, x_max, y_max = bbox_person
		shapely_face = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
		draw.rectangle((x_min, y_min, x_max, y_max), outline=color, width=3)
		draw.text((x_min, y_min), f'Label: {label}')
	return img

def box_iou(box1, box2, over_second=False):
	def box_area(box):
		# box = 4xn
		return (box[2] - box[0]) * (box[3] - box[1])
	area1 = box_area(box1.T)
	area2 = box_area(box2.T)
	# inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
	inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
	iou = inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
	if over_second:
		return (inter / area2 + iou) / 2  # mean(inter / area2, iou)
	else:
		return iou

def assign_faces(persons_bboxes, faces_bboxes, iou_thresh=0.0001):
	assigned_faces = [None for _ in range(len(faces_bboxes))]
	unassigned_persons_inds = [p_ind for p_ind in range(len(persons_bboxes))]
	if len(persons_bboxes) == 0 or len(faces_bboxes) == 0:
		return assigned_faces, unassigned_persons_inds
	cost_matrix = box_iou(torch.stack(persons_bboxes), torch.stack(faces_bboxes), over_second=True).cpu().numpy()
	persons_indexes, face_indexes = [], []
	if len(cost_matrix) > 0:
		persons_indexes, face_indexes = linear_sum_assignment(cost_matrix, maximize=True)
	matched_persons = set()
	for person_idx, face_idx in zip(persons_indexes, face_indexes):
		ciou = cost_matrix[person_idx][face_idx]
		if ciou > iou_thresh:
			if person_idx in matched_persons:
				# Person can not be assigned twice, in reality this should not happen
				continue
			assigned_faces[face_idx] = person_idx
			matched_persons.add(person_idx)
	unassigned_persons_inds = [p_ind for p_ind in range(len(persons_bboxes)) if p_ind not in matched_persons]
	return assigned_faces, unassigned_persons_inds

model = YOLO('YOLO_PATH/yolov8x.pt') # device='gpu'
dino_processor = AutoProcessor.from_pretrained("DINO_PATH/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("DINO_PATH/grounding-dino-base").to("cuda")
score_threshold_face = 0.16  ## 0.16 ##vielleicht hier etwas höher auf 0.2?
score_threshold_person = 0.4  # 0.4
queries = f"person. face."
total_time = []
for file in tqdm(files[:100], total=100):
	start = time.time()
	image = Image.open(file).convert("RGB")
	width, height = image.size
	target_sizes=[(width, height)]
	inputs = dino_processor(text=queries, images=image, return_tensors="pt").to("cuda")
	with torch.no_grad():
		outputs = dino_model(**inputs)
	results = dino_processor.post_process_grounded_object_detection(
		outputs,
		inputs.input_ids,
		box_threshold=score_threshold_face,
		text_threshold=0.3,
		target_sizes=[image.size[::-1]]
	)
	boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
	result_labels = []
	person_bboxes, face_bboxes = [], []
	person_bboxes_inds, face_bboxes_inds = [], []
	idx = 0
	for _, (box, score, label) in enumerate(zip(boxes, scores, labels)):
		box = [int(i) for i in box.tolist()]
		if score < score_threshold_face:
			continue
		if label == "person":
			continue
		elif label == "face":
			result_labels.append((box, label))
			face_bboxes.append(box)
			face_bboxes_inds.append(idx)
			idx += 1
		else:
			continue
	results = model.predict(file)
	for _, r in enumerate(results[0].boxes):
		box = r.xyxy.squeeze().tolist()
		if r.cls == 0 and r.conf >= score_threshold_person:
			## Person
			result_labels.append((box, "person"))
			person_bboxes.append(box)
			person_bboxes_inds.append(idx)
			idx += 1
	face_to_person_map = {ind: None for ind in face_bboxes_inds}
	end = time.time()
	person_bboxes = [torch.tensor(x) for x in person_bboxes]
	face_bboxes = [torch.tensor(x) for x in face_bboxes]
	assigned_faces, unassigned_persons_inds = assign_faces(person_bboxes, face_bboxes)
	for face_ind, person_ind in enumerate(assigned_faces):
		face_ind = face_bboxes_inds[face_ind]
		person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None
		face_to_person_map[face_ind] = person_ind
	unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds]
	colors = ["darkred", "darkblue", "darkgreen", "gold", "aqua", "lime", "navy", "thistle", "white", "violet", "chocolate", "darkcyan", "darkorchid", "lightgrey", "mediumturquoise", "mediumslateblue", "mediumaquamarine", "maroon", "magenta", "limegreen", "powderblue", "peru", "salmon", "silver", "steelblue", "tan", "tomato", "yellowgreen", "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgreen", "lightgray", "lightgrey", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"]
	colors_unassigned_persons = "darksalmon"
	total_time.append(end-start)
	image_total = Image.open(file).convert("RGB")
	for idx, ((face_ind, person_ind), color) in enumerate(zip(face_to_person_map.items(), colors)):
		image_original = Image.open(file).convert("RGB")
		bbox_face = result_labels[face_ind][0]
		if person_ind:
			bbox_person = result_labels[person_ind][0]
			img = draw_image(image_original, os.path.basename(file), bbox_face, bbox_person, "label", idx, color)
			img_total = draw_image(image_total, os.path.basename(file), bbox_face, bbox_person, "label", idx, color)
		else:
			img = draw_image(image_original, os.path.basename(file), bbox_face, None, "label", idx, color)
			img_total = draw_image(image_total, os.path.basename(file), bbox_face, None, "label", idx, color)
		img.save(f"SAVE_DIR/{os.path.basename(file)[:-4]}_{idx}.jpg")
	image_total.save(f"SAVE_DIR/{os.path.basename(file)[:-4]}_total.jpg")
	if unassigned_persons_inds:
		image_total = Image.open(file).convert("RGB")
		for idx, person_ind in enumerate(unassigned_persons_inds):
			image_original = Image.open(file).convert("RGB")
			bbox_person = result_labels[person_ind][0]
			#img = draw_image(image_original, os.path.basename(file), None, bbox_person, "label", idx, color)
			image_total = draw_image(image_total, os.path.basename(file), None, bbox_person, "label", idx, colors_unassigned_persons)
			#img.save(f"/home/tobi/Desktop/test_{idx}.jpg")
		image_total.save(f"SAVE_DIR/{os.path.basename(file)[:-4]}_persons.jpg")

