from PIL import Image, ImageFile, ImageDraw
from shapely.geometry import Polygon
import pandas as pd

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def compute(bbox_face, bbox_person):
	poly_1 = Polygon(bbox_face)
	poly_2 = Polygon(bbox_person)
	iou = poly_1.intersection(poly_2).area / poly_1.area
	return iou

def draw_image(img, file_name, bbox_face, bbox_person, age, idx):
	draw = ImageDraw.Draw(img)
	x_min, y_min, x_max, y_max = bbox_face
	shapely_face = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
	draw.rectangle((x_min, y_min, x_max, y_max), outline=(255, 255, 255), width=3)
	x_min, y_min, x_max, y_max = bbox_person
	shapely_person= [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
	draw.rectangle((x_min, y_min, x_max, y_max), outline=(0, 255, 255), width=3)
	iou = compute(shapely_face, shapely_person)
	if iou != 1:
		draw.text((x_min, y_min), f'{idx} ; IoU: {iou}')
		print(idx, file_name, iou)
		img.save(f"SAVE_IMAGE_DIRECTORY/{idx}.jpg")


df_path = "FILE_DIRECTORY.csv"
df = pd.read_csv(df_path).sample(1000)
for idx, row in df.iterrows():
	img = pil_loader(f"DIRECTORY_TO_IMAGES/{row.img_name}")
	bbox_face = [row.face_x0, row.face_y0, row.face_x1, row.face_y1]
	bbox_person = [row.person_x0, row.person_y0, row.person_x1, row.person_y1]
	draw_image(img, row.img_name, bbox_face, bbox_person, row.age, idx)
