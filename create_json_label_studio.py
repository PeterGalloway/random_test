for file in tqdm(files[:20], total=len(files[:20])):
	image = Image.open(file).convert("RGB")
	width, height = image.size
	target_sizes=[(width, height)]
	inputs = dino_processor(text=queries, images=image, return_tensors="pt").to("cuda")
	with torch.no_grad():
			outputs = dino_model(**inputs)
	results = dino_processor.post_process_grounded_object_detection(
			outputs,
			inputs.input_ids,
			box_threshold=score_threshold,
			text_threshold=0.3,
			target_sizes=[image.size[::-1]]
	)
	boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
	#file = file.replace("/home/tobi/.local/share/label-studio/media","/data")
	file = f"LOAD IMAGE_DIRECTORY/{os.path.basename(file)}"
	image.save("SAVE_PATH" + os.path.basename(file))
	s = "{\n"
	s += "\"data\": {\n"
	s += f"\"url\": \"{file}\"\n"
	s += "},\n"
	s += "\"predictions\": [\n"
	s += "{\n"
	s += "\"model_version\": \"version 1\",\n"
	s += "\"score\": 0.5,\n"
	s += "\"result\": [\n"
	for _, (box, score, label) in enumerate(zip(boxes, scores, labels)):
			s += "{\n"
			s += f"\"original_width\": {width},\n"
			s += f"\"original_height\": {height},\n"
			s += f"\"image_rotation\":0,\n"
			s += "\"value\": {\n"
			s += f"\"x\": {(box[0]/width)*100},\n"
			s += f"\"y\": {(box[1]/height)*100},\n"
			s += f"\"width\": {((box[2]-box[0])/width)*100},\n"
			s += f"\"height\": {((box[3]-box[1])/height)*100},\n"
			s += f"\"rotation\": 0,\n"
			s += f"\"rectanglelabels\": [\n"
			s += f"\"{label}\"\n"
			s += f"]\n"
			s += "},\n"
			s += "\"from_name\": \"label\","
			s += "\"to_name\": \"image\","
			s += f"\"type\": \"rectanglelabels\",\n"
			s += f"\"origin\": \"Dino\"\n"
			s += "},\n"
	s = s[:-2]
	s += "]\n}\n]\n}"
	with open("SAVE_JSON_DIRECTORY/" + os.path.basename(file)[:-3] + "json", "w") as f:
		f.write(s)