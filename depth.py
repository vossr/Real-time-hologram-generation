from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import numpy as np
import torch
import cv2

model_name = "Intel/dpt-large"
# model_name = "Intel/dpt-hybrid-midas"
processor = DPTImageProcessor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)

device = "cuda"
if torch.cuda.is_available():
	model.to(device)
else:
	print("no cuda")
	exit(-1)


#io cv2 rgb
def depth_estimation(frame):
	image_pil = Image.fromarray(frame)
	inputs = processor(images=image_pil, return_tensors="pt").to(device)

	# with torch.no_grad():
	with torch.inference_mode():
		outputs = model(**inputs)
		predicted_depth = outputs.predicted_depth

	# resize to original
	prediction = torch.nn.functional.interpolate(
		predicted_depth.unsqueeze(1),
		size=image_pil.size[::-1],
		mode="bicubic",
		align_corners=False,
	)

	output = prediction.squeeze().cpu().numpy()
	formatted = (output * 255 / np.max(output)).astype("uint8")
	depth_pil = Image.fromarray(formatted)
	depth = cv2.cvtColor(np.array(depth_pil), cv2.COLOR_GRAY2RGB)
	return depth
