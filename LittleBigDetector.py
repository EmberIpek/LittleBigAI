import cv2
import time
import onnxruntime as ort
import numpy as np

t1=time.time()
#cv2.waitKey(0)
cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Algorithm will consider {bird, cat, dog, frog} to be small objects,
# {airplane, automobile, deer, horse, ship, truck} to be large objects
big_objects_set = {'airplane', 'automobile', 'deer', 'horse', 'ship', 'truck'}
little_objects_set = {'bird', 'cat', 'dog', 'frog'}

FPS=1.0

video = cv2.VideoCapture(0, cv2.CAP_V4L2)
# shrink video size -> better fps
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

patch_size = 128
step_size = 64
# patch_sizes = [64, 128]
# step_factors = [0.5, 0.5]

#EmberNet = cv2.dnn.readNet("EmberNet.onnx")
# OpenCV is not cooperating, use ONNX runtime
ort_session = ort.InferenceSession("EmberNet.onnx")

try:
	while True:
		t2=time.time()
		
		success, img = video.read()
		img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img_edges = cv2.Canny(img, 50, 200)
		big_object_found = []
		
		# for patch_size, step_factor in zip(patch_sizes, step_factors):
		# 	step_size = int(patch_size * step_factor)
		for y in range(0, img.shape[0]-patch_size+1, step_size):
			for x in range(0, img.shape[1]-patch_size+1, step_size):
				patch = img[y:y+patch_size, x:x+patch_size]
				# # EmberNet expects 32×32 RGB images

				# doesn't work, opencv did not like embernet kernel sizes
				# blob = cv2.dnn.blobFromImage(cv2.resize(patch,(32,32)), 1/255.0)
				# EmberNet.setInput(blob)
				# out = EmberNet.forward()
				# pred_class = int(np.argmax(out))
				# pred_class_name = cifar10_classes[pred_class]


				# use canny edge detection to skip blank/low texture regions before classification
				patch_edges = cv2.Canny(patch, 50, 200)
				if np.mean(patch_edges) < 25:     # increase threshold
					continue

				# classify only meaningful patches
				patch_input = cv2.resize(patch, (32,32)).astype(np.float32) / 255.0
				patch_input = np.transpose(patch_input, (2,0,1))[np.newaxis, :, :, :]

				out = ort_session.run(None, {"input": patch_input})
				scores = out[0][0]
				pred_class = int(np.argmax(scores))
				pred_class_name = cifar10_classes[pred_class]
				pred_score = float(scores[pred_class])

				# confidence threshold
				if ((pred_class_name in big_objects_set) and (pred_score > 0.1)):
					big_object_found.append((x, y, patch_size, patch_size, pred_class_name))
		
		#face_found = face_cascade.detectMultiScale(img_gray, minSize=(20,20))
		for(x, y, w, h, class_name) in big_object_found:
			cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
			cv2.putText(img, class_name, (x+w, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0) )
			#cv2.rectangle(img_edges, (x, y), (x+w, y+h), (255,0,0), 5)
		
		FPSstr=f"{FPS:3.1f} fps"
		cv2.putText(img,FPSstr,(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
		cv2.imshow("CAPTURE", img)
		cv2.imshow("EDGES", img_edges)
		
		print(f"Frame Rate = {FPS:4.1f} fps",22*'\b',end='',flush=True)
		cv2.waitKey(1)
		t3=time.time()
		FPS=1.0/(t3-t2)
except KeyboardInterrupt:
	print("\n\nCapture is done!!!")

cv2.destroyAllWindows()