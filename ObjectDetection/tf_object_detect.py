
import os
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from matplotlib import pyplot as plt
from PIL import Image

CWD_PATH = os.getcwd()
# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

class tfImageObjectDetect(object):
	def __init__(self, **options):
		# Loading label map
		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
        		                                                    use_display_name=True)
		self.category_index = label_map_util.create_category_index(categories)

		# Load a frozen TF model
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

    # Private function for detect object from image_np array
	def detect_objects(self, image_np, sess):
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		# Actual detection.
		(boxes, scores, classes, num_detections) = sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})

		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			np.squeeze(boxes),
			np.squeeze(classes).astype(np.int32),
			np.squeeze(scores),
			self.category_index,
			use_normalized_coordinates=True,
			line_thickness=8)
		return image_np

    # Private function to load image to numpy array
	def load_image_into_numpy_array(self, image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	def save_numpy_array_into_image(self, image_np, image_path):
		img = Image.fromarray(image_np, 'RGB')
		img.save(image_path)

	# Public public function to dipslay image from path -- not use for now
	def show_image_from_path(self, image_file_path):
		for image_path in image_file_path:
			image = Image.open(image_path)
			image_np = load_image_into_numpy_array(image)
			plt.imshow(image_np)
			print(image.size, image_np.shape)

	# Public function to do object detect on input image - one image at a time!
	# start tensorflow image detections
	def image_object_detect(self, image_path, dst_image_path):
		with self.detection_graph.as_default():
			with tf.Session(graph=self.detection_graph) as sess:
				image = Image.open(image_path)
				image_np = self.load_image_into_numpy_array(image)
				image_process = self.detect_objects(image_np, sess)
				print(image_process.shape)
				self.save_numpy_array_into_image(image_process, dst_image_path)
