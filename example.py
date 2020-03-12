
# Evaluation Script
import tensorflow as tf
from personlab.model import train
from personlab.model import evaluate
from personlab.models.mobilenet_v2 import mobilenet_v2_model
from personlab.data import coco
from matplotlib import pyplot as plt
from personlab import display, config

base_dir = 'dataset/coco/'
anno_dir = base_dir + 'annotations/'
train_base_dir = base_dir + 'train2017/'
val_base_dir = base_dir + 'val2017/'

train_inst_json = anno_dir + 'instances_train2017.json'
train_person_json = anno_dir + 'person_keypoints_train2017.json'
val_inst_json = anno_dir + 'instances_val2017.json'
val_person_json = anno_dir + 'person_keypoints_val2017.json'

pm_check_path = 'weights/mobilenet/mobilenet_v2_1.0_224.ckpt'
log_dir = 'logs/sample/'

# ------------------ [Training script] -----------------------
tf.enable_eager_execution() # must enable eager mode immedately after importing

train_gen = coco.CocoDataGenerator(train_base_dir, train_inst_json, train_person_json)
train(mobilenet_v2_model, train_gen.loader, pm_check_path, log_dir)


# ------------------ [Evaluation script] -----------------------
tf.reset_default_graph()
latest_ckp = tf.train.latest_checkpoint('./')
val_gen = coco.CocoDataGenerator(val_base_dir, val_inst_json, val_person_json)
checkpoint_path = tf.train.latest_checkpoint(log_dir)
output = evaluate(mobilenet_v2_model, val_gen.loader, checkpoint_path, num_batches=2)

# ----------------------------------

plt.rcParams['figure.figsize'] = [20, 20]
b_i = 5
plt.figure()
plt.title('Original Image')
plt.imshow(output['image'][b_i])

plt.figure()

plt.subplot(2, 2, 1)
plt.title('Skeleton(True)')
plt.imshow(display.summary_skeleton(output['image'][b_i], output['kp_map_true'][b_i]))

plt.subplot(2, 2, 2)
plt.title('Segmentation(True)')
plt.imshow(display.show_heatmap(output['image'][b_i], output['seg_true'][b_i]))

plt.subplot(2, 2, 3)
plt.title('Skeleton(Prediction)')
plt.imshow(display.summary_skeleton(output['image'][b_i], output['kp_map_pred'][b_i]))

plt.subplot(2, 2, 4)
plt.title('Segmentation(Prediction)')
plt.imshow(display.show_heatmap(output['image'][b_i], output['seg_pred'][b_i]))
