from matplotlib import pyplot as plt
from personlab import display, config
from personlab.data.coco import coco

base_dir = 'dataset/coco/'
anno_dir = base_dir + 'annotations/'
train_base_dir = base_dir + 'train2017/'
train_inst_json = anno_dir + 'instances_train2017.json'
train_person_json = anno_dir + 'person_keypoints_train2017.json'

gen = coco.CocoDataGenerator(train_base_dir, train_inst_json, train_person_json)

for dat in gen.read_dir_and_do_all():
    image, hm, seg_all, so_x, so_y, mo_x, mo_y, lo_x, lo_y = dat
    break

plt.rcParams['figure.figsize'] = [20, 10]
kp_i = 6
e_i = 6
print(config.KP_NAMES)
plt.subplot(2, 4, 1)
plt.imshow(display.show_heatmap(image, hm[..., kp_i]))
plt.subplot(2, 4, 2)
plt.imshow(display.show_heatmap(image, seg_all[..., 0]))
plt.subplot(2, 4, 3)
plt.imshow(so_x[..., kp_i])
plt.subplot(2, 4, 4)
plt.imshow(so_y[..., kp_i])
plt.subplot(2, 4, 5)
plt.imshow(mo_x[..., e_i])
plt.subplot(2, 4, 6)
plt.imshow(mo_y[..., e_i])
plt.subplot(2, 4, 7)
plt.imshow(lo_x[..., kp_i])
plt.subplot(2, 4, 7)
plt.imshow(lo_x[..., kp_i])
