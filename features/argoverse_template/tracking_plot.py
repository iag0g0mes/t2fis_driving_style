import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import argoverse.visualization.visualization_utils as viz_util
import matplotlib.pyplot as plt #plotar
import cv2 #opencv
from PIL import Image

##set root_dir to the correct path to your dataset folder
root_dir =  '/home/iago/Documents/argoverse/data/tracking/sample/sample'

argoverse_loader = ArgoverseTrackingLoader(root_dir)

argoverse_data = argoverse_loader[0]
print(argoverse_data)

argoverse_data = argoverse_loader.get('c6911883-1843-3727-8eaa-41dc8cda8993')
print(argoverse_data)

camera = "ring_front_center"	
num_imgs = len(argoverse_data.image_list[camera])

for idx in range(0, num_imgs):

    img = argoverse_data.get_image_sync(idx, camera=camera)
    objects = argoverse_data.get_label_object(idx)
    calib = argoverse_data.get_calibration(camera)
    img_vis = Image.fromarray(viz_util.show_image_with_boxes(img, objects, calib))
    
    plt.imshow(img_vis)
    plt.title("Right Front Center")
    plt.axis("off")

    plt.pause(0.1)






