'''
To register COCO datasets we need to import register_coco_instances from data.datasets
register_coco_instances automatically registers the data
register_coco_instances(nameofData,fileType: ususaly dict, json file location, image file location)

to get the registered metadata and datasets need to import MetadataCatalog and DataCatalog from data
the call the get function with the name
'''

#import for register and getting datasets
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog,MetadataCatalog

#import for visualizing datasets
import random
from detectron2.utils.visualizer import Visualizer
import cv2

#train the model R50-FPN Mask R-CNN model
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
##################################################
##################################################

#Register the dataset and visualize the dataset
register_coco_instances("fruits_nuts",{},"./data/trainval.json","./data/images")
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
fruits_nuts_data = DatasetCatalog.get("fruits_nuts")

f = DatasetCatalog._REGISTERED["fruits_nuts"]
print(f)

for d in random.sample(fruits_nuts_data,0):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:,:,::-1],metadata=fruits_nuts_metadata,scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("image", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)

#print the data and metadata
print(fruits_nuts_metadata)
print(fruits_nuts_data)


##################################################
##################################################

#train the model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = 300
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

'''
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
'''
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("fruits_nuts", )
predictor = DefaultPredictor(cfg)

for d in random.sample(fruits_nuts_data, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)


