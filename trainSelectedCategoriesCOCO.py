from pycocotools.coco import COCO
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import ColorMode

import os
import cv2
import random
"""
After downloading COCO dataset, It should have the following structure:
cocoDATA
    --Annotations
    --train2017
    --val2017

The code bellow will extract selected features from the annotations file and create a new json file
in coco format from those features.
"""
def getJsonFile(dataDir:str,imageFolderName:str,selectedCategories:list) -> str:
    """
    :param dataDir: path to coco data folder.
    :param imageFolderName: name of the image folder. There are two image folder. val2017 and train2017.
    :param selectedCategories: list of selected categories
    :return: the file name of the new json file created from the selected categories.
            json file is created in the working directory.
    """
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,imageFolderName)
    coco = COCO(annFile)
    categories = coco.loadCats(coco.getCatIds())
    nameOfCategories = [categorie['name'] for categorie in categories]
    print('COCO categories: \n{}\n'.format(' '.join(nameOfCategories)))
    nameOfSuperCategories = set([categorie['supercategory'] for categorie in categories])
    print('COCO supercategories: \n{}'.format(' '.join(nameOfSuperCategories)))

    categoryIds = coco.getCatIds(catNms=selectedCategories);
    imgIds = coco.getImgIds(catIds=categoryIds);
    annIds = coco.getAnnIds(imgIds, categoryIds)
    loadAnns = coco.loadAnns(annIds)
    loadCats = coco.loadCats(categoryIds)
    loadImgs = coco.loadImgs(imgIds)

    newAnnotationData = {'images': loadImgs, 'annotations': loadAnns, 'categories': loadCats}
    outputJsonFile ="_".join(selectedCategories)+'.json'
    with open(outputJsonFile, 'w') as fp:
        json.dump(newAnnotationData, fp)

    return outputJsonFile


def trainDataset(datasetName:str):
    """
    :param datasetName: name of the registered dataset
    :return: a output folder on the working directory containing "model_final.pth" - it takes a few minutes to create the folder
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (datasetName,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

def predictDataset(datasetName,dataset_data,dataset_metadata):
    """
    :param datasetName: name of the registered dataset
    :param dataset_data: DatasetCatalog of the registered dataset
    :param dataset_metadata: MetadatasetCatalog of the registerd data
    :return: shows 10 randomly selected pictures using opencv2. the ungemented portions are greyed out.
            press 0 key to see next image
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 300
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
    cfg.DATASETS.TEST = (datasetName,)
    predictor = DefaultPredictor(cfg)

    for d in random.sample(dataset_data, 10):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=dataset_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("image", v.get_image()[:, :, ::-1])
        cv2.waitKey(0)


dataDir='../cocoData'
imageFolderName='val2017'
imageRoot ='{}/{}'.format(dataDir,imageFolderName)
selectedCategories=['person', 'bicycle', 'car']

#get the annotation file for selected categories
#annotation file will look like annotationFile = "person_bicycle_car.json"
annotationFile = getJsonFile(dataDir,imageFolderName,selectedCategories)

datasetName = "_".join(selectedCategories)
register_coco_instances(datasetName,{},annotationFile,imageRoot)
dataset_metadata= MetadataCatalog.get(datasetName)
dataset_data = DatasetCatalog.get(datasetName)
print(datasetName)
trainDataset(datasetName)
predictDataset(datasetName,dataset_data,dataset_metadata)



