#To register COCO datasets we need to import register_coco_instances from data.datasets
#register_coco_instances automatically registers the data
#register_coco_instances(nameofData,fileType: ususaly dict, json file location, image file location)

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog,MetadataCatalog
register_coco_instances("fruits_nuts",{},"./data/trainval.json","./data/images")
fruits_nuts_metadata = MetadataCatalog.get("fruits_nuts")
fruits_nuts_data = DatasetCatalog.get("fruits_nuts")

print(fruits_nuts_metadata)
print(fruits_nuts_data)
