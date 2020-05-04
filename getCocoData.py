from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
pylab.rcParams['figure.figsize'] = (8.0,10.0)

dataDir='../cocoData'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))


catIds = coco.getCatIds(catNms=['person','car','bus']);
imgIds = coco.getImgIds(catIds=catIds );
annIds = coco.getAnnIds(imgIds,catIds)
loadAnns = coco.loadAnns(annIds)
loadCats = coco.loadCats(catIds)
loadImgs = coco.loadImgs(imgIds)

# print(newImages)
newData={'images':loadImgs,'annotations':loadAnns,'categories':loadCats}
# print(newData)

with open('person_car_bus_val2017.json', 'w') as fp:
    json.dump(newData, fp)

# print(imgIds)

#436738


# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# # load and display image
# I = io.imread('%s/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# # I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()

# load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
