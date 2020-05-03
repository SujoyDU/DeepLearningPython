import json

# with open("../cocoData/annotations/instances_train2017.json","r") as f:
#     data = json.load(f)

with open("./data/trainval.json","r") as f:
    data = json.load(f)

categoryList={}
#get all the categories name
for d in data['categories']:
    categoryList.update({d['name']:d['id']})

# print(categoryList)

imageCatid = {}
for d in data['annotations']:
    if d['image_id'] not in imageCatid.keys():
        imageCatid.update({d['image_id']:[{d['id']:d['category_id']}]})

    else: imageCatid[d['image_id']].append({d['id']:d['category_id']})

# print(imageCatid)

#get those annotations that have hazelnut in them
annoId = categoryList['hazelnut']
# print(annoId)
# imageCatid[val][10]

newAnnotations=[]
xList = list(imageCatid.keys())
for idx in xList:
    for l in imageCatid[idx]:
        for x in l.values():
            if x==annoId:
                for k in l.keys():
                    print(k)
                    for d in data['annotations']:
                        if d['id']==k:
                            newAnnotations.append(d)


# print(newAnnotations)

newImages=[]
flag =0
for idx in xList:
    flag=1
    for l in imageCatid[idx]:
        if(flag==1):
            for x in l.values():
                if x==annoId:
                    flag=0
                    for d in data['images']:
                        if d['id']==idx:
                            newImages.append(d)
                            break






# print(newImages)
newData={'images':newImages,'annotations':newAnnotations,'categories':data['categories']}
print(newData)

with open('result.json', 'w') as fp:
    json.dump(newData, fp)

# for val in imageCatid.keys():
#     for a in imageCatid[val]:
#         if a.values()==annoId:
#             print(a)




