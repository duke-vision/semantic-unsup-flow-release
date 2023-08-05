import numpy as np
import cv2

from utils.cityscape_labels import labels

train_labels = {'name': [], 'id': [], 'category': [], 'category_id': [], 'color': []}
id2trainId = []
cateId2trainId = [] # we use the first class in each category to represent that category

for label in labels:
    if not label.ignoreInEval:
        train_labels['name'].append(label.name)
        train_labels['id'].append(label.id)
        train_labels['category'].append(label.category)
        train_labels['category_id'].append(label.categoryId)
        train_labels['color'].append(label.color)
    id2trainId.append(label.trainId)
    if label.categoryId >= len(cateId2trainId):
        cateId2trainId.append(label.trainId)
        
train_labels['id'] = np.array(train_labels['id'])
train_labels['category_id'] = np.array(train_labels['category_id'])
train_labels['color'] = np.array(train_labels['color'], dtype=np.uint8)
id2trainId = np.array(id2trainId, dtype=np.uint8)
cateId2trainId = np.array(cateId2trainId, dtype=np.uint8)

num_class = len(train_labels['id'])  # should be 19
num_category = len(cateId2trainId)  # should be 8

def read_semantics(sem_file):
    sem = cv2.imread(sem_file, cv2.IMREAD_UNCHANGED)
    sem[sem == 255] = 0
    res = id2trainId[sem]
    return res, res != 255
   
def trainId2color(sem, sem_valid=None):
    cmap = np.vstack((train_labels['color'], [255, 255, 255])) # add the invalid color
    sem_map = sem.astype(int)
    
    if sem_valid is not None:
        sem_map[~sem_valid] = -1
    
    return cmap[sem_map]

def trainId2cateId(sem_map):
    valid = (sem_map >= 0) * (sem_map < num_class)
    sem_map[~valid] = 0
    res = train_labels['category_id'][sem_map]
    res[~valid] = 0
    return res

def cateId2color(sem_cate, sem_valid=None):
    if sem_valid is None:
        sem_valid = (sem_cate >= 0) * (sem_cate < num_category)
    
    sem_cate[~sem_valid] = 0
    sem = cateId2trainId[sem_cate]
    
    cmap = np.vstack((train_labels['color'], [255, 255, 255])) # add the invalid color
    sem_valid = (sem >= 0) * (sem < num_class)
    sem[~sem_valid] = -1
    
    return cmap[sem]
    