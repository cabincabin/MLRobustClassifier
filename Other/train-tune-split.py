'''
#####

INPUT: two dicts one with the training data and the other with the tuning data
VARIABLES: 
    ImageLabelDict_tune= tuning data dict
    ImageLabelDict= training data dict
    
#####
'''



###Initialize data strcutures
X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

'''
### Using dict for training data import image vectors
'''
keys=list(ImageLabelDict.keys())
label=0
for val in keys:
  imgs=ImageLabelDict.get(val)
  labels.append(val)
  for img in imgs:
        vec=img[0]
        X_train.append(vec)
        Y_train.append(label)
  label+=1
  
'''  
#Extract classes from tuning labels
'''
IdsFromLabels_tune = CreateDictLabels(points_tune)
idsToRemoveFromEachBatch_tune = IdsFromLabels_tune.copy()
remove=[]
for val in IdsFromLabels_tune.items():
  if val[0] not in classes:
    remove.append(val[0])
for val in remove:  
  del IdsFromLabels_tune[val]
  
  
'''  
####create list of classes included in the tuning dict   
'''
tuning_class=[]
for val in IdsFromLabels_tune.items():
  tuning_class.append(val[0])


'''  
#####  modified from previous example because needed to exclude class labels 
#####  for which there are no images
'''
for val in tuning_class:
  imgs=ImageLabelDict_tune.get(val)
  for img in imgs:
      vec=img[0]
      X_test.append(vec)
      label=labels.index(val)
      Y_test.append(label)

'''  
##### modeling code expects np array. convert from list to np arrray 

'''      
img_len=256
img_width=256

X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_test=np.array(X_test)
Y_test=np.array(Y_test)





"""
####

WHERE THIS CODE GOES:
    
    ----This code should replace the code directly under main
    ----The code block this is replacing ends right before the following commands:
         
    # We reshape the input data to have a depth of 1 (grey scale)
    if keras.backend.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_len, img_width)
        X_test = X_test.reshape(X_test.shape[0], 1, img_len, img_width)
        input_shape = (1, img_len, img_width)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_len, img_width, 1)
        X_test = X_test.reshape(X_test.shape[0], img_len, img_width, 1)
        input_shape = (img_len, img_width, 1)
    

####
""" 