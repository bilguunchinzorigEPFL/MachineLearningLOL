from sklearn.svm import SVC 
import numpy as np
import cv_helpers as hp

# Load data 

data,labels=hp.load_processed_data('training_cv2.npy')
print(len(data))
print(len(labels))
print(np.sum(data,axis=0))
data_test,labels_test=hp.load_processed_data('testing_cv2.npy')
print(np.sum(data_test,axis=0))

# Training
name='SCV'
model=SVC()
model.fit(data,labels)
print('Trained')
joblib.dump(model,'trained-models_monika/'+name)
print('trained-models_monika/'+name)
model=joblib.load()
print(np.mean(np.abs(labels_test-model.predict(data_test))))

