import os
import random

#the path directory where are the images contained in oder directorys
path = '/Users/nicolas/workspace/machinelearning/retraining/tf_files/people_photos'
dataset_result_path = 'dataset'
name_dataset_file = 'dataset.txt'
name_datatest_file = 'datatest.txt'
name_labels_file = 'labels.txt'
valid_extensions = ['.jpg','.jpeg']

percentaje_of_data_to_test = 0.2

labels_string = []

#we iterate over the path to find directories
for f in os.listdir(path):
    if os.path.isdir(path + os.sep + f):
        labels_string.append(f)


label_id = 0
label_dict = {}
dataset_prepared_to_file = []
datatest_prepared_to_file = []
for label in labels_string:
    for f in os.listdir(path + os.sep + label):
        if os.path.isfile(path + os.sep + label + os.sep + f):
            name_no_extension, file_extension = os.path.splitext(path + os.sep + label + os.sep + f)
            if file_extension in valid_extensions:
                filename = path + os.sep + label + os.sep + f
                filename = filename +' '+ str(label_id)
                if random.random() > percentaje_of_data_to_test:
                    dataset_prepared_to_file.append(filename)
                else:
                    datatest_prepared_to_file.append(filename)
    label_dict[label_id] = label
    label_id += 1

print 'DataSet ' + str(len(dataset_prepared_to_file))
print 'DataTest ' + str(len(datatest_prepared_to_file))

print 'ID and Labels:'
for k,v in label_dict.iteritems():
    print str(k)+' '+str(v)

with open(dataset_result_path+os.sep+name_dataset_file,'w+') as data_set_file:
    for d in dataset_prepared_to_file:
        data_set_file.write(d+'\n')
    data_set_file.close()

with open(dataset_result_path+os.sep+name_datatest_file,'w+') as data_test_file:
    for d in datatest_prepared_to_file:
        data_test_file.write(d+'\n')
    data_test_file.close()

with open(dataset_result_path+os.sep+name_labels_file,'w+') as labels_file:
    for k,v in label_dict.iteritems():
        labels_file.write(str(k)+' '+str(v)+'\n')



# Load path/class_id image file:
# now is time to create the hdf5 dataset
dataset_file_to_hdf5 = dataset_result_path+os.sep+name_dataset_file
datatest_file_to_hdf5 = dataset_result_path+os.sep+name_datatest_file


# Build a HDF5 dataset (only required once)
from tflearn.data_utils import build_hdf5_image_dataset

build_hdf5_image_dataset(dataset_file_to_hdf5, image_shape=(32, 32), mode='file', output_path=dataset_result_path+os.sep+'dataset.h5', categorical_labels=True, normalize=True)
print 'Dataset HDF5 Ready'
build_hdf5_image_dataset(datatest_file_to_hdf5, image_shape=(32, 32), mode='file', output_path=dataset_result_path+os.sep+'datatest.h5', categorical_labels=True, normalize=True)
print 'Datatest HDF5 Ready'


#just for debbuging
#for d in dataset_prepared_to_file:
    #print d

#for t in datatest_prepared_to_file:
    #print t
