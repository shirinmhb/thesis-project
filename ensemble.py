import time
import numpy as np
import pandas as pd
import pickle,random
from sklearn.metrics.pairwise import rbf_kernel
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
np.set_printoptions(threshold=sys.maxsize)
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransKernelZSL:
    def __init__(self, X_train_trans, X_train_val, X_test_val, class_attributes, classes_seen, classes_unseen,
        classes_all, X_test_seen, X_test_unseen, 
        iter_num_total=10, lambda_=0.4, margin=0.05, gamma=1/2048, alpha=0.03, samples_per_iteration=500, learning_rate=0.3):
        
        self.X_train_trans = X_train_trans
        self.X_train_val = X_train_val
        self.X_test_val = X_test_val
        self.class_attributes = class_attributes
        self.classes_seen = classes_seen
        self.classes_unseen = classes_unseen
        self.classes_all = classes_all
        self.X_test_seen = X_test_seen
        self.X_test_unseen = X_test_unseen

        self.lambda_ = lambda_
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha
        self.iter_num_total = iter_num_total
        self.samples_per_iteration = samples_per_iteration
        self.learning_rate = learning_rate
        
    
    def _estimate_mius(self, X_, classes_):
        miu_matrix = np.zeros((len(classes_),X_.shape[1]-1))
        for class_index in range(len(classes_)):
            class_ = classes_[class_index]
            X_class = X_[X_[:,-1]==class_][:,:-1]
            if len(X_class) == 0:
                miu = np.zeros((X_class.shape[1],1))
            else:
                miu = (np.sum(X_class,axis=0)/len(X_class))[:,np.newaxis]
            miu_matrix[class_index] = miu.ravel()
        return miu_matrix

    def _estimate_kernel_semantics(self, target_semantics, source_semantics):
        K = np.power(1+target_semantics @ source_semantics.T,2)
        return K

    def _calculate_kernel_transformer(self, semantics, mius, lambda_):
        W = np.linalg.inv(semantics + lambda_*np.identity(semantics.shape[0])) @ mius 
        return W.T

    def calculate_top1_average_accuracy(self, X_, M, prototypes, classes_, all_classes, coeffs = None):
        acc_sum = 0
        num_nonempty_classes = 0
        for class_ in classes_:
            X_class = X_[X_[:,-1]==class_][:,:-1]
            if len(X_class) == 0:
                continue
            num_nonempty_classes += 1
            X_M = X_class @ M.T
            rbf = rbf_kernel(X_M, prototypes, gamma = self.gamma)
            if coeffs is not None: #alpha is set
                predicted_indexes = np.argmin(-2*rbf+coeffs,axis = 1)
            else:
                predicted_indexes = np.argmin(-2*rbf,axis = 1)
            predicted_labels = all_classes[predicted_indexes]
            acc = np.count_nonzero(predicted_labels==class_) / len(X_class)
            acc_sum += acc
        return acc_sum/num_nonempty_classes

    def _calculate_harmonic_mean(self, acc1, acc2):
        return 2*acc1*acc2 / (acc1+acc2)

    def _calculate_hinge_loss(self, rbf_x_correct_label, rbf_x_opposing_label):
        return max(0,(2 * (rbf_x_opposing_label-rbf_x_correct_label)+self.margin))

    def _update_M(self, M, x, M_x, correct_class_protoype, opposing_label_prototype, rbf_x_correctLabel, rbf_x_opposingLabel, samples_num):
        part1 = (rbf_x_correctLabel-rbf_x_opposingLabel) * M_x @ x.T
        part2 = (rbf_x_opposingLabel * opposing_label_prototype - rbf_x_correctLabel * correct_class_protoype) @ x.T
        part1 += part2 
        return M - 4 * self.gamma * self.learning_rate /samples_num * part1


    def _tune_M(self, prototypes_train, class_num_to_index, M, S): 
        classes_number = len(self.classes_all)
        np.random.shuffle(self.X_train_trans)
        X_selected = self.X_train_trans[:self.samples_per_iteration]
        
        for x in X_selected:
            x_label = int(x[-1])
            x_features = x[:-1][:,np.newaxis]
            
            current_class_sample_num = len(self.X_train_trans[self.X_train_trans[:,-1]==x_label])
            x_label_index = class_num_to_index[x_label]
            correct_class_protoype = prototypes_train[x_label_index][:,np.newaxis]            
            M_x = M @ x_features
            
            for k in range(1,S):
                opposing_label_index = random.randint(0,classes_number-1)
                opposing_label = self.classes_all[opposing_label_index]
                
                while opposing_label == x_label_index:
                    opposing_label_index = random.randint(0,classes_number-1)
                    opposing_label = self.classes_all[opposing_label_index]
                
                opposing_label_prototype = prototypes_train[opposing_label_index][:,np.newaxis]
                rbf_x_correctLabel = rbf_kernel(M_x.T,correct_class_protoype.T).item()
                rbf_x_opposingLabel = rbf_kernel(M_x.T,opposing_label_prototype.T).item()
                hinge_loss = self._calculate_hinge_loss(rbf_x_correctLabel,rbf_x_opposingLabel)
                if hinge_loss > 0:
                    M = self._update_M(M,x_features,M_x,correct_class_protoype,opposing_label_prototype,rbf_x_correctLabel,rbf_x_opposingLabel, current_class_sample_num)
                    M_x = M @ x_features
        return M

    def _learn_parameters(self, prototypes_train, prototypes_test, prototypes_test_unseen, M):
        class_num_to_index = {}
        for i in range(len(self.classes_all)):
            class_num_to_index[self.classes_all[i]] = i
            
        coeffs_seen = self.alpha*np.ones((1,len(prototypes_train)))
        coeffs_unseen = 0*np.ones((1,len(prototypes_test_unseen)))
        coeffs = np.concatenate((coeffs_seen,coeffs_unseen),axis = 1)
        pre_train_trans_acc_seen = 0
        S = (len(self.classes_all)-1)        
        # f = open('cub-log-trans.txt','a+')
        for iter_num in range(1,self.iter_num_total+1):
            # print('iter '+str(iter_num)+' started')
            M = self._tune_M(prototypes_test, class_num_to_index, M, S)
            # train_trans_acc_seen = self.calculate_top1_average_accuracy(self.X_train_trans, M, prototypes_test, self.classes_all, self.classes_all)
            # train_acc_seen = self.calculate_top1_average_accuracy(self.X_train_trans, M, prototypes_test, self.classes_seen, self.classes_all)
            # test_acc_seen = self.calculate_top1_average_accuracy(self.X_test_seen, M, prototypes_train, self.classes_seen, self.classes_seen)
            # zsl_acc = self.calculate_top1_average_accuracy(self.X_test_unseen, M, prototypes_test_unseen, self.classes_unseen, self.classes_unseen)
            # gzsl_acc = self.calculate_top1_average_accuracy(self.X_test_unseen, M, prototypes_test, self.classes_unseen, self.classes_all, coeffs)
            # f.write('index:'+str(index)+'\n')
            # f.write('model:'+str(model)+'\n')
            # f.write('t:'+str(iter_num)+'\n')
            # f.write('train acc trans(all):'+str(train_trans_acc_seen)+'\n')
            # f.write('train acc seen:'+str(train_acc_seen)+'\n')
            # f.write('test acc seen:'+str(test_acc_seen)+'\n')
            # f.write('zsl acc:'+str(zsl_acc)+'\n')
            # f.write('Gzsl acc unseen:'+str(gzsl_acc)+'\n')
            # f.write('\n')
            # f.flush()
            # os.fsync(f.fileno())
            # if abs(pre_train_trans_acc_seen - train_trans_acc_seen) < 0.0002:
            #     break
            # pre_train_trans_acc_seen = train_trans_acc_seen
        # f.close()
        return M

    def kernel(self, initial_M, prototypes_train_val=None, prototypes_test_unseen=None, prototypes_test=None):
        
        if prototypes_train_val is None:
            attributes_train_val = self.class_attributes[self.classes_seen]
            semantics_train_val = self._estimate_kernel_semantics(attributes_train_val,attributes_train_val)
            prototypes_train_val = self._estimate_mius(self.X_train_val, self.classes_seen)

            kernel_transformer = np.float32(self._calculate_kernel_transformer(semantics_train_val, prototypes_train_val, self.lambda_)) #estimate kernel ridge parameters
            
            attributes_test_unseen = self.class_attributes[self.classes_unseen]
            semantics_test_unseen = self._estimate_kernel_semantics(attributes_test_unseen, attributes_train_val)
            prototypes_test_unseen = semantics_test_unseen @ kernel_transformer.T 
            prototypes_test = np.concatenate((prototypes_train_val,prototypes_test_unseen),axis = 0)

        M = self._learn_parameters(prototypes_train_val, prototypes_test, prototypes_test_unseen, initial_M)
        return {'M': M, 'P_train': prototypes_train_val, 'P_unseen': prototypes_test_unseen, 'P_test': prototypes_test}
    
    def predict_labels(self, X_, M, prototypes, all_classes, coeffs = None):
        X_M = X_ @ M.T
        rbf = rbf_kernel(X_M,prototypes,gamma = self.gamma)
        if coeffs is not None: #alpha is set
            predicted_indexes = np.argmin(-2*rbf+coeffs,axis = 1)
        else:
            predicted_indexes = np.argmin(-2*rbf,axis = 1)
        predicted_labels = all_classes[predicted_indexes]
        return predicted_labels

class KernelZSL:
    def __init__(self, X_train_val, X_test_val, class_attributes, classes_seen, classes_unseen,
        classes_test, X_test_seen, X_test_unseen, 
        iter_num_total=100, lambda_=0.4, margin=0.05, gamma=1/2048, alpha=0.03, samples_per_iteration=2000, learning_rate=0.3):
        
        self.X_train_val = X_train_val
        self.X_test_val = X_test_val
        self.class_attributes = class_attributes
        self.classes_seen = classes_seen
        self.classes_unseen = classes_unseen
        self.classes_test = classes_test
        self.X_test_seen = X_test_seen
        self.X_test_unseen = X_test_unseen

        self.lambda_ = lambda_
        self.margin = margin
        self.gamma = gamma
        self.alpha = alpha
        self.iter_num_total = iter_num_total
        self.samples_per_iteration = samples_per_iteration
        self.learning_rate = learning_rate
        
    
    def _estimate_mius(self, X_, classes_):
        miu_matrix = np.zeros((len(classes_),X_.shape[1]-1))
        for class_index in range(len(classes_)):
            class_ = classes_[class_index]
            X_class = X_[X_[:,-1]==class_][:,:-1]
            if len(X_class) == 0:
                miu = np.zeros((X_class.shape[1],1))
            else:
                miu = (np.sum(X_class,axis=0)/len(X_class))[:,np.newaxis]
            miu_matrix[class_index] = miu.ravel()
        return miu_matrix

    def _estimate_kernel_semantics(self, target_semantics, source_semantics):
        K = np.power(1+target_semantics @ source_semantics.T,2)
        return K

    def _calculate_kernel_transformer(self, semantics, mius, lambda_):
        W = np.linalg.inv(semantics + lambda_*np.identity(semantics.shape[0])) @ mius 
        return W.T

    def calculate_top1_average_accuracy(self, X_, M, prototypes, classes_, all_classes, coeffs = None):
        acc_sum = 0
        num_nonempty_classes = 0
        for class_ in classes_:
            X_class = X_[X_[:,-1]==class_][:,:-1]
            if len(X_class) == 0:
                continue
            num_nonempty_classes += 1
            X_M = X_class @ M.T
            rbf = rbf_kernel(X_M, prototypes, gamma = self.gamma)
            if coeffs is not None: #alpha is set
                predicted_indexes = np.argmin(-2*rbf+coeffs,axis = 1)
            else:
                predicted_indexes = np.argmin(-2*rbf,axis = 1)
            predicted_labels = all_classes[predicted_indexes]
            acc = np.count_nonzero(predicted_labels==class_) / len(X_class)
            acc_sum += acc
        return acc_sum/num_nonempty_classes

    def _calculate_harmonic_mean(self, acc1, acc2):
        return 2*acc1*acc2 / (acc1+acc2)

    def _calculate_hinge_loss(self, rbf_x_correct_label, rbf_x_opposing_label):
        return max(0,(2 * (rbf_x_opposing_label-rbf_x_correct_label)+self.margin))

    def _update_M(self, M, x, M_x, correct_class_protoype, opposing_label_prototype, rbf_x_correctLabel, rbf_x_opposingLabel, samples_num):
        part1 = (rbf_x_correctLabel-rbf_x_opposingLabel) * M_x @ x.T
        part2 = (rbf_x_opposingLabel * opposing_label_prototype - rbf_x_correctLabel * correct_class_protoype) @ x.T
        part1 += part2 
        return M - 4 * self.gamma * self.learning_rate /samples_num * part1


    def _tune_M(self, prototypes_train, class_num_to_index, M, S): 
        classes_number = len(self.classes_seen)
        np.random.shuffle(self.X_train_val)
        X_selected = self.X_train_val[:self.samples_per_iteration]
        
        for x in X_selected:
            x_label = int(x[-1])
            x_features = x[:-1][:,np.newaxis]
            
            current_class_sample_num = len(self.X_train_val[self.X_train_val[:,-1]==x_label])
            x_label_index = class_num_to_index[x_label]
            correct_class_protoype = prototypes_train[x_label_index][:,np.newaxis]            
            M_x = M @ x_features
            

            for k in range(1,S):
                opposing_label_index = random.randint(0,classes_number-1)
                opposing_label = self.classes_seen[opposing_label_index]
                
                while opposing_label == x_label_index:
                    opposing_label_index = random.randint(0,classes_number-1)
                    opposing_label = self.classes_seen[opposing_label_index]
                
                opposing_label_prototype = prototypes_train[opposing_label_index][:,np.newaxis]
                rbf_x_correctLabel = rbf_kernel(M_x.T,correct_class_protoype.T).item()
                rbf_x_opposingLabel = rbf_kernel(M_x.T,opposing_label_prototype.T).item()
                hinge_loss = self._calculate_hinge_loss(rbf_x_correctLabel,rbf_x_opposingLabel)
                if hinge_loss > 0:
                    M = self._update_M(M,x_features,M_x,correct_class_protoype,opposing_label_prototype,rbf_x_correctLabel,rbf_x_opposingLabel, current_class_sample_num)
                    M_x = M @ x_features
        return M

    def _learn_parameters(self, prototypes_train, prototypes_test, prototypes_test_unseen, M):
        S = (len(self.classes_seen)-1)
        class_num_to_index = {}
        for i in range(len(self.classes_seen)):
            class_num_to_index[self.classes_seen[i]] = i
            
        coeffs_seen = self.alpha*np.ones((1,len(prototypes_train)))
        coeffs_unseen = 0*np.ones((1,len(prototypes_test_unseen)))
        coeffs = np.concatenate((coeffs_seen,coeffs_unseen),axis = 1)
        
        S = (len(self.classes_seen)-1)        
        # f = open('log.txt','a+')
        train_acc_seen_prevous = 0
        for iter_num in range(1,self.iter_num_total+1):
            # print('iter '+str(iter_num)+' started')
            M = self._tune_M(prototypes_train, class_num_to_index, M, S)
        #     zsl_acc = self.calculate_top1_average_accuracy(self.X_test_unseen, M, prototypes_test_unseen, self.classes_unseen, self.classes_unseen)
        #     train_acc_seen = self.calculate_top1_average_accuracy(self.X_train_val, M, prototypes_train, self.classes_seen, self.classes_seen)
        #     test_acc_seen = self.calculate_top1_average_accuracy(self.X_test_seen, M, prototypes_train, self.classes_seen, self.classes_seen)
        #     test_acc_unseen_weighted = self.calculate_top1_average_accuracy(self.X_test_unseen, M, prototypes_test, self.classes_unseen, self.classes_test, coeffs)
        #     f.write('index: '+str(index)+'\n')
        #     f.write('model: '+str(model)+'\n')
        #     f.write('t:'+str(iter_num)+'\n')
        #     f.write('train acc seen:'+str(train_acc_seen)+'\n')
        #     f.write('test acc seen:'+str(test_acc_seen)+'\n')
        #     f.write('zsl acc:'+str(zsl_acc)+'\n')
        #     f.write('Gzsl acc unseen:'+str(test_acc_unseen_weighted)+'\n')
        #     f.write('\n')
        #     f.flush()
        #     os.fsync(f.fileno())
        #     if abs(train_acc_seen - train_acc_seen_prevous) <= 0.0001:
        #         break
        #     train_acc_seen_prevous = train_acc_seen
        #     # serializeObject(M,'M'+str(iter_num))
        #     # print('iter '+str(iter_num)+' done')
        # f.close()
        return M

    def kernel(self, initial_M, prototypes_train_val=None, prototypes_test_unseen=None, prototypes_test=None):
        
        if prototypes_train_val is None:
            attributes_train_val = self.class_attributes[self.classes_seen]
            semantics_train_val = self._estimate_kernel_semantics(attributes_train_val,attributes_train_val)
            prototypes_train_val = self._estimate_mius(self.X_train_val, self.classes_seen)

            kernel_transformer = np.float32(self._calculate_kernel_transformer(semantics_train_val, prototypes_train_val, self.lambda_)) #estimate kernel ridge parameters
            
            attributes_test_unseen = self.class_attributes[self.classes_unseen]
            semantics_test_unseen = self._estimate_kernel_semantics(attributes_test_unseen, attributes_train_val)
            prototypes_test_unseen = semantics_test_unseen @ kernel_transformer.T 
            prototypes_test = np.concatenate((prototypes_train_val,prototypes_test_unseen),axis = 0)

        M = self._learn_parameters(prototypes_train_val, prototypes_test, prototypes_test_unseen, initial_M)
        return {'M': M, 'P_train': prototypes_train_val, 'P_unseen': prototypes_test_unseen, 'P_test': prototypes_test}
    
    def predict_labels(self, X_, M, prototypes, all_classes, coeffs = None):
        X_M = X_ @ M.T
        rbf = rbf_kernel(X_M,prototypes,gamma = self.gamma)
        if coeffs is not None: #alpha is set
            predicted_indexes = np.argmin(-2*rbf+coeffs,axis = 1)
        else:
            predicted_indexes = np.argmin(-2*rbf,axis = 1)
        predicted_labels = all_classes[predicted_indexes]
        return predicted_labels
     
def readClassesSplit(seen_class_file,unseen_class_file,class_name_to_index_dic,dataset_name):
    seen_classes = []
    unseen_classes = []
    if dataset_name == 'SUN' or dataset_name == 'AWA' or dataset_name == 'AWA2' or dataset_name == 'APY':
        f = open(seen_class_file ,'r')
        for line in f:
            seen_classes.append(class_name_to_index_dic[line.split('\n')[0]])
        f.close()
        f = open(unseen_class_file ,'r')
        for line in f:
            unseen_classes.append(class_name_to_index_dic[line.split('\n')[0]])
        f.close()
        seen_class_labels = np.array(seen_classes) 
        unseen_class_labels = np.array(unseen_classes)
    elif dataset_name == 'CUB':
        f = open(seen_class_file ,'r')
        for line in f:
            seen_classes.append(int(line.split('.')[0]))
        f.close()
        f = open(unseen_class_file ,'r')
        for line in f:
            unseen_classes.append(int(line.split('.')[0]))
        f.close()
        seen_class_labels = np.array(seen_classes) - 1
        unseen_class_labels = np.array(unseen_classes) - 1
    return seen_class_labels,unseen_class_labels
            
def readFeaturesFromText(file_name):
    dataframe = pd.read_csv(file_name,header=None,delim_whitespace=True)   
    return dataframe.values
def readImagesClassesFromText(file_name):
    dataframe = pd.read_csv(file_name,header=None,delim_whitespace=True)   
    return dataframe.values -1
def readClassAttributes(file_name):
    dataframe = pd.read_csv(file_name,header=None,delim_whitespace=True)
    return (dataframe.values).astype(np.float32)  
def readImagesLocation(file_name):
    dataframe = pd.read_csv(file_name,header=None,delim_whitespace=True)
    return (dataframe.values-1).astype(int)  

def prepairData(dataset_name,visual_features_file,images_labels_file,classes_name_file,semantic_attributes_file,\
seen_classes_file,unseen_classes_file,train_val_loc_file,test_seen_loc_file,test_unseen_loc_file):
    
    images_features = readFeaturesFromText(visual_features_file)
    images_classes = readImagesClassesFromText(images_labels_file)
                   
    class_name_to_index_dic = {} #dictionary to convert name of classes to indexes
    classes = pd.read_csv(classes_name_file,header=None,delim_whitespace=True)[0]
    for i in range(classes.size):
        class_name_to_index_dic[classes[i]] = i 
    class_attributes = readClassAttributes(semantic_attributes_file)
    
    classes_seen,classes_unseen = readClassesSplit(seen_classes_file,unseen_classes_file,class_name_to_index_dic,dataset_name)
    
    classes_test = np.concatenate((classes_seen,classes_unseen),axis = 0)
    
    train_val_loc = readImagesLocation(train_val_loc_file)
    test_seen_loc = readImagesLocation(test_seen_loc_file)
    test_unseen_loc = readImagesLocation(test_unseen_loc_file)
    test_loc = np.concatenate((test_seen_loc,test_unseen_loc),axis = 0)
    
    X_with_labels = np.float32(np.concatenate((images_features,images_classes),axis = 1))
    
    X_train_val = X_with_labels[train_val_loc.ravel()]
    X_test = X_with_labels[test_loc.ravel()]
    X_test_seen = X_with_labels[test_seen_loc.ravel()]
    X_test_unseen = X_with_labels[test_unseen_loc.ravel()]
    
    return X_train_val, X_test, class_attributes, classes_seen, classes_unseen, classes_test, X_test_seen, X_test_unseen 

def serializeObject(object_,file_name):
    file_object = open(file_name,'wb')
    pickle.dump(object_, file_object,protocol = 2)
    file_object.close()
    return

def deserializeObject(file_name):
    file_object = open(file_name,'rb')
    object_ = pickle.load(file_object)
    file_object.close() 
    return object_

def cal_avg_accuracy(y_test, classes_unseen, predicted):
    import math
    sum_accuracy = 0
    classes = np.unique(y_test)
    for cls in classes:
        indices = np.where(y_test == cls)
        real_class_labels = y_test[indices]
        predicted_class_labels = predicted[indices]
        class_accuracy = np.sum(real_class_labels == predicted_class_labels) / len(real_class_labels)
        if not math.isnan(class_accuracy):
            sum_accuracy += class_accuracy

    average_per_class_accuracy = sum_accuracy / len(classes)
    return average_per_class_accuracy


class Discriminator:
    def __init__(self, input_dim, hidden_units=[128, 64], num_classes=2):
        """
        Initialize a simple feed-forward neural network.

        :param input_dim: number of features in the input data
        :param hidden_units: list of integers, number of hidden units for each hidden layer
        :param num_classes: number of output classes, default to 2 (1 for model1, 2 for model2)
        """
        self.model = keras.Sequential()
        # Input layer
        self.model.add(layers.Input(shape=(input_dim,)))
        # Hidden layers
        for units in hidden_units:
            self.model.add(layers.Dense(units, activation='relu'))
        # Output layer
        self.model.add(layers.Dense(num_classes, activation='softmax'))
        
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, X, labels, epochs=10, batch_size=32, validation_split=0.1, reset=False):
        """
        Append new data and labels to existing data, then fit.

        :param X: new input data
        :param labels: new labels for the input data
        :param epochs: number of training epochs
        :param batch_size: batch size for training
        :param validation_split: fraction of data to use as validation set
        :param reset: if True, resets the training data and starts fresh
        """
        if reset or not hasattr(self, 'training_data'):
            self.training_data = X
            self.training_labels = labels
        else:
            self.training_data = np.vstack([self.training_data, X])
            self.training_labels = np.hstack([self.training_labels, labels])

        self.model.fit(self.training_data, self.training_labels, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)
    
    def predict(self, X):
        """
        Predict the class for input data.

        :param X: input data
        :return: list of predicted class labels
        """
        predictions = np.argmax(self.model.predict(X, verbose=0), axis=1)
        return predictions


def performance_discr(X_, Y_, prototypes1, prototypes2, classes, all_classes=None, coeffs=None):

    if coeffs is not None:
        pred_model1 = kzsl1.predict_labels( X_, M1, prototypes1, all_classes, coeffs)
        pred_model2 = kzsl2.predict_labels( X_, M2, prototypes2, all_classes, coeffs)
    else:
        pred_model1 = kzsl1.predict_labels( X_, M1, prototypes1, classes)
        pred_model2 = kzsl2.predict_labels( X_, M2, prototypes2, classes)

    model1_correct_predictions = (pred_model1 == Y_).astype(int)
    model2_correct_predictions = (pred_model2 == Y_).astype(int)
    model1_corrects = np.where((model1_correct_predictions == 1) & (model2_correct_predictions == 0))[0]
    model2_corrects = np.where((model1_correct_predictions == 0) & (model2_correct_predictions == 1))[0]
    both_corrects = np.where((model1_correct_predictions == 1) & (model2_correct_predictions == 1))[0]
    both_incorrects = np.where((model1_correct_predictions == 0) & (model2_correct_predictions == 0))[0]

    N1 = len(model1_corrects) / len(Y_)
    N2 = len(model2_corrects) / len(Y_)
    N_plus = len(both_corrects) / len(Y_)
    N_minus = len(both_incorrects) / len(Y_)

    # print(N1, N2, N_plus, N_minus)

    X_discr = np.concatenate((X_[model1_corrects], X_[model2_corrects]), axis=0)
    Y_discr = np.concatenate((np.ones(shape=(model1_corrects.shape[0])), np.zeros(shape=(model2_corrects.shape[0]))), axis=0)
    Y_pred_discr = discr.predict(X_discr)

    # Compute the confusion matrix elements
    TP = np.sum((Y_discr == 1) & (Y_pred_discr == 1))
    TN = np.sum((Y_discr == 0) & (Y_pred_discr == 0))
    FP = np.sum((Y_discr == 0) & (Y_pred_discr == 1))
    FN = np.sum((Y_discr == 1) & (Y_pred_discr == 0))

    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * (precision * recall) / (precision + recall)

    return (N1, N2, N_plus, N_minus ,recall, specificity, F1, accuracy)


def softmax(x, temperature=0.1, axis=1):
    e_x = np.exp((x - np.max(x, axis=axis, keepdims=True)) / temperature)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def predict_label_score(X_, M, prototypes, all_classes, gamma, coeffs=None):
    X_M = X_ @ M.T
    rbf = rbf_kernel(X_M, prototypes, gamma=gamma)

    if coeffs is not None:
        distances = -2*rbf + coeffs
    else:
        distances = -2*rbf
    
    # Find the indexes of the minimum distances
    min_indexes = np.argmin(distances, axis=1)
    
    # Convert distances to softmax probabilities using the custom softmax
    softmax_scores = softmax(-distances)
    
    # Extract the most probable labels
    predicted_labels = all_classes[min_indexes]
    
    # Extract their associated softmax scores using fancy indexing
    scores = softmax_scores[np.arange(len(min_indexes)), min_indexes]

    # Return as a list of tuples (label, score)
    return list(zip(predicted_labels, scores))

def predict_ensemble(X_, Y_, classes, prototypes1, prototypes2, all_classes = None, coeffs = None):

    pred_discr = discr.predict(X_)
    X_pos = X_[pred_discr == 1]
    X_neg = X_[pred_discr == 0]
    Y_pos = Y_[pred_discr == 1]
    Y_neg = Y_[pred_discr == 0]
    if coeffs is not None: 
        y_pred_pos = kzsl1.predict_labels( X_pos, M1, prototypes1, all_classes, coeffs)
        y_pred_neg = kzsl2.predict_labels( X_neg, M2, prototypes2, all_classes, coeffs)
    else:
        y_pred_pos = kzsl1.predict_labels( X_pos, M1, prototypes1, classes)
        y_pred_neg = kzsl2.predict_labels( X_neg, M2, prototypes2, classes)
        
    avg_acc_pos = cal_avg_accuracy(Y_pos, classes, y_pred_pos)
    # acc_pos = np.mean(Y_pos == y_pred_pos)

    avg_acc_neg = cal_avg_accuracy(Y_neg, classes, y_pred_neg)
    # acc_neg = np.mean(Y_neg == y_pred_neg)

    Y_pos_neg = np.concatenate((Y_pos, Y_neg))
    Y_pred_pos_neg = np.concatenate((y_pred_pos, y_pred_neg))
    avg_acc = cal_avg_accuracy(Y_pos_neg, classes, Y_pred_pos_neg)
    # acc = np.mean(Y_pos_neg == Y_pred_pos_neg)
    return len(X_pos)/len(X_), len(X_neg)/len(X_), avg_acc_pos, avg_acc_neg, avg_acc


data_dictionary = deserializeObject('data_dict-cub')
X_train_val = data_dictionary["X_train_val"]
X_test_val = data_dictionary["X_test_val"]
class_attributes = data_dictionary["class_attributes"]
classes_seen = data_dictionary["classes_seen"]
classes_unseen = data_dictionary["classes_unseen"]
classes_all = data_dictionary["classes_test"]
X_test_seen = data_dictionary["X_test_seen"]
X_test_unseen = data_dictionary["X_test_unseen"]
index = 0
model = 1
index = 0

# #step1: train first model on all data and find pos and neg
# initial_M = np.float32(np.eye(2048))
# kzsl1 = KernelZSL( X_train_val, X_test_val, class_attributes, classes_seen, classes_unseen, classes_all, X_test_seen, X_test_unseen)
# model1 = kzsl1.kernel(initial_M)
# M1 = model1['M']
# prototypes_seen1 = model1['P_train']
# prototypes_unseen1 = model1['P_unseen']
# prototypes_all1 = model1['P_test']

# X_train = X_train_val[:,:-1]
# y_train = X_train_val[:,-1]
# predicted_labels = kzsl1.predict_labels( X_train, M1, prototypes_seen1, classes_seen)
# print("acc 1st model on all data", np.mean(predicted_labels == y_train))
# print("avg acc 1st model on all data", cal_avg_accuracy(y_train, classes_seen, predicted_labels))

# correct_flags = (predicted_labels == y_train).astype(bool)
# pos_train = X_train_val[correct_flags]
# neg_train = X_train_val[~correct_flags]
# print("pos", len(pos_train)/len(X_train), "neg", len(neg_train)/len(X_train))
# #validation:
# X_train_pos = pos_train[:,:-1]
# y_train_pos = pos_train[:,-1]
# y_predicted_pos = kzsl1.predict_labels( X_train_pos, M1, prototypes_seen1, classes_seen)
# print("acc 1st model on pos", np.mean(y_predicted_pos == y_train_pos))
# print("avg acc 1st model on pos", cal_avg_accuracy(y_train_pos, classes_seen, y_predicted_pos))
# model = 2
# #step2: train a copy of 1st model on negative instances as the 2nd model
# kzsl2 = KernelZSL( neg_train, X_test_val, class_attributes, classes_seen, classes_unseen, classes_all, X_test_seen, X_test_unseen)
# model2 = kzsl2.kernel(M1, prototypes_seen1, prototypes_unseen1, prototypes_all1)
# M2 = model2['M']
# prototypes_seen2 = model2['P_train']
# prototypes_unseen2 = model2['P_unseen']
# prototypes_all2 = model2['P_test']
# X_train_neg = neg_train[:,:-1]
# y_train_neg = neg_train[:,-1]
# y_predicted_neg = kzsl2.predict_labels( X_train_neg, M2, prototypes_seen2, classes_seen)
# print("acc 2nd model on neg", np.mean(y_predicted_neg == y_train_neg))
# print("avg acc 2nd model on neg", cal_avg_accuracy(y_train_neg, classes_seen, y_predicted_neg))

# serializeObject(model1, "m1_initial_new_cub")
# serializeObject(model2, "m2_initial_new_cub")

model1 = deserializeObject("m1_initial_new_cub")
model2 = deserializeObject("m2_initial_new_cub")
kzsl1 = deserializeObject("kzsl1_initial")
kzsl2 = deserializeObject("kzsl2_initial")

M1 = model1['M']
prototypes_seen1 = model1['P_train']
prototypes_unseen1 = model1['P_unseen']
prototypes_all1 = model1['P_test']

M2 = model2['M']
prototypes_seen2 = model2['P_train']
prototypes_unseen2 = model2['P_unseen']
prototypes_all2 = model2['P_test']

coeffs_seen = kzsl1.alpha*np.ones((1,len(prototypes_seen1)))
coeffs_unseen = 0*np.ones((1,len(prototypes_unseen1)))
coeffs = np.concatenate((coeffs_seen,coeffs_unseen),axis = 1)

X_train = X_train_val[:,:-1]
y_train = X_train_val[:,-1]
X_gzsl = np.concatenate( (X_train_val[:,:-1], X_test_unseen[:,:-1]) )
Y_gzsl = np.concatenate( (X_train_val[:,-1], X_test_unseen[:,-1]) )
headers = "iteration, category, pos, neg, avg_acc_partition1_after_train, avg_acc_partition2_after_train, avg_acc_all_after_train,\n"

filename = "cub_results_trans.txt"
with open(filename, 'a') as f:
    # write headers for your data
    f.write(headers)

discr = Discriminator(input_dim=2048)
for index in range(1, 300):
    psuedo_labels_scores_model1 = predict_label_score( X_test_unseen[:,:-1], M1, prototypes_unseen1, classes_unseen, kzsl1.gamma)
    psuedo_labels_scores_model2 = predict_label_score( X_test_unseen[:,:-1], M2, prototypes_unseen2, classes_unseen, kzsl1.gamma)

    # Initialize lists to store the chosen labels, source models, and disagreement indexes
    chosen_labels = []
    source_models = []
    hard_samples_indexes = []
    # Iterate through the pseudo labels from both models
    #todo: count true labels
    for idx, (label1, label2) in enumerate(zip(psuedo_labels_scores_model1, psuedo_labels_scores_model2)):
        if label1[1] > label2[1]:  # Compare the softmax scores
            # Choose label from model1
            chosen_labels.append(label1[0])
            source_models.append(1)
        else:
            # Choose label from model2
            chosen_labels.append(label2[0])
            source_models.append(2)
        # Check if models disagree on labels
        if label1[0] != label2[0]:
            hard_samples_indexes.append(idx)

    # Convert lists to numpy arrays for any further processing if needed
    chosen_pseudo_labels = np.array(chosen_labels)
    source_models = np.array(source_models)
    hard_samples_indexes = np.array(hard_samples_indexes)
    labels_model1 = [label[0] for label in psuedo_labels_scores_model1]
    psuedo_labels_model1 = np.array(labels_model1)
    labels_model2 = [label[0] for label in psuedo_labels_scores_model2]
    psuedo_labels_model2 = np.array(labels_model2)

    # step 3: seperate instances based on correct prediction model1 and model2
    pred_model1 = kzsl1.predict_labels( X_train, M1, prototypes_seen1, classes_seen)
    pred_model2 = kzsl2.predict_labels( X_train, M2, prototypes_seen2, classes_seen)

    pred_trans_model1 = np.concatenate((pred_model1, psuedo_labels_model1[hard_samples_indexes]) )
    pred_trans_model2 = np.concatenate((pred_model2, psuedo_labels_model2[hard_samples_indexes]) )
    y_train_trans = np.concatenate((y_train, chosen_pseudo_labels[hard_samples_indexes]) )
    X_trans_discr = np.concatenate((X_train, X_test_unseen[hard_samples_indexes,:-1]) )

    model1_correct_predictions = (pred_trans_model1 == y_train_trans).astype(int)
    model2_correct_predictions = (pred_trans_model2 == y_train_trans).astype(int)
    model1_corrects = np.where((model1_correct_predictions == 1) & (model2_correct_predictions == 0))[0]
    model2_corrects = np.where((model1_correct_predictions == 0) & (model2_correct_predictions == 1))[0]
    both_corrects = np.where((model1_correct_predictions == 1) & (model2_correct_predictions == 1))[0]
    both_incorrects = np.where((model1_correct_predictions == 0) & (model2_correct_predictions == 0))[0]

    # N1 = len(model1_corrects) / len(y_train)
    # N2 = len(model2_corrects) / len(y_train)
    # N_plus = len(both_corrects) / len(y_train)
    # N_minus = len(both_incorrects) / len(y_train)

    # print(N1, N2, N_plus, N_minus)
    #step 4: train discriminator on N1 & N2
    X_discr = np.concatenate((X_trans_discr[model1_corrects], X_trans_discr[model2_corrects]), axis=0)
    Y_discr = np.concatenate((np.ones(shape=(model1_corrects.shape[0])), np.zeros(shape=(model2_corrects.shape[0]))), axis=0)
    discr.fit(X_discr, Y_discr)

    # Y_pred_discr = discr.predict(X_discr)

    # # Compute the confusion matrix elements
    # TP = np.sum((Y_discr == 1) & (Y_pred_discr == 1))
    # TN = np.sum((Y_discr == 0) & (Y_pred_discr == 0))
    # FP = np.sum((Y_discr == 0) & (Y_pred_discr == 1))
    # FN = np.sum((Y_discr == 1) & (Y_pred_discr == 0))

    # recall = TP / (TP + FN)
    # specificity = TN / (TN + FP)
    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    # precision = TP / (TP + FP)
    # F1 = 2 * (precision * recall) / (precision + recall)
    
    # performance_discr_tseen = performance_discr(X_test_seen[:,:-1], X_test_seen[:,-1], prototypes_all1, prototypes_all2, classes_seen, classes_all, coeffs)
    # performance_discr_zsl = performance_discr(X_test_unseen[:,:-1], chosen_pseudo_labels, prototypes_unseen1, prototypes_unseen2, classes_unseen)
    # performance_discr_gzsl = performance_discr(X_test_unseen[:,:-1], chosen_pseudo_labels, prototypes_all1, prototypes_all2, classes_unseen, classes_all, coeffs)
    
    # accs_zsl = predict_ensemble(X_test_unseen[:,:-1], X_test_unseen[:,-1], classes_unseen, prototypes_unseen1, prototypes_unseen2)
    # accs_train = predict_ensemble(X_train_val[:,:-1], X_train_val[:,-1], classes_seen, prototypes_seen1, prototypes_seen2)
    # accs_test_seen = predict_ensemble(X_test_seen[:,:-1], X_test_seen[:,-1], classes_seen, prototypes_all1, prototypes_all2, classes_all, coeffs)
    # accs_gzsl = predict_ensemble(X_test_unseen[:,:-1], X_test_unseen[:,-1], classes_unseen, prototypes_all1, prototypes_all2, classes_all, coeffs)

    #step5: partition all instances based on discriminator
    X_unseen = X_test_unseen[:,:-1]
    X_trans = np.concatenate((X_train, X_unseen) )
    Y_trans_pred_discr = discr.predict(X_trans)
    # Number of instances in X_train
    num_train = X_train.shape[0]
    # Predictions corresponding to X_train
    train_preds = Y_trans_pred_discr[:num_train]
    # Predictions corresponding to X_unseen
    unseen_preds = Y_trans_pred_discr[num_train:]
    # Find the desired indices
    train_indices_1 = np.where(train_preds == 1)[0]
    train_indices_0 = np.where(train_preds == 0)[0]
    unseen_indices_1 = np.where(unseen_preds == 1)[0]
    unseen_indices_0 = np.where(unseen_preds == 0)[0]
    X_unseen_pseudo_labeled = np.concatenate((X_test_unseen[:,:-1], chosen_pseudo_labels.reshape(-1, 1)),axis=1)
    # X_unseen_pseudo_labeled_model1 = np.concatenate((X_test_unseen[:,:-1], psuedo_labels_model1.reshape(-1, 1)),axis=1)
    # X_unseen_pseudo_labeled_model2 = np.concatenate((X_test_unseen[:,:-1], psuedo_labels_model2.reshape(-1, 1)),axis=1)
    data_pos = np.concatenate(( X_train_val[train_indices_1], X_unseen_pseudo_labeled[unseen_indices_1] ))
    data_neg = np.concatenate(( X_train_val[train_indices_0], X_unseen_pseudo_labeled[unseen_indices_0] ))

    #step 6: retrain the first model on pos
    model = 1
    kzsl1 = TransKernelZSL( data_pos, X_train_val, X_test_val, class_attributes, classes_seen, classes_unseen, classes_all, X_test_seen, X_test_unseen)
    model1 = kzsl1.kernel(M1, prototypes_seen1, prototypes_unseen1, prototypes_all1)

    M1 = model1['M']
    prototypes_seen1 = model1['P_train']
    prototypes_unseen1 = model1['P_unseen']
    prototypes_all1 = model1['P_test']

    #step 7: retrain the second model on neg
    model = 2
    kzsl2 = TransKernelZSL( data_neg, X_train_val, X_test_val, class_attributes, classes_seen, classes_unseen, classes_all, X_test_seen, X_test_unseen)
    model2 = kzsl2.kernel(M2, prototypes_seen2, prototypes_unseen2, prototypes_all2)

    M2 = model2['M']
    prototypes_seen2 = model2['P_train']
    prototypes_unseen2 = model2['P_unseen']
    prototypes_all2 = model2['P_test']

    accs_zsl_after_train = predict_ensemble(X_test_unseen[:,:-1], X_test_unseen[:,-1], classes_unseen, prototypes_unseen1, prototypes_unseen2)
    accs_train_after_train = predict_ensemble(X_train_val[:,:-1], X_train_val[:,-1], classes_seen, prototypes_seen1, prototypes_seen2)
    accs_test_seen_after_train =  predict_ensemble(X_test_seen[:,:-1], X_test_seen[:,-1], classes_seen, prototypes_all1, prototypes_all2, classes_all, coeffs)
    accs_gzsl_after_train = predict_ensemble(X_test_unseen[:,:-1], X_test_unseen[:,-1], classes_unseen, prototypes_all1, prototypes_all2, classes_all, coeffs)

    categories_results = {
        "train": accs_train_after_train,
        "gzsl_seen": accs_test_seen_after_train,
        "zsl": accs_zsl_after_train,
        "gzsl_unseen": accs_gzsl_after_train
    }

    with open(filename, 'a') as f:
        for category, results in categories_results.items():
            # Prepare the data string
            data_str = f"{index}, {category}, " + ", ".join([str(val) for val in results]) + "\n"
            f.write(data_str)

