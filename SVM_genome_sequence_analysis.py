#!/usr/bin/env python
# coding: utf-8

# In[92]:


import matplotlib.pyplot as plt
import numpy as np

from libsvm.svmutil import *


train_data_path = "784_assignment_3_data/train.txt"
test_data_path = "784_assignment_3_data/test.txt"


# function responsible for importing filtering formating and pre-processing of data
train_x, train_y = data_processing(train_data_path)
test_x, test_y = data_processing(test_data_path)

print("train_x shape ",np.shape(train_x))
print("train_y shape ",np.shape(train_y))
print("test_x shape ",np.shape(test_x))
print("test_y shape ",np.shape(test_y))


# picking different values of C and sigma

C_list = np.array([0.001,0.01,0.1,1,5,10,50,60,70,100,100000000])
# for larger values of C we can see in execution that System is trying to overfit and accuracy is going down 
# as C value is very large processing time is also increasing as system is tring to fit every point while training
gamma_list = np.array([0.001,0.01,0.1,1,5,10,100])


#----------------------------------
# Part I
k_fold = 5
k_fold_train_x,k_fold_train_y,k_fold_validate_x,k_fold_validate_y = train_data_split(train_x,train_y,k_fold)

accuracy_matrix_linear = np.zeros((k_fold,len(C_list)))


print("For part A we are required to split data into",k_fold," parts")
for i in range(np.shape(k_fold_train_x)[0]):
    print("considering split ",i+1," for validation")
    for j in range(np.shape(accuracy_matrix_linear)[1]):
        C = C_list[j]
        predicted_linear_a,predicted_linear_b,predicted_linear_c = libsvm_prediction(k_fold_train_x[i],k_fold_train_y[i],k_fold_validate_x[i],k_fold_validate_y[i],False,C,True)
        accuracy_matrix_linear[i][j] = predicted_linear_b[0]
            
plots(accuracy_matrix_linear,False,C_list,False,True)




#----------------------------------
# Part II
k_fold = 2
# divide full train data in 2 set
cv_k_fold_train_x,cv_k_fold_train_y,cv_k_fold_validate_x,cv_k_fold_validate_y = train_data_split(train_x,train_y,k_fold)




# implementation of K_fold validation
k_fold = 5
k_fold_train_x,k_fold_train_y,k_fold_validate_x,k_fold_validate_y = train_data_split(cv_k_fold_train_x[0],cv_k_fold_train_y[0],k_fold)
#print("my train_data_split")
#print(np.shape(k_fold_train_x))
#print(np.shape(k_fold_train_y))
#print(np.shape(k_fold_validate_x))
#print(np.shape(k_fold_validate_y))

# managing accuracy matrix for each validation set and with 
#all combinations of C and sigma size must be k*len(C_list)*len(gamma_list)
accuracy_matrix_gaussian = np.zeros((k_fold,len(C_list),len(gamma_list)))


for i in range(np.shape(k_fold_train_x)[0]):
    print("considering split ",i+1," for validation")
    for j in range(np.shape(accuracy_matrix_gaussian)[1]):
        C = C_list[j]
        for k in range(np.shape(accuracy_matrix_gaussian)[2]):
            gamma = gamma_list[k]
            predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(k_fold_train_x[i],k_fold_train_y[i],k_fold_validate_x[i],k_fold_validate_y[i],gamma,C,False)
            accuracy_matrix_gaussian[i][j][k] = predicted_gaussian_b[0]

#print(accuracy_matrix_gaussian)

# function averaging accuracy of all valdidation set and ploting accuracy as a graph as a function of C
plots(False,accuracy_matrix_gaussian,C_list,gamma_list,False)


print("---------------------------------------------")
print("using best values of C and gamma values for llinear and RBF kernel. these results are on training data.....")
predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(cv_k_fold_train_x[0],cv_k_fold_train_y[0],cv_k_fold_validate_x[0],cv_k_fold_validate_y[0],0.1,60,False)
print("gaussian libsvm")
print(predicted_gaussian_b)

predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(cv_k_fold_train_x[1],cv_k_fold_train_y[1],cv_k_fold_validate_x[1],cv_k_fold_validate_y[1],0.1,60,False)
print("gaussian libsvm")
print(predicted_gaussian_b)





print("---------------------updating class labels of test file using optimal values of C and sigma")
predicted_linear_a,predicted_linear_b,predicted_linear_c = libsvm_prediction(train_x,train_y,test_x,test_y,0.1,60,True)
write_back_calculated_classes(predicted_linear_a,False,test_data_path,True)


predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(train_x,train_y,test_x,test_y,0.1,60,False)
write_back_calculated_classes(False,predicted_gaussian_a,test_data_path,False)

print("showing accuracy to be zero as we dont have the data on test file to check please ignore this")
print("done with file updation")


# In[ ]:





# In[90]:


predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(cv_k_fold_train_x[1],cv_k_fold_train_y[1],cv_k_fold_validate_x[1],cv_k_fold_validate_y[1],0.1,50,False)
print("gaussian libsvm")
print(predicted_gaussian_b)

predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = libsvm_prediction(cv_k_fold_train_x[0],cv_k_fold_train_y[0],cv_k_fold_validate_x[0],cv_k_fold_validate_y[0],0.1,50,False)
print("gaussian libsvm")
print(predicted_gaussian_b)


# In[9]:


print(len(predicted_linear_c))


# In[35]:


def write_back_calculated_classes(predicted_classes_linear,predicted_classes_gaussian,path,is_linear_svm):
    file = open(path,"r")
    
    if is_linear_svm == True:     
        new_file_path = "784_assignment_3_data/linear_svm_updated_test_file.txt"
    else:
        new_file_path = "784_assignment_3_data/rbf_svm_updated_test_file.txt"
        
    new_file = open(new_file_path,"w")
    
    
    counter = 0
    flag = True
    
    while flag:
        line = file.readline()
        if not line:
            print("End of file")
            flag = False
        else:
            splited_Data = line.split(" ")
            
            if is_linear_svm == True:
                splited_Data[0] = (str)(predicted_classes_linear[counter])
            else:
                splited_Data[0] = (str)(predicted_classes_gaussian[counter])
                
            
            updated_splited_Data = " ".join(splited_Data)
            
            new_file.write(updated_splited_Data)
            
            counter = counter + 1
    file.close()
    new_file.close()


# In[91]:


def plots(accuracy_matrix_linear,accuracy_matrix_gaussian,C_list,gamma_list,is_linear_svm):    
    X = C_list
    
    #### ploting for linear SVM 
    if is_linear_svm == True:
        Y = (np.sum(accuracy_matrix_linear, axis = 0))/np.shape(accuracy_matrix_linear)[0]
        print("C_list    ",C_list)
        print("Accuracy  ",Y)
        plt.scatter(X,Y)
        plt.plot(X,Y)
        plt.savefig("accuracy_VS_C")
        plt.show()

        max_index = Y.argmax()
        print("We have considered k_fold dataset and given accuracy is average of all of them")
        print("best accuracy for linear case is ",Y[max_index]," and best value of C is ",C_list[max_index])
    else:

        ##### Accuracy of RBF
        print("Average accuracy as a function of C and gamma over all validation set for RBF")
        accuracy_rbf = (np.sum(accuracy_matrix_gaussian, axis = 0))/np.shape(accuracy_matrix_gaussian)[0]
        print(accuracy_rbf)
        print("For RBF best values of C is 0.1 and best gamma value is 50")


# In[37]:


def libsvm_prediction(train_x,train_y,test_x,test_y,gamma,penalty,is_linear_svm):
    
    train = svm_problem(train_y,train_x)

    if is_linear_svm == True:
        linear_variables = svm_parameter("-s 0 -c "+ str(penalty) + " -t 0")
        linear_train = svm_train(train,linear_variables)
        predicted_linear_a,predicted_linear_b,predicted_linear_c = svm_predict(test_y,test_x,linear_train)
        return predicted_linear_a,predicted_linear_b,predicted_linear_c
    else:
        gaussian_variables = svm_parameter("-s 0 -c "+ str(penalty) + " -t 2 -g " + str(gamma))
        gaussian_train = svm_train(train,gaussian_variables)
        predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c = svm_predict(test_y,test_x,gaussian_train)
        return predicted_gaussian_a,predicted_gaussian_b,predicted_gaussian_c


# In[38]:


def train_data_split(train_data_x,train_data_y,k_fold):
    x_list = []
    y_list = []
    validate_x_list = []
    validate_y_list = []
    
    total_data_points = np.shape(train_data_y)[0]
    number_of_data_points_per_set = (int)(total_data_points/k_fold)
    
    
    for i in range(k_fold):
        temp = [ k for k in range(total_data_points)]
        indexes_to_remove = [ i*number_of_data_points_per_set + j for j in range(number_of_data_points_per_set)]
        
        indexes_to_remove_validate = list(set(temp) - set(indexes_to_remove))

        x_list.append(np.delete(train_data_x, indexes_to_remove, 0))
        y_list.append(np.delete(train_data_y, indexes_to_remove, 0))
    
        validate_x_list.append(np.delete(train_data_x, indexes_to_remove_validate, 0))
        validate_y_list.append(np.delete(train_data_y, indexes_to_remove_validate, 0))
    
    return np.array(x_list),np.array(y_list),np.array(validate_x_list),np.array(validate_y_list)


# In[39]:


def data_processing(path):
    file = open(path,"r")
    
    flag = True
    
    x = []
    y = []    
    
    counter = 0
    while flag:
        line = file.readline()
        if not line:
            print("End of file")
            flag = False
        else:
            counter = counter + 1
            features = np.zeros(8)
            i = 0
            for data in line.split():
                if i == 0:
                    y.append((float)(data))
                else:
                    split_str = (str)(i) + ':'
                    temp = data.split(split_str)
                    
                    if len(temp) == 1:
                        # this is for missing data we just need to skip this step by doing i = i + 1 and 0 will be assigned
                        i = i + 1
                    else:
                        features[i-1] = (float)(temp[1])
                i = i + 1
            x.append(features)
    file.close()
    #print("total lines in file is ",counter)
    return np.array(x),np.array(y)


# In[ ]:




