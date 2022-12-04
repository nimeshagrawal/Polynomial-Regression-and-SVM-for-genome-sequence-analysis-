#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import cvxpy as cp
data = pd.read_csv("g4_data.csv")


# In[159]:


def regression(data):
    sns.scatterplot(x = data['x'][:], y = data['y'][:]).set(title = 'Data_Visualisation')
    X_train, X_test, y_train, y_test = train_test_split(data['x'][:].to_numpy(), data['y'][:].to_numpy(), test_size=0.3,random_state = 1)
    train_error = []
    test_error = []
    train_fit = []
    test_fit = []
    for degree in range(11):
        
        #design matrix using polynomial basis function
        phi = np.zeros((X_train.shape[0], degree + 1))
        for i in range(X_train.shape[0]):
            x_new= []
            for j in range(degree + 1):
                x_new.append(np.power(X_train[i],j))
            phi[i] = x_new
            

        #optimal weights
        phi_inv = np.linalg.pinv(phi)
        #print(phi_inv.shape)

        w_ML = np.matmul(phi_inv,y_train)

        #train prediction
        y_predicted = []
        for i in range(X_train.shape[0]):
            y_pred = 0
            for j in range(w_ML.shape[0]):
                y_pred = y_pred + w_ML[j]*np.power(X_train[i],j)
            y_predicted.append(y_pred)
            
       
       
        #test prediction
        y_test_pred = []
        for i in range(X_test.shape[0]):
            y_pred = 0
            for j in range(w_ML.shape[0]):
                y_pred = y_pred + w_ML[j]*np.power(X_test[i],j)
            y_test_pred.append(y_pred)
        #Plotting for degree with best R2_score
        if degree == 3:
            fig, axes = plt.subplots(2, 1,figsize=(6, 8))
            opt_w = w_ML
            sns.scatterplot(ax = axes[0],x = X_train, y = y_train)
            sns.lineplot(ax = axes[0],x = X_train, y = y_predicted).set(title = 'Train fit')
            plt.savefig('trainfit')
            sns.scatterplot(ax = axes[1],x = X_test, y = y_test)
            sns.lineplot(ax = axes[1],x = X_test, y = y_test_pred).set(title = 'Test fit')
        #performance metric
        mse_train = mean_squared_error(y_train, y_predicted)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_predicted)
        r2_test = r2_score(y_test, y_test_pred)
        train_fit.append(r2_train)
        test_fit.append(r2_test)


        train_error.append(mse_train)
        test_error.append(mse_test)

    #Goodness of fit measure
    df1 = pd.DataFrame(list(zip(train_fit, test_fit)),columns =['Trainfit', 'Testfit'])
    df1.index.name = 'Degree'
    print(df1)
    df2 = pd.DataFrame(list(zip(train_error, test_error)),columns =['Trainerror', 'Testerror'])
    df2.index.name = 'Degree'
    print(df2)

    #Variance Calculation
    data_arr_1 = data.to_numpy()
    print(data_arr_1.shape)
    opt_degree = 3
    phi = np.zeros((data_arr_1.shape[0], opt_degree + 1))
    for i in range(data_arr_1.shape[0]):
        x_new= []
        for j in range(opt_degree + 1):
            x_new.append(np.power(data_arr_1[i][0],j))
        phi[i] = x_new
    print(phi.shape)

    var = 0
    for i in range(data_arr_1.shape[0]):
        var = var + np.power((data_arr_1[i][1] - np.dot(opt_w,phi[i])),2)
    var = var/(data_arr_1.shape[0])
    print("Variance", var)
    print("Weights:",opt_w)

    fig, ax2 = plt.subplots()
    plt.yscale("log")
    ax2.plot(np.arange(0,11), train_error,label = 'Train_error')
    ax2.plot(np.arange(0,11), test_error,label = 'Test_error')
    plt.legend()
   


# In[160]:


regression(data)


# In[163]:


#With L2_norm(Ridge) regulariser
def regression(data):
    lmbda = [0, 0.01, 0.1, 1, 10, 100]
    #sns.scatterplot(x = data_new['x'][:], y = data_new['y'][:])
    X_train, X_test, y_train, y_test = train_test_split(data['x'][:].to_numpy(), data['y'][:].to_numpy(), random_state = 1)
    dic = {}
    dic1 = {}
    dic2 = {}
    for l in lmbda:
        train_error = []
        test_error = []
        train_fit = []
        test_fit = []
        for degree in range(16):
            #design matrix using polynomial basis function
            phi = np.zeros((X_train.shape[0], degree + 1))
            for i in range(X_train.shape[0]):
                x_new= []
                for j in range(degree + 1):
                    x_new.append(np.power(X_train[i],j))
                phi[i] = x_new

            #optimal weights
            phi_inv = np.matmul((np.linalg.inv(np.matmul(phi.T, phi) + l*np.eye(degree + 1))), phi.T)
            #print(phi_inv.shape)

            w_ML = np.matmul(phi_inv,y_train)

            #train prediction
            y_predicted = []
            for i in range(X_train.shape[0]):
                y_pred = 0
                for j in range(w_ML.shape[0]):
                    y_pred = y_pred + w_ML[j]*np.power(X_train[i],j)
                y_predicted.append(y_pred)
            if degree == 10:
                fig, ax1 = plt.subplots()
                opt_w = w_ML
                sns.scatterplot(ax = ax1,x = X_train, y = y_train)
                sns.lineplot(ax = ax1,x = X_train, y = y_predicted)
                plt.title("Lambda = {}".format(l))
        
            #test prediction
            y_test_pred = []
            for i in range(X_test.shape[0]):
                y_pred = 0
                for j in range(w_ML.shape[0]):
                    y_pred = y_pred + w_ML[j]*np.power(X_test[i],j)
                y_test_pred.append(y_pred)

            #performance metric
            mse_train = mean_squared_error(y_train, y_predicted)
            mse_test = mean_squared_error(y_test, y_test_pred)
            r2_train = r2_score(y_train, y_predicted)
            r2_test = r2_score(y_test, y_test_pred)
            train_fit.append(r2_train)
            test_fit.append(r2_test)


            train_error.append(mse_train)
            test_error.append(mse_test)

        #Goodness of fit measure
        df1 = pd.DataFrame(list(zip(train_fit, test_fit)),columns =['Trainfit', 'Testfit'])
        df1.rename_axis("Degree", axis='columns')
        df2 = pd.DataFrame(list(zip(train_error, test_error)),columns =['Trainerror', 'Testerror'])
    

        #Variance Calculation
        data_arr_1 = data.to_numpy()
        opt_degree = 10
        phi = np.zeros((data_arr_1.shape[0], opt_degree + 1))
        for i in range(data_arr_1.shape[0]):
            x_new= []
            for j in range(opt_degree + 1):
                x_new.append(np.power(data_arr_1[i][0],j))
            phi[i] = x_new
        var = 0
        for i in range(data_arr_1.shape[0]):
            var = var + np.power((data_arr_1[i][1] - np.dot(opt_w,phi[i])),2)
        var = var/(data_arr_1.shape[0])
        print("Lambda = {}".format(l))
        print("Variance", var)
        print("Weights:", opt_w)
        fig, ax2 = plt.subplots()
        plt.yscale("log")
        ax2.plot(np.arange(0,16), train_error,label = 'Train_error')
        ax2.plot(np.arange(0,16), test_error,label = 'Test_error')
        plt.title("Lambda = {}".format(l))
        plt.legend()
        dic[l] = test_fit
        dic1[l] = train_error
        dic2[l] = test_error
    new = pd.DataFrame.from_dict(dic)
    new1 = pd.DataFrame.from_dict(dic1)
    new2 = pd.DataFrame.from_dict(dic2)
    print("R2 score test fit for different lambda:")
    print(new)
    print("Train error for different lambda:")
    print(new1)
    print("Test error for different lambda:")
    print(new2)
    new1 = np.array(new1)
    new2 = np.array(new2)
    fig, ax3 = plt.subplots()
    #plt.yscale("log")
    ax3.plot(np.arange(0,6), new1[8],label = 'Train_error')
    ax3.plot(np.arange(0,6), new2[8],label = 'Test_error')
    plt.xlabel("Lambda Index in Range [0, 0.01, 0.1, 1, 10, 100]")
    plt.ylabel("error")
    plt.legend()


# In[164]:


regression(data)


# In[52]:


def regression_try(data):
    #sns.scatterplot(x = data['x'][:], y = data['y'][:])
    X_train, X_test, y_train, y_test = train_test_split(data['x'][:].to_numpy(), data['y'][:].to_numpy(), random_state = 1)
    print(X_train.shape)

    lbd = [0, 0.001, 0.01, 0.1, 1.0, 10]
    
    temp_test = []
    temp_train = []
    for l in lbd:
        train_error = []
        test_error = []
        train_fit = []
        test_fit = []
        for degree in range(1,11):

            #design matrix using polynomial basis function
            phi_train = np.zeros((X_train.shape[0], degree + 1))
            for i in range(X_train.shape[0]):
                x_new= []
                for j in range(degree + 1):
                    x_new.append(np.power(X_train[i],j))
                phi_train[i] = x_new
            #print(phi_train.shape)

            #design matrix using polynomial basis function
            phi_test = np.zeros((X_test.shape[0], degree + 1))
            for i in range(X_test.shape[0]):
                x_new= []
                for j in range(degree + 1):
                    x_new.append(np.power(X_test[i],j))
                phi_test[i] = x_new

                error_train = []
                error_test = []
            #for i in range():
            lembda = 0.1   
            w_ML = REG_OPT(phi_train,y_train, l,restrictions_on_weight=False)


             #train prediction
            y_predicted = []
            for i in range(X_train.shape[0]):
                y_pred = 0
                for j in range(w_ML.shape[0]):
                    y_pred = y_pred + w_ML[j]*np.power(X_train[i],j)
                y_predicted.append(y_pred)

            #Plotting for degree with best R2_score
            if degree == 7:
                fig, ax1 = plt.subplots()
                opt_w = w_ML
                sns.scatterplot(ax = ax1,x = X_train, y = y_train)
                sns.lineplot(ax = ax1,x = X_train, y = y_predicted)
            #test prediction
            y_test_pred = []
            for i in range(X_test.shape[0]):
                y_pred = 0
                for j in range(w_ML.shape[0]):
                    y_pred = y_pred + w_ML[j]*np.power(X_test[i],j)
                y_test_pred.append(y_pred)

            #performance metric
            mse_train = mean_squared_error(y_train, y_predicted)
            mse_test = mean_squared_error(y_test, y_test_pred)
            r2_train = r2_score(y_train, y_predicted)
            r2_test = r2_score(y_test, y_test_pred)
            train_fit.append(r2_train)
            test_fit.append(r2_test)
            if degree == 7: 
                temp_test.append(mse_test)
                temp_train.append(mse_train)


            train_error.append(mse_train)
            test_error.append(mse_test)

        #Goodness of fit measure
        df1 = pd.DataFrame(list(zip(train_fit, test_fit)),columns =['Trainfit', 'Testfit'])
        df1.rename_axis("Degree", axis='columns')
        print(df1)
        df2 = pd.DataFrame(list(zip(train_error, test_error)),columns =['Trainerror', 'Testerror'])
        print(df2)


        #Variance Calculation
        data_arr_1 = data.to_numpy()
        print(data_arr_1.shape)
        opt_degree = 7
        phi = np.zeros((data_arr_1.shape[0], opt_degree + 1))
        for i in range(data_arr_1.shape[0]):
            x_new= []
            for j in range(opt_degree + 1):
                x_new.append(np.power(data_arr_1[i][0],j))
            phi[i] = x_new
        print(phi.shape)

        var = 0
        for i in range(data_arr_1.shape[0]):
            var = var + np.power((data_arr_1[i][1] - np.dot(opt_w,phi[i])),2)
        var = var/(data_arr_1.shape[0])
        print("Variance", var)
        print("Weights:",opt_w)

        fig, ax2 = plt.subplots()
        plt.yscale("log")
        ax2.plot(np.arange(1,11), train_error,label = 'Train_error')
        ax2.plot(np.arange(1,11), test_error,label = 'Test_error')
        plt.legend()
        plt.show()
        
    fig, ax3 = plt.subplots()
    #plt.yscale("log")
    ax3.plot(np.arange(0,len(lbd)), temp_train,label = 'Train_error')
    ax3.plot(np.arange(0,len(lbd)), temp_test,label = 'Test_error')
    plt.legend()
    plt.show()
   
    print("Train", temp_train)
    print("Test", temp_test)


# In[ ]:


def inner_prod_cp(w,x):
    return cp.sum(cp.multiply(w,x))
def REG_OPT(X,Y,lembda, restrictions_on_weight=False):

  d=X.shape[1:]  
  #print("d",np.array(d)[0])  
  w = cp.Variable(d)
  #obj=cp.Minimize((1/2)*cp.norm2(w) + cp.sum(qi)*C)
  tot_sum = 0  
  for i in range (d[0]):
    tot_sum = tot_sum + cp.abs(Y[i] - inner_prod_cp(w,X[i]))
  
  obj=cp.Minimize(tot_sum + lembda*cp.norm1(w))  
  #  
  constraints=[]
  #for i in range(M):
  #  constraints+=[cp.multiply(Y[i],(inner_prod_cp(w,X[i])))+qi[i]>=1,qi[i]>=0]
  max_X=np.max(X,axis=0)
  min_X=np.min(X,axis=0)
  if restrictions_on_weight:
    for i in range(np.array(d)[0]):
      constraints+=[w[i]<=max_X[i]]
      constraints+=[w[i]>=min_X[i]]
  problem = cp.Problem(obj)
  problem.solve(solver=cp.GLPK)
  #print(w.value)  
  return w.value


# In[ ]:


regression_try(data[0:20])

