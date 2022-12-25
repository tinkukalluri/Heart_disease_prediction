import gc
from django.shortcuts import render , redirect
from django.contrib import messages
from django.http import HttpResponse , JsonResponse , HttpResponseRedirect
# Create your views here.
from users.forms import UserRegistrationForm, HeartDataForm , HeartDataFormPrediction
from users.models import UserRegistrationModel, HeartDataModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from django_pandas.io import read_frame
#%matplotlib inline
from sklearn.model_selection import train_test_split
import os
import pickle
#print(os.listdir())

import warnings
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegisterAction(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            # return HttpResponseRedirect('./CustLogin')
            form = UserRegistrationForm()
            return render(request, 'Register.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'Register.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {
                    
                })
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
            # return render(request, 'user/userpage.html',{})
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

def UserHomePage(request):
    return render(request, 'users/UserHomePage.html')


def authenticate_view(request):
    if request.method=="GET":
        print("authenticate" , request.session.exists(request.session.session_key))
        if request.session.exists(request.session.session_key):
            print("suucessfull login.........................")
            return redirect('UserHomePage')
            # return HttpResponseRedirect('/userhomepage/')
        else:
            return HttpResponseRedirect('/UserLogin/')


def UserAddData(request):
    if request.method == 'POST':
        form = HeartDataForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            # return HttpResponseRedirect('./CustLogin')
            form = HeartDataForm()
            return render(request, 'users/UserAddData.html', {'form': form})
        else:
            print("Invalid form")
    else:
        form = HeartDataForm()
    return render(request, 'users/UserAddData.html', {'form': form})


def UserDataView(request):
    data_list = HeartDataModel.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(data_list, 10)
    try:
        users = paginator.page(page)
    except PageNotAnInteger:
        users = paginator.page(1)
    except EmptyPage:
        users = paginator.page(paginator.num_pages)

    return render(request, 'users/DataView_list.html', {'users': users})


def UserMachineLearning(request):
    #gc.collect()
    dataset = HeartDataModel.objects.all()
    dataset = read_frame(dataset)
    #dataset.fillna
    print(dataset.head())
    print(type(dataset))
    print(dataset.shape)
    print(dataset.head(5))
    print(dataset.sample(5))
    print(dataset.describe())
    dataset.info()
    info = ["age", "1: male, 0: female",
            "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
            "resting blood pressure", " serum cholestoral in mg/dl", "fasting blood sugar > 120 mg/dl",
            "resting electrocardiographic results (values 0,1,2)", " maximum heart rate achieved",
            "exercise induced angina", "oldpeak = ST depression induced by exercise relative to rest",
            "the slope of the peak exercise ST segment", "number of major vessels (0-3) colored by flourosopy",
            "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

    for i in range(len(info)):
        print(dataset.columns[i] + ":\t\t\t" + info[i])
    #X = dataset.drop(['target'], axis=1).values
    #print("x",X)
    dataset["target"].describe()
    print(dataset["target"].unique())
    y = dataset["target"]   
    print("y",y)
    sns.countplot(y)
    plt.show()
    print("Dataset Head",dataset.head(25))
    target_temp = dataset.target.value_counts()

    print("target Label Count=",target_temp)
    print("Percentage of patience without heart problems: " + str(round(target_temp[0] * 100 / 303, 2)))
    print("Percentage of patience with heart problems: " + str(round(target_temp[1] * 100 / 303, 2)))
    print(dataset["sex"].unique())
    print("x============================",dataset["sex"])
    print("y=============================",y)
    sns.countplot(data=dataset , x="sex" , hue="target")
    plt.show()
    # dataset["cp"].unique()
    # sns.barplot(data=dataset , x="cp" , hue="target")
    # plt.show()
    # dataset["fbs"].describe()
    # dataset["fbs"].unique()
    # sns.barplot(data=dataset , x="fbs" , hue="target")
    # plt.show()
    # dataset["restecg"].unique()
    # sns.barplot(data=dataset , x="restecg" , hue="target")
    # plt.show()
    # dataset["exang"].unique()
    # sns.barplot(data=dataset , x="exang" , hue="target")
    # plt.show()
    # dataset["slope"].unique()
    # sns.barplot(data=dataset , x="slope" , hue="target")
    # plt.show()
    # dataset["ca"].unique()
    # sns.countplot(data=dataset , x="ca" , hue="target")
    # plt.show()
    # sns.barplot(data=dataset , x="ca" , hue="target")
    # plt.show()
    # dataset["thal"].unique()
    # sns.barplot(data=dataset , x="thal" , hue="target")
    # plt.show()
    # sns.distplot(dataset["thal"])
    # plt.show()

    
    from sklearn.model_selection import train_test_split
    df_csv = pd.read_csv(r"media\heart.csv", sep=",")
    dataset=dataset.drop(['id'] , axis=1)
    print(df_csv.shape , dataset.shape)
    print(df_csv )
    print(dataset)
    fin_dataset=pd.concat([df_csv , dataset] , axis=0 , ignore_index=True)
    print('===========fin_data=========================')
    print(fin_dataset)
    # df_csv_x=df_csv_x.drop(["target"])
    # df_csv_y=df_csv_x['target']
    # df_db_x=dataset.drop(["target" , 'id'], axis=1)
    # df_db_y= dataset["target"]
    predictors = fin_dataset.drop(['target'] , axis=1)
    target = fin_dataset["target"]
    print("=============fin-predictors=================")
    print(predictors)
    print('=============fin-target=====================')
    print(target)
    print('==================data-split=========================')
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
    print(X_train, X_test, Y_train, Y_test)
    print("==================size====================")
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    # Linear regression
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    best = -1
    for _ in range(20):
        lr.fit(X_train, Y_train)
        Y_pred_lr = lr.predict(X_test)
        Y_pred_lr.shape
        print(Y_pred_lr.shape)
        score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
        print("Accuracy: " + str(score_lr))
        if score_lr > best:
            best = score_lr
            with open(r"assets\pre_trained_models\logisticRegression.pickle", "wb") as f:
                pickle.dump(lr, f)
    print("The accuracy score achieved using Linear regression is: " + str(best) + " %")
    score_lr=best



    # Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    lr = GaussianNB()
    best = -1
    for _ in range(20):
        lr.fit(X_train, Y_train)
        Y_pred_lr = lr.predict(X_test)
        Y_pred_lr.shape
        print(Y_pred_lr.shape)
        score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
        print("Accuracy: " + str(score_lr))
        if score_lr > best:
            best = score_lr
            with open(r"assets\pre_trained_models\Naive_Bayes.pickle", "wb") as f:
                pickle.dump(lr, f)
    print("The accuracy score achieved using Linear regression is: " + str(best) + " %")
    score_nb=best



    # SVM
    from sklearn import svm

    lr = svm.SVC(kernel='linear')
    best = -1
    for _ in range(20):
        lr.fit(X_train, Y_train)
        Y_pred_lr = lr.predict(X_test)
        Y_pred_lr.shape
        print(Y_pred_lr.shape)
        score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
        print("Accuracy: " + str(score_lr))
        if score_lr > best:
            best = score_lr
            with open(r"assets\pre_trained_models\SVM.pickle", "wb") as f:
                pickle.dump(lr, f)
    print("The accuracy score achieved using Linear regression is: " + str(best) + " %")
    score_svm=best


    # K Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier

    lr = KNeighborsClassifier(n_neighbors=7)
    best = -1
    for _ in range(20):
        lr.fit(X_train, Y_train)
        Y_pred_lr = lr.predict(X_test)
        Y_pred_lr.shape
        print(Y_pred_lr.shape)
        score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
        print("Accuracy: " + str(score_lr))
        if score_lr > best:
            best = score_lr
            with open(r"assets\pre_trained_models\K_Nearest_Neighbors.pickle", "wb") as f:
                pickle.dump(lr, f)
    print("The accuracy score achieved using Linear regression is: " + str(best) + " %")
    score_knn=best



    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier

    max_accuracy = 0

    lr = DecisionTreeClassifier(random_state=0)
    best = -1
    for _ in range(20):
        lr.fit(X_train, Y_train)
        Y_pred_lr = lr.predict(X_test)
        Y_pred_lr.shape
        print(Y_pred_lr.shape)
        score_lr = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)
        print("Accuracy: " + str(score_lr))
        if score_lr > best:
            best = score_lr
            with open(r"assets\pre_trained_models\Decision_Tree.pickle", "wb") as f:
                pickle.dump(lr, f)
    print("The accuracy score achieved using Linear regression is: " + str(best) + " %")
    score_dt=best



    # Neural Network
    from keras.models import Sequential
    from keras.layers import Dense
    model = Sequential()
    model.add(Dense(11, activation='relu', input_dim=13))
    model.add(Dense(6, activation='relu', input_dim=13))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train,Y_train,epochs=300)
    model.fit(X_train, Y_train, epochs=300)
    with open(r"assets\pre_trained_models\Neural_Network.pickle", "wb") as f:
        pickle.dump(model, f)
    Y_pred_nn=model.predict(X_test)
    rounded = [round(x[0]) for x in Y_pred_nn]
    Y_pred_nn = rounded
    score_nn = round(accuracy_score(Y_pred_nn, Y_test) * 100, 2)
    print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")


    scores = [score_lr, score_nb, score_svm, score_knn, score_dt, score_nn]
    algorithms = ["LR", "Naive Bayes", "SVM", "K-Nearest Neighbors", "Decision Tree", "Neural Network"]

    for i in range(len(algorithms)):
        print("The accuracy score achieved using " + algorithms[i] + " is: " + str(scores[i]) + " %")
        sns.set(rc={'figure.figsize': (15, 8)})
        plt.xlabel("Algorithms")
        plt.ylabel("Accuracy score")

    sns.barplot(x=algorithms,y= scores)
    plt.show()
    dict = {
        "score_lr" :score_lr,
        "score_nb" :score_nb,
        "score_svm" :score_svm,
        "score_knn" :score_knn,
        "score_dt" :score_dt,
        "score_nn" :score_nn,
    }
    return render(request, 'users/Machinelearning.html', dict)


def predict_(request):
    if request.method == 'POST':
        myform = HeartDataFormPrediction(request.POST)
        print(myform.errors)
        if myform.is_valid():
            data = myform.cleaned_data
            ml_algo=data['ml_algo']
            row=[]
            for key in dict(data).keys():
                if data[key] == None or key=='ml_algo':
                    continue
                print(key , data[key])
                row.append(int(data[key]))
            print('========row=========' , row)
            r=np.array([np.array(row)])
            print('========np-row=========' , r)
            target = -1
            if int(ml_algo)==1:
                target=predict_LogisticRegression(r)
            elif int(ml_algo)==2:
                target=Naive_Bayes(r)
            elif int(ml_algo)==3:
                target=SVM(r)
            elif int(ml_algo)==4:
                target=K_Nearest_Neighbors(r)
            elif int(ml_algo)==5:
                target=Decision_Tree(r)
            elif int(ml_algo)==6:
                target=Neural_Network(r)
            return HttpResponse(target)
        else:
            return render(request, 'users/UserHomePage.html', {'form': myform})
    else:
        form = HeartDataFormPrediction()
        print("form==============" , form)
        return render(request, 'users/UserHomePage.html', {'form': myform})




def predict_LogisticRegression(r):
    # LOAD MODEL
    pickle_in = open(r"assets\pre_trained_models\logisticRegression.pickle", "rb")
    logistic = pickle.load(pickle_in)
    target=logistic.predict(r)
    print(target)
    return target[0]

def Naive_Bayes(r):
    pickle_in = open(r"assets\pre_trained_models\Naive_Bayes.pickle", "rb")
    Naive_Bayes = pickle.load(pickle_in)
    target=Naive_Bayes.predict(r)
    print(target)
    return target[0]

def SVM(r):
    pickle_in = open(r"assets\pre_trained_models\SVM.pickle", "rb")
    SVM = pickle.load(pickle_in)
    target=SVM.predict(r)
    print(target)
    return target[0]

def K_Nearest_Neighbors(r):
    pickle_in = open(r"assets\pre_trained_models\K_Nearest_Neighbors.pickle", "rb")
    K_Nearest_Neighbors = pickle.load(pickle_in)
    target=K_Nearest_Neighbors.predict(r)
    print(target)
    return target[0]

def Decision_Tree(r):
    pickle_in = open(r"assets\pre_trained_models\Decision_Tree.pickle", "rb")
    Decision_Tree = pickle.load(pickle_in)
    target=Decision_Tree.predict(r)
    print(target)
    return target[0]

def Neural_Network(r):
    pickle_in = open(r"assets\pre_trained_models\Neural_Network.pickle", "rb")
    Neural_Network = pickle.load(pickle_in)
    target=Neural_Network.predict(r)
    print(target)
    return target[0]


