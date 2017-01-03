from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import tree
import csv
import io
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import copy
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

rf = RandomForestClassifier(n_estimators=100)
dt = tree.DecisionTreeClassifier()
gnb = GaussianNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
svm_cf = svm.SVC()
maxlist = list()
data = list()
nusvm = svm.NuSVC()
adaboost = AdaBoostClassifier(n_estimators =100)


def CVAndOutPutAccuracies(trainData, classData, fold_no):
    
    print "Length of labelled array : "+str(len(trainData))
    print "Length of labels array : "+str(len(classData))
    print "Feature length : "+str(len(trainData[0]))
    
    print "Max Accuracy, Mean Accuracy, Min Accuracy"
    rf_accuracy = cross_val_score(rf,trainData, classData, 'accuracy',fold_no)
    rf_f1_score = cross_val_score(rf,trainData, classData, 'f1_weighted', fold_no)
    print "Random Forest Accuracy scores :"
    print ""+str(rf_accuracy.max()*100.00)+","+str(rf_accuracy.mean()*100.00)+","+str(rf_accuracy.min()*100.0)
    print "Random Forest F1 scores :"
    print ""+str(rf_f1_score.max()*100.00)+","+str(rf_f1_score.mean()*100.00)+","+str(rf_f1_score.min()*100.0)
    
    dt_accuracy = cross_val_score(dt, trainData, classData, 'accuracy', fold_no)
    dt_f1_score = cross_val_score(dt, trainData, classData, 'f1_weighted', fold_no)
    print "Decision Tree Accuracies :"
    print ""+str(dt_accuracy.max()*100.00)+","+str(dt_accuracy.mean()*100.00)+","+str(dt_accuracy.min()*100.0)
    print "Decision Tree F1 scores :"
    print ""+str(dt_f1_score.max()*100.00)+","+str(dt_f1_score.mean()*100.00)+","+str(dt_f1_score.min()*100.0)
    
    mnb_accuracy = cross_val_score(mnb, trainData,classData, 'accuracy', fold_no)
    mnb_f1_score = cross_val_score(mnb, trainData,classData, 'f1_weighted', fold_no)
    print "Multinomial Tree Accuracies :"
    print ""+str(mnb_accuracy.max()*100.00)+","+str(mnb_accuracy.mean()*100.00)+","+str(mnb_accuracy.min()*100.0)
    print "Multinomial Tree F1 scores :"
    print ""+str(mnb_f1_score.max()*100.00)+","+str(mnb_f1_score.mean()*100.00)+","+str(mnb_f1_score.min()*100.0)
    
    svc_accuracy = cross_val_score(svm_cf, trainData,classData, 'accuracy', fold_no)
    svc_f1_score = cross_val_score(svm_cf, trainData,classData, 'f1_weighted', fold_no)
    print "SVM Tree Accuracies :"
    print ""+str(svc_accuracy.max()*100.00)+","+str(svc_accuracy.mean()*100.00)+","+str(svc_accuracy.min()*100.0)
    print "SVM Tree F1 scores :"
    print ""+str(svc_f1_score.max()*100.00)+","+str(svc_f1_score.mean()*100.00)+","+str(svc_f1_score.min()*100.0)

    nusvc_accuracy = cross_val_score(nusvm, trainData, classData, 'accuracy', fold_no)
    nusvc_f1_score = cross_val_score(nusvm, trainData,classData, 'f1_weighted', fold_no)
    print "Non Linear SVM Tree Accuracies :"
    print ""+str(nusvc_accuracy.max()*100.00)+","+str(nusvc_accuracy.mean()*100.00)+","+str(nusvc_accuracy.min()*100.0)
    print "Non Linear SVM Tree F1 scores :"
    print ""+str(nusvc_f1_score.max()*100.00)+","+str(nusvc_f1_score.mean()*100.00)+","+str(nusvc_f1_score.min()*100.0)
    
    adaboost_accuracy = cross_val_score(adaboost, trainData,classData, 'accuracy', fold_no)
    adaboost_f1_score = cross_val_score(adaboost, trainData,classData, 'f1_weighted', fold_no)
    print "Adaboost Accuracies :"
    print ""+str(adaboost_accuracy.max()*100.00)+","+str(adaboost_accuracy.mean()*100.00)+","+str(adaboost_accuracy.min()*100.0)
    print "Adaboost F1 scores :"
    print ""+str(adaboost_f1_score.max()*100.00)+","+str(adaboost_f1_score.mean()*100.00)+","+str(adaboost_f1_score.min()*100.0)
    
    bagging = BaggingClassifier(mnb,max_samples=1.0, max_features=1.0)
    bagging_mnb_accuracy = cross_val_score(bagging, trainData,classData, 'accuracy', fold_no)
    bagging_mnb_f1_score = cross_val_score(bagging, trainData,classData, 'f1_weighted', fold_no)
    print "Bagging MNB Accuracies :"
    print ""+str(bagging_mnb_accuracy.max()*100.00)+","+str(bagging_mnb_accuracy.mean()*100.00)+","+str(bagging_mnb_accuracy.min()*100.0)
    print "Bagging MNB F1 scores :"
    print ""+str(bagging_mnb_f1_score.max()*100.00)+","+str(bagging_mnb_f1_score.mean()*100.00)+","+str(bagging_mnb_f1_score.min()*100.0)
    
    bagging = BaggingClassifier(dt,max_samples=1.0, max_features=1.0)
    bagging_dt_accuracy = cross_val_score(bagging, trainData,classData, 'accuracy', fold_no)
    bagging_dt_f1_score = cross_val_score(bagging, trainData,classData, 'f1_weighted', fold_no)
    print "Bagging DT Accuracies :"
    print ""+str(bagging_dt_accuracy.max()*100.00)+","+str(bagging_dt_accuracy.mean()*100.00)+","+str(bagging_dt_accuracy.min()*100.0)
    print "Bagging DT F1 scores :"
    print ""+str(bagging_dt_f1_score.max()*100.00)+","+str(bagging_dt_f1_score.mean()*100.00)+","+str(bagging_dt_f1_score.min()*100.0)
    
    bagging = BaggingClassifier(svm_cf,max_samples=1.0, max_features=1.0)
    bagging_svc_accuracy = cross_val_score(bagging, trainData,classData, 'accuracy', fold_no)
    bagging_svc_f1_score = cross_val_score(bagging, trainData,classData, 'f1_weighted', fold_no)
    print "Bagging SVC Accuracies :"
    print ""+str(bagging_svc_accuracy.max()*100.00)+","+str(bagging_svc_accuracy.mean()*100.00)+","+str(bagging_svc_accuracy.min()*100.0)
    print "Bagging SVC F1 scores :"
    print ""+str(bagging_svc_f1_score.max()*100.00)+","+str(bagging_svc_f1_score.mean()*100.00)+","+str(bagging_svc_f1_score.min()*100.0)
    
    bagging = BaggingClassifier(nusvm,max_samples=1.0, max_features=1.0)
    bagging_nu_accuracy = cross_val_score(bagging, trainData,classData, 'accuracy', fold_no)
    bagging_nu_f1_score = cross_val_score(bagging, trainData,classData, 'f1_weighted', fold_no)
    print "Bagging NU Accuracies :"
    print ""+str(bagging_nu_accuracy.max()*100.00)+","+str(bagging_nu_accuracy.mean()*100.00)+","+str(bagging_nu_accuracy.min()*100.0)
    print "Bagging NU F1 scores :"
    print ""+str(bagging_nu_f1_score.max()*100.00)+","+str(bagging_nu_f1_score.mean()*100.00)+","+str(bagging_nu_f1_score.min()*100.0)

def learnAndOutputAccuracyOnTestSet(X_test, X_train, y_train, y_test):
    rf.fit(X_train,y_train)
    print "Random Forest Test Set Accuracy :"+str(rf.score(X_test,y_test))
    
    dt.fit(X_train,y_train)
    print "Decision Tree Test Set Accuracy :"+str(dt.score(X_test,y_test))
    
    gnb.fit(X_train,y_train)
    print "Gaussian Test Set Accuracy :"+str(gnb.score(X_test,y_test))
    
    bnb.fit(X_train,y_train)
    print "B Naive Bayes Test Set Accuracy :"+str(bnb.score(X_test,y_test))
    
    mnb.fit(X_train,y_train)
    print "M Naive Bayes Test Set Accuracy :"+str(mnb.score(X_test,y_test))
    
    

rowList = list()
classes = list()
count = 0
def filterOutMatches(filename):
    global classes
    global rowList
    global maxlist
    global data
    tempRowList = list()
    tempClasses = list()
    f = io.open("cleaned_training_data.csv","r")
    filterList = io.open("matchIDs_Intl_IPL.txt", "r")
    matchlist = list()
    for line in filterList:
        matchlist.append(str(line).strip("\n").strip())
    reader = csv.reader(f)
    rowList = list()
    classes = list()
    tempdata = list()
    for row in reader :
        if(row[0] in matchlist):
            tempRowList.append(row[2:222])
            tempClasses.append(row[222])
            tempdata.append(row[2:223])
    rowList = copy.deepcopy(tempRowList)
    classes = copy.deepcopy(tempClasses)

    data = copy.deepcopy(np.array(tempdata))
    """from random import shuffle
    shuffle(data)
    r = data[0:len(data),0:len(data)-1]
    output_data = data[0:len(data),-1]
    mnb_accuracy = cross_val_score(mnb, input_data, output_data,'accuracy',10)
    print mnb_accuracy"""
    
    
    maxList = [-999]*10
    
    for row in rowList :
        index = 0
        index2 = 0
        for r in row :
            if(float(row[index])>maxList[index2]):
                maxList[index2] = float(row[index])
            index2 = (index +1)%10
            index+=1
    maxlist = copy.deepcopy(maxList)

filterOutMatches("data.csv")  


trainingset = io.open("trainingset.txt","w")
trainingset.write(unicode(rowList))

classset = io.open("classset.txt","w")
classset.write(unicode(classes))

#vary the training set size
trainData = rowList[0:100]+rowList[500:600]
labelData = classes[0:100]+classes[500:600]

normalizedTrainSet = list()
for t in trainData:
    index = 0
    i=0
    count=0
    normalizedTrainSetRow =list()
    value = 0
    while(index<len(t)):
        if(float(t[index])==-1):
            value = 0.5
        else :
            value = float(t[index])/maxlist[index%10]
        normalizedTrainSetRow.append(value)
        index+=1
    normalizedTrainSet.append(normalizedTrainSetRow)
5
#CVAndOutPutAccuracies(normalizedTrainSet, labelData, 10)
X_train, X_test, y_train, y_test = train_test_split(rowList, classes, test_size=0.3, random_state=42)
#learnAndOutputAccuracyOnTestSet(X_test, X_train, y_train, y_test)

modifiedTrainSet = list()
spamWriter = csv.writer(open('trainingSet2.csv', 'wb'))
trainingfile = io.open("features_2.csv","wb")
for t in trainData:
    index = 10
    i=0
    count=0
    modifiedTrainSetRow =list()
    while(index<=len(t)):
        sum1=0
        sum2=0
        while(i<index):
            if(i%10<=3):
                if(float(t[i])==-1):
                    sum1+=0.5
                else :
                    #print "i : "+str(i)+" t[i]: "+ str(t[i])+"maxlist[i%10] : "+str(maxlist[i%10])
                    sum1+=float(t[i])/maxlist[i%10]
                    #print " *** sum = "+str(sum1)
            elif(i%10>=6 and i%10<=7):
                if(float(t[i])==-1):
                    sum2+=0.5
                else :
                    sum2+=float(t[i])/maxlist[i%10]
                    #print "i : "+str(i)+" t[i]: "+ str(t[i])+"maxlist[i%10] : "+str(maxlist[i%10])
                    #print " *** sum2 = "+str(sum2)
            elif(i%10>7):
                if(float(t[i])==-1):
                    sum2+=0
                else :
                    #sum2-=float(t[i])/maxlist[i%10]
                    #print "i : "+str(i)+" t[i]: "+ str(t[i])+"maxlist[i%10] : "+str(maxlist[i%10])
                    #print " *** sum2 = "+str(sum2)
                    5
            i+=1
        i = index
        index+=10
        count+=1
        #print "Player "+str(count)+" : sum1 : "+str(sum1)+" sum2 :"+str(sum2)
        modifiedTrainSetRow.append(sum1)
        modifiedTrainSetRow.append(sum2)
    l=0
    sumteam1=0
    sumteam2 = 0
    while(l<len(modifiedTrainSetRow)):
        if(l<11):
            sumteam1+=modifiedTrainSetRow[l]
        else : 
            sumteam2+=modifiedTrainSetRow[l]
        l+=1
    modifiedTrainSet.append(modifiedTrainSetRow)
    trainingfile.write(unicode(str(modifiedTrainSetRow))+"\n")

for r in modifiedTrainSet:
    spamWriter.writerow(r)
    
classfile = io.open("classifications.csv","wb")
spamWriter = csv.writer(open('classifications2.csv', 'wb'))
for r in labelData :
    classfile.write(unicode(str(r)+"\n"))
    spamWriter.writerow([r])


#CVAndOutPutAccuracies(modifiedTrainSet, labelData, 10)

"""Backward Elimination of Features"""
trainData = copy.deepcopy(trainData)

featureList = ['AverageRuns', 'AverageStrikeRate', 'NoOf50s', 'NoOf100s', 'NoOfMatches', 'combinedRatingCurrentAndAverageBattingPosition', 'AvgTotalNoOfWickets', 'AvgEconomy', 'AvgNoOfWides', 'AvgNoOfNoBalls']
feature_length_per_player = len(normalizedTrainSet[0])/22
trainDataCopy = copy.deepcopy(trainData)
while(feature_length_per_player>0):
    feature_to_remove = 0
    accuracy_list = list()
    while(feature_to_remove<feature_length_per_player):
        j = 0
        for r in trainData:
            index = feature_to_remove
            i = 0
            while(index<len(r)):
                #print "feature removed from player"+str(i)+" "+str(trainData[j][index])+ " index"+ str(index)
                del trainData[j][index]
                index+=feature_length_per_player-1
                i+=1
            j+=1
        accuracies = cross_val_score(adaboost,trainData,labelData, 'accuracy',5)
        index  =0
        f = io.open("backward_elminiation_iters.txt","a")
        print "****Accuracies after removing feature : "+str(feature_to_remove)
        f.write(unicode("****Accuracies after removing feature : "+str(feature_to_remove)))
        print "FeatureLength = "+str(len(trainData[0]))+" Maximum accuracy = "+str(accuracies.max()*100.00)+" Mean Accuracy = "+str(accuracies.mean()*100.00)+" Min Accuracy = "+str(accuracies.min()*100.0)
        f.write(unicode("FeatureLength = "+str(len(trainData[0]))+" Maximum accuracy = "+str(accuracies.max()*100.00)+" Mean Accuracy = "+str(accuracies.mean()*100.00)+" Min Accuracy = "+str(accuracies.min()*100.0)))
        print "\n"
        f.write(unicode("\n"))
        f.close()
        tempList = list()
        mean_acc = accuracies.mean()*100.0
        if(mean_acc<50):
            mean_acc = 100-mean_acc
        tempList.append(mean_acc)
        tempList.append(feature_to_remove)
        accuracy_list.append(tempList)
        feature_to_remove+=1
        trainData = copy.deepcopy(trainDataCopy)
    accuracy_list.sort(key=lambda x:x[0])
    remove_index = accuracy_list[-1][1]
    j = 0
    f2 = io.open("backward_elmination_stats.txt","a")
    f2.write(unicode("feature length"+ str(len(trainData[0]))+" remove:"+str(featureList[remove_index])+"\n")+unicode(accuracy_list)+unicode(str("\n")))
    f2.close()
    del featureList[remove_index]
    print "@@@@Removing feature "+str(remove_index)+"@@@@"
    for r in trainDataCopy:
        index = remove_index
        while(index<len(r)):
            del trainDataCopy[j][index]
            index+=feature_length_per_player
        j+=1
    feature_length_per_player = len(trainDataCopy[0])/22
