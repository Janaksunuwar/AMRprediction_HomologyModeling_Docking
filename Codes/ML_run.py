#All set, Intersection set and Random set performance with comparision plot

def ML_Run():
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    np.random.seed(1)
    import sklearn
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.ensemble
    import matplotlib.pyplot as plt
    from matplotlib import rc
    %matplotlib inline
    global data
    
    #Split data into features and labels
    X = data.iloc[:,1:-1] 
    Y = data.iloc[:,-1] # last column label

    #Label size and matrix size
    All_Set_Data_size = data.groupby(antb).size()
    All_Set_Matrix_size = data.shape

    #Import classifiers
    from sklearn import model_selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_score, recall_score
    import pickle
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import LeaveOneOut 
    from sklearn.model_selection import cross_val_score

    #Create dataframes for outputs
    Training_Performance = pd.DataFrame(columns=[])
    Test_Performance = pd.DataFrame(columns=[])
    Tf_CV = pd.DataFrame(columns=[])
    Area_Under_ROC = pd.DataFrame(columns=[])
    Area_Under_Precision_Recall = pd.DataFrame(columns=[])
    Model_Predict = pd.DataFrame(columns=[])

    #Split data into 6 equal parts
    skf = StratifiedKFold(n_splits=validation_no, random_state=42)
    i = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Build and evaluate models
        models = []
        models.append(('LogR', LogisticRegression(max_iter=10000)))
        models.append(('gNB', GaussianNB()))
        models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
        models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
        models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('mNB', MultinomialNB()))
        models.append(('ABC', AdaBoostClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))
        models.append(('ETC', ExtraTreesClassifier()))
        models.append(('BC', BaggingClassifier()))

        #Training performances
        myDF1 = pd.DataFrame(columns=[])
        for name, model in models:    
            model = model.fit(X_train, Y_train)
            Y_train_pred = model.predict(X_train)
            Tr_precision = precision_score(Y_train, Y_train_pred).round(3)
            Tr_recall = recall_score(Y_train, Y_train_pred).round(3)
            Tr_f1 = f1_score (Y_train, Y_train_pred).round(3)
            myDF1 = myDF1.append({'classifier': name, f'tr_precision{i+1}': Tr_precision, f'tr_recall{i+1}': Tr_recall, f'tr_f1 {i+1}':Tr_f1}, ignore_index = True)
        Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)

        #Ten-fold cross validation
        myDF3 = pd.DataFrame(columns=[])
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=42)
            results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            mean= results.mean().round(3)
            std = results.std()
            myDF3 = myDF3.append({'classifier': name, f'ten_f_CV{i+1}':mean}, ignore_index = True)
        Tf_CV = pd.concat([Tf_CV, myDF3], axis = 1)
        
        #Test performances
        myDF2 = pd.DataFrame(columns=[])
        for name, model in models:  
            model = model.fit(X_train, Y_train)
            Y_test_pred = model.predict(X_test)
            report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
            Te_precision = precision_score(Y_test, Y_test_pred).round(3)
            Te_recall = recall_score(Y_test, Y_test_pred).round(3)
            Te_f1 = f1_score (Y_test, Y_test_pred).round(3)
            myDF2 = myDF2.append({'classifier': name, f'te_precision{i+1}': Te_precision, f'te_recall{i+1}': Te_recall, f'te_f1 {i+1}':Te_f1}, ignore_index = True)
        Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)

        #AU_ROC
        myDF4 = pd.DataFrame(columns=[])
        for name, model in models:
            model = model.fit(X_train, Y_train)
            y_pred_proba = model.predict_proba(X_test)[::,1]
            fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba, pos_label = None)
            a_u_c = roc_auc_score(Y_test, y_pred_proba).round(3)
            myDF4 = myDF4.append({'a classifier': name, f'au ROC {i+1}': a_u_c}, ignore_index = True)
        Area_Under_ROC = pd.concat([Area_Under_ROC, myDF4], axis = 1)

        #AUPR
        myDF5 = pd.DataFrame(columns=[])
        for name, model in models:
            #predict probabilities
            y_pred_proba = model.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            y_pred_proba = y_pred_proba[:, 1]
            #predict class vlaues
            y_pred = model.predict(X_test)
            # calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba)
            # calculate au precision-recall curve
            area = auc(recall, precision).round(3)
            # calculate f1 score
            f1 = f1_score(Y_test, y_pred).round(3)
            myDF5 = myDF5.append({'a classifier': name, f'au PR {i+1}': area}, ignore_index = True)
        Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF5], axis = 1)
        i += 1

    #Model names
    Models = Tf_CV.iloc[:, 0] 
    
    #Calculating the mean of all folds
    #training f1 average
    tr_f1_avg = Training_Performance[Training_Performance.columns[1::4]].mean(axis=1).round(3).rename('tr_f1_avg_as', inplace=True)
    tr_f1_stdev = Training_Performance[Training_Performance.columns[1::4]].std(axis=1).round(3).rename('tr_f1_stdev_as', inplace=True)

    #Training precision average
    tr_precision_avg = Training_Performance[Training_Performance.columns[2::4]].mean(axis=1).round(3).rename('tr_precision_avg_as', inplace=True)
    tr_precision_stdev = Training_Performance[Training_Performance.columns[2::4]].std(axis=1).round(3).rename('tr_precision_stdev_as', inplace=True)
    
    #Training recall average
    tr_recall_avg = Training_Performance[Training_Performance.columns[3::4]].mean(axis=1).round(3).rename('tr_recall_avg_as', inplace=True)
    tr_recall_stdev = Training_Performance[Training_Performance.columns[3::4]].std(axis=1).round(3).rename('tr_recall_stdev_as', inplace=True)
 
    #Test f1 average
    te_f1_avg = Test_Performance[Test_Performance.columns[1::4]].mean(axis=1).round(3).rename('te_f1_avg_as', inplace=True)
    te_f1_stdev = Test_Performance[Test_Performance.columns[1::4]].std(axis=1).round(3).rename('te_f1_stdev_as', inplace=True)
    
    #Test precision average
    te_precision_avg = Test_Performance[Test_Performance.columns[2::4]].mean(axis=1).round(3).rename('te_precision_avg_as', inplace=True)
    te_precision_stdev = Test_Performance[Test_Performance.columns[2::4]].std(axis=1).round(3).rename('te_precision_stdev_as', inplace=True)

    #Test recall average
    te_recall_avg = Test_Performance[Test_Performance.columns[3::4]].mean(axis=1).round(3).rename('te_recall_avg_as', inplace=True)
    te_recall_stdev = Test_Performance[Test_Performance.columns[3::4]].std(axis=1).round(3).rename('te_recall_stdev_as', inplace=True)
    
    #Tf_CV average
    Tf_CV_Avg = Tf_CV[Tf_CV.columns[1::2]].mean(axis=1).round(3).rename('Tf_CV_Avg_as', inplace=True)
    Tf_CV_stdev = Tf_CV[Tf_CV.columns[1::2]].std(axis=1).round(3).rename('Tf_CV_stdev_as', inplace=True)
    
    #Area_Under_ROC average
    au_ROC_avg = Area_Under_ROC[Area_Under_ROC.columns[1::2]].mean(axis=1).round(3).rename('au_ROC_avg_as', inplace=True)
    au_ROC_stdev = Area_Under_ROC[Area_Under_ROC.columns[1::2]].std(axis=1).round(3).rename('au_ROC_stdev_as', inplace=True)
    
    #Area_Under_Precision_Recall average
    au_PR_avg= Area_Under_Precision_Recall[Area_Under_Precision_Recall.columns[1::2]].mean(axis=1).round(3).rename('au_PR_avg_as', inplace=True)
    au_PR_stdev= Area_Under_Precision_Recall[Area_Under_Precision_Recall.columns[1::2]].std(axis=1).round(3).rename('au_PR_stdev_as', inplace=True)

    #Accumulating all dataframes
    frames2 = [Models, 
               tr_precision_avg, tr_precision_stdev, tr_recall_avg, tr_recall_stdev, tr_f1_avg, tr_f1_stdev, 
               te_precision_avg, te_precision_stdev, te_recall_avg, te_recall_stdev, te_f1_avg, te_f1_stdev,
               Tf_CV_Avg, Tf_CV_stdev, au_ROC_avg, au_ROC_stdev, au_PR_avg, au_PR_stdev]
    
    #Result all set
    Final_All_set_Results= pd.concat(frames2, axis=1)
    
    #Leave one out cross validation
    #List for output
    Loo = []
    
    #Leave one out validation
    cv = LeaveOneOut()
    from numpy import mean
    from numpy import std
    #Evaluate each model for Loo CV
    for name, model in models:
        # fit model
        scores = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=-1)
        Loo_avg = mean(scores).round(3)
        Loo_stdev = std(scores).round(3)
        Loo.append({'Loo_CV_as': Loo_avg, 'Loo_stdev_as': Loo_stdev})
    
    Loo_CV = pd.DataFrame(Loo)
    Final_All_set_Results = pd.concat([Final_All_set_Results, Loo_CV], axis=1)
    
    print(f'All Set Results {antb} {bacteria}')
    display(Final_All_set_Results)
    
    #Result out
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} All set Results\n')
        rf.write(f'\n{All_Set_Data_size}\n')
        rf.write(f'\nmatrix_size: {All_Set_Matrix_size}\n\n')
    
        Final_All_set_Results.to_csv(rf)
    
    #Export results separately
    Final_All_set_Results.to_csv(f'{bacteria}_{antb}_All_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #Selecting important features in each fold from tree based-classifiers
    clfk = ExtraTreesClassifier(random_state=1)
    
    #Dataframes for output
    feat_Df = pd.DataFrame(columns=[])
    scores = []
    test_scores = []
    check_feat = []
    Output = pd.DataFrame()
    
    #Split the data
    skf = StratifiedKFold(n_splits=validation_no, random_state=42)
    j = 0
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
        #Fit the model
        modelk = clfk.fit(X_train,Y_train)
        predictions = clfk.predict(X_test)
        scores.append(clfk.score(X_test, Y_test))
        feat = clfk.feature_importances_
    
        #Select the column header from first to the second last
        colhead = list(np.array([data.columns[1:-1]]).T)
    
        #Zip two columns into a dataframe
        list_of_tuplesk= list(zip(colhead, feat))
    
        #Create features dataframe
        feature_importancek = pd.DataFrame(list_of_tuplesk, columns = [f'Feature fold{j}', f'Importance fold{j}'])
    
        #Sort the dataframe, descending
        feature_importance_sortedk = feature_importancek.sort_values(by=f'Importance fold{j}', ascending=False)
    
        #Remove the square brackets from the dataframe
        feature_importance_sortedk [f'Feature fold{j}'] = feature_importance_sortedk[f'Feature fold{j}'].str.get(0)
        feature_importance_sortedk = feature_importance_sortedk.round(3)
    
        #Sort the features
        feat_sort_df = pd.DataFrame(feature_importance_sortedk)
        feat_sort_df.reset_index(drop=True, inplace=True)
        feat_Df.reset_index(drop=True, inplace=True)
        feat_Df = pd.concat([feat_Df, feat_sort_df], axis= 1)
        j += 1
    
    #Select the top genes out from range
    top_genes_range = 8000
    
    #Make dataframe of selected top dataframes
    Top_consistent = feat_Df.iloc[0:top_genes_range, :]
    
    #Separate each column to separate dataframe and find common in all
    cdf1 = Top_consistent[['Feature fold0']].rename(columns={"Feature fold0": "Feature"})
    cdf2 = Top_consistent[['Feature fold1']].rename(columns={"Feature fold1": "Feature"})
    cdf3 = Top_consistent[['Feature fold2']].rename(columns={"Feature fold2": "Feature"})
    cdf4 = Top_consistent[['Feature fold3']].rename(columns={"Feature fold3": "Feature"})
    cdf5 = Top_consistent[['Feature fold4']].rename(columns={"Feature fold4": "Feature"})
    cdf6 = Top_consistent[['Feature fold5']].rename(columns={"Feature fold5": "Feature"})
    
    #export separate raw consistent genes without merge
    feat_Df.to_csv(f'{bacteria}_{antb}_Raw_Consistent_Genes_Per_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #Merging common in all folds
    merge12 = pd.merge(cdf1, cdf2, how='inner', on=['Feature'])
    merge123 = pd.merge(merge12, cdf3, how='inner', on=['Feature'])
    merge1234 = pd.merge(merge123, cdf4, how='inner', on=['Feature'])
    merge12345 = pd.merge(merge1234, cdf5, how='inner', on=['Feature'])
    Consistent_Genes_per_fold = pd.merge(merge12345, cdf6, how='inner', on=['Feature'])
    
    Final_Consistent_Genes_per_fold = Consistent_Genes_per_fold.iloc[:100, :]
    
    #Create a result file
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Consistent Genes per {validation_no} fold validation All Set\n')
        Final_Consistent_Genes_per_fold.to_csv(rf)
    
    #Export consistent genes as separate file
    Final_Consistent_Genes_per_fold.to_csv(f'{bacteria}_{antb}_Consistent_Genes_Per_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #INTERSECTION SET RUN
    #Read gene_ast matrix
    open_gene_ast = pd.read_csv(file_name)
    
    #Open consistent genes based per validation
    open_consistent_genes = pd.read_csv(f'{bacteria}_{antb}_Consistent_Genes_Per_{validation_no}-fold_CV.csv')
    
    #Make separate dataframe with just consistent genes
    target_genesTT = open_consistent_genes[['Feature']].rename(columns={'Feature': 'Consistent genes'})
    
    #No of top consistent genes
    num = 100
    target_genesTT = target_genesTT.iloc[:num, :]
    
    #Sort the consistent genes
    target_genesTT = target_genesTT.sort_values('Consistent genes')
    
    #Adding antibiotic lable at the end
    target_genes_good = target_genesTT.append({'Consistent genes': f'{antb}'}, ignore_index=True)
    
    #Converting consistent genes to a list
    column_list = target_genes_good['Consistent genes'].tolist()
    
    #Adding phenotype lable at the end
    target_genes_good1 = target_genesTT.append({'Consistent genes': 'phenotype'}, ignore_index=True)
    
    #Converting consistent genes with phenotype to a list
    column_list1 = target_genes_good1['Consistent genes'].tolist()
    
    #Make data consisting only with consistent genes 
    data_I = open_gene_ast[column_list]
    
    #Label size and matrix size
    Intersection_Data_size = data_I.groupby(antb).size()
    Intersection_Matrix_size = data_I.shape
    
    #Split the data to features and labels
    X = data_I.iloc[:, 0:-1]
    Y = data_I.iloc[:,-1]
    
    #Create dataframes for outputs
    Training_Performance = pd.DataFrame(columns=[])
    Test_Performance = pd.DataFrame(columns=[])
    Tf_CV = pd.DataFrame(columns=[])
    Area_Under_ROC = pd.DataFrame(columns=[])
    Area_Under_Precision_Recall = pd.DataFrame(columns=[])
    Model_Predict = pd.DataFrame(columns=[])
    
    #Split data into 6 equal parts
    skf = StratifiedKFold(n_splits=validation_no, random_state=42)
    i = 0
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
        #Build and evaluate models
        models = []
        models.append(('LogR', LogisticRegression(max_iter=10000)))
        models.append(('gNB', GaussianNB()))
        models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
        models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
        models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('mNB', MultinomialNB()))
        models.append(('ABC', AdaBoostClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))
        models.append(('ETC', ExtraTreesClassifier()))
        models.append(('BC', BaggingClassifier()))
    
        #Training performances
        myDF1 = pd.DataFrame(columns=[])
        for name, model in models:    
            model = model.fit(X_train, Y_train)
            Y_train_pred = model.predict(X_train)
            Tr_precision = precision_score(Y_train, Y_train_pred).round(3)
            Tr_recall = recall_score(Y_train, Y_train_pred).round(3)
            Tr_f1 = f1_score (Y_train, Y_train_pred).round(3)
            myDF1 = myDF1.append({'classifier': name, f'tr_precision{i+1}': Tr_precision, f'tr_recall{i+1}': Tr_recall, f'tr_f1 {i+1}':Tr_f1}, ignore_index = True)
        Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)
    
        #Test performances
        myDF2 = pd.DataFrame(columns=[])
        for name, model in models:  
            model = model.fit(X_train, Y_train)
            Y_test_pred = model.predict(X_test)
            report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
            Te_precision = precision_score(Y_test, Y_test_pred).round(3)
            Te_recall = recall_score(Y_test, Y_test_pred).round(3)
            Te_f1 = f1_score (Y_test, Y_test_pred).round(3)
            myDF2 = myDF2.append({'classifier': name, f'te_precision{i+1}': Te_precision, f'te_recall{i+1}': Te_recall, f'te_f1 {i+1}':Te_f1}, ignore_index = True)
        Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)
    
        #Ten-fold cross validation
        myDF3 = pd.DataFrame(columns=[])
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=42)
            results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            mean = results.mean().round(3)
            std = results.std()
            myDF3 = myDF3.append({'classifier': name, f'ten_f_CV{i+1}':mean}, ignore_index = True)
        Tf_CV = pd.concat([Tf_CV, myDF3], axis = 1)
    
        #AU_ROC
        myDF4 = pd.DataFrame(columns=[])
        for name, model in models:
            model = model.fit(X_train, Y_train)
            y_pred_proba = model.predict_proba(X_test)[::,1]
            fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba, pos_label = None)
            a_u_c = roc_auc_score(Y_test, y_pred_proba).round(3)
            myDF4 = myDF4.append({'a classifier': name, f'au ROC {i+1}': a_u_c}, ignore_index = True)
        Area_Under_ROC = pd.concat([Area_Under_ROC, myDF4], axis = 1)
    
        #AUPR
        myDF5 = pd.DataFrame(columns=[])
        for name, model in models:
            #predict probabilities
            y_pred_proba = model.predict_proba(X_test)
            # keep probabilities for the positive outcome only
            y_pred_proba = y_pred_proba[:, 1]
            #predict class vlaues
            y_pred = model.predict(X_test)
            # calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba)
            # calculate au precision-recall curve
            area = auc(recall, precision).round(3)
            # calculate f1 score
            f1 = f1_score(Y_test, y_pred).round(3)
            myDF5 = myDF5.append({'a classifier': name, f'au PR {i+1}': area}, ignore_index = True)
        Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF5], axis = 1)
        i += 1
    
    #Model names
    Models = Tf_CV.iloc[:, 0] 
    
    #training f1 average
    tr_f1_avg = Training_Performance[Training_Performance.columns[1::4]].mean(axis=1).round(3).rename('tr_f1_avg_is', inplace=True)
    tr_f1_stdev = Training_Performance[Training_Performance.columns[1::4]].std(axis=1).round(3).rename('tr_f1_stdev_is', inplace=True)

    
    #Training precision average
    tr_precision_avg = Training_Performance[Training_Performance.columns[2::4]].mean(axis=1).round(3).rename('tr_precision_avg_is', inplace=True)
    tr_precision_stdev = Training_Performance[Training_Performance.columns[2::4]].std(axis=1).round(3).rename('tr_precision_stdev_is', inplace=True)
    
    
    #Training recall average
    tr_recall_avg = Training_Performance[Training_Performance.columns[3::4]].mean(axis=1).round(3).rename('tr_recall_avg_is', inplace=True)
    tr_recall_stdev = Training_Performance[Training_Performance.columns[3::4]].std(axis=1).round(3).rename('tr_recall_stdev_is', inplace=True)
 
    
    #Test f1 average
    te_f1_avg = Test_Performance[Test_Performance.columns[1::4]].mean(axis=1).round(3).rename('te_f1_avg_is', inplace=True)
    te_f1_stdev = Test_Performance[Test_Performance.columns[1::4]].std(axis=1).round(3).rename('te_f1_stdev_is', inplace=True)
    
    
    #Test precision average
    te_precision_avg = Test_Performance[Test_Performance.columns[2::4]].mean(axis=1).round(3).rename('te_precision_avg_is', inplace=True)
    te_precision_stdev = Test_Performance[Test_Performance.columns[2::4]].std(axis=1).round(3).rename('te_precision_stdev_is', inplace=True)

    
    #Test recall average
    te_recall_avg = Test_Performance[Test_Performance.columns[3::4]].mean(axis=1).round(3).rename('te_recall_avg_is', inplace=True)
    te_recall_stdev = Test_Performance[Test_Performance.columns[3::4]].std(axis=1).round(3).rename('te_recall_stdev_is', inplace=True)
    
    #Tf_CV average
    Tf_CV_Avg = Tf_CV[Tf_CV.columns[1::2]].mean(axis=1).round(3).rename('Tf_CV_Avg_is', inplace=True)
    Tf_CV_stdev = Tf_CV[Tf_CV.columns[1::2]].std(axis=1).round(3).rename('Tf_CV_stdev_is', inplace=True)
    
    #Area_Under_ROC average
    au_ROC_avg = Area_Under_ROC[Area_Under_ROC.columns[1::2]].mean(axis=1).round(3).rename('au_ROC_avg_is', inplace=True)
    au_ROC_stdev = Area_Under_ROC[Area_Under_ROC.columns[1::2]].std(axis=1).round(3).rename('au_ROC_stdev_is', inplace=True)
    
    #Area_Under_Precision_Recall average
    au_PR_avg= Area_Under_Precision_Recall[Area_Under_Precision_Recall.columns[1::2]].mean(axis=1).round(3).rename('au_PR_avg_is', inplace=True)
    au_PR_stdev= Area_Under_Precision_Recall[Area_Under_Precision_Recall.columns[1::2]].std(axis=1).round(3).rename('au_PR_stdev_is', inplace=True)

    #Accumulating all dataframes
    frames2 = [Models, 
               tr_precision_avg, tr_precision_stdev, tr_recall_avg, tr_recall_stdev, tr_f1_avg, tr_f1_stdev, 
               te_precision_avg, te_precision_stdev, te_recall_avg, te_recall_stdev, te_f1_avg, te_f1_stdev,
               Tf_CV_Avg, Tf_CV_stdev, au_ROC_avg, au_ROC_stdev, au_PR_avg, au_PR_stdev]
    
    #Result all set
    Intersection_set_Results= pd.concat(frames2, axis=1)
    
    #Leave one out cross validation
    #List for output
    Loo = []
    
    #Leave one out validation
    cv = LeaveOneOut()
    from numpy import mean
    from numpy import std
    
    #Accumulate models  
    models = []
    models.append(('LogR', LogisticRegression(max_iter=10000)))
    models.append(('gNB', GaussianNB()))
    models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
    models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
    models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('mNB', MultinomialNB()))
    models.append(('ABC', AdaBoostClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('ETC', ExtraTreesClassifier()))
    models.append(('BC', BaggingClassifier()))
    
    #Evaluate each model for Loo CV
    for name, model in models:
        # fit model
        scores = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=-1)
        Loo_avg = mean(scores).round(3)
        Loo_stdev = std(scores).round(3)
        Loo.append({'Loo_CV_is': Loo_avg, 'Loo_stdev_is': Loo_stdev})
    
    Loo_CV = pd.DataFrame(Loo)
   
    Final_Intersection_set_Results = pd.concat([Intersection_set_Results, Loo_CV], axis=1)
    
    print(f'Intersection Set Results {antb} {bacteria}')
    display(Final_Intersection_set_Results)
    #Results out
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Intersection Set Results\n')
    
        Final_Intersection_set_Results.to_csv(rf)

    # export result separately
    Final_Intersection_set_Results.to_csv(f'{bacteria}_{antb}_Intersection_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #RANDOM SET RUN
    #No of genes to shuffle
    num = 100
    
    #Read gene_ast matrix
    label = data[[antb]]
    daa = data.drop(['assembly_accession', antb], axis=1)
    
    #Create a dataframe for the final output of the program
    Random_Set_Results = pd.DataFrame(columns=[])
    Loo_CV = pd.DataFrame(columns=[])
    
    #Select 10 random sets
    for k in range(10):
        samp = daa.sample(n=num, replace = False, axis=1)
        data = pd.concat([samp, label], axis=1)
        Final_Randon_data_size = data.groupby(antb).size()
        X = data.iloc[:, 1:num]
        Y = data.iloc[:,-1]
    
        #Dataframes for results
        Tf_CV = pd.DataFrame(columns=[])
        Training_Performance = pd.DataFrame(columns=[])
        Test_Performance = pd.DataFrame(columns=[])
        Area_Under_ROC = pd.DataFrame(columns=[])
        Area_Under_Precision_Recall = pd.DataFrame(columns=[])
        Model_Predict = pd.DataFrame(columns=[])
        skf = StratifiedKFold(n_splits=validation_no, random_state=42)
        ij = 0
    
        #Split the data
        for train_index, test_index in skf.split(X, Y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
            #Build model and evaluate models
            models = []
            models.append(('LogR', LogisticRegression(max_iter=10000)))
            models.append(('gNB', GaussianNB()))
            models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
            models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
            models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
            models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('mNB', MultinomialNB()))
            models.append(('ABC', AdaBoostClassifier()))
            models.append(('GBC', GradientBoostingClassifier()))
            models.append(('ETC', ExtraTreesClassifier()))
            models.append(('BC', BaggingClassifier()))
    
            #Training performance
            myDF1 = pd.DataFrame(columns=[])
            for name, model in models:    
                model = model.fit(X_train, Y_train)
                Y_train_pred = model.predict(X_train)
                Tr_precision = precision_score(Y_train, Y_train_pred).round(3)
                Tr_recall = recall_score(Y_train, Y_train_pred).round(3)
                Tr_f1 = f1_score (Y_train, Y_train_pred).round(3)
                myDF1 = myDF1.append({'classifier': name, f'tr_precision{ij+1}': Tr_precision, f'tr_recall{ij+1}': Tr_recall, f'tr_f1 {ij+1}':Tr_f1}, ignore_index = True)
            Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)
    
            #Test performance
            myDF2 = pd.DataFrame(columns=[])
            for name, model in models:  
                model = model.fit(X_train, Y_train)
                Y_test_pred = model.predict(X_test)
                report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
                Te_precision = precision_score(Y_test, Y_test_pred).round(3)
                Te_recall = recall_score(Y_test, Y_test_pred).round(3)
                Te_f1 = f1_score (Y_test, Y_test_pred).round(3)
                myDF2 = myDF2.append({'classifier': name, f'te_precision{ij+1}': Te_precision, f'te_recall{ij+1}': Te_recall, f'te_f1 {ij+1}':Te_f1}, ignore_index = True)
            Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)
    
            #Ten-fold cross validation
            myDF = pd.DataFrame(columns=[])
            for name, model in models:
                kfold = model_selection.StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
                results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
                mean= results.mean().round(3)
                std = results.std()
                myDF = myDF.append({'classifier': name, f'ten_f_CV{ij+1}':mean}, ignore_index = True)
            Tf_CV = pd.concat([Tf_CV, myDF], axis = 1)
    
            #AU_ROC
            myDF3 = pd.DataFrame(columns=[])
            for name, model in models:
                model = model.fit(X_train, Y_train)
                y_pred_proba = model.predict_proba(X_test)[::,1]
                # keep probabilities for the positive outcome only
    
                fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba, pos_label = None)
                a_u_c = roc_auc_score(Y_test, y_pred_proba).round(3)
                myDF3 = myDF3.append({'classifier': name, f'au ROC {ij+1}': a_u_c}, ignore_index = True)
    
            Area_Under_ROC = pd.concat([Area_Under_ROC, myDF3], axis = 1)
    
            #AUPR
            myDF4 = pd.DataFrame(columns=[])
            for name, model in models:
                #predict probabilities
                y_pred_proba = model.predict_proba(X_test)
                # keep probabilities for the positive outcome only
                y_pred_proba = y_pred_proba[:, 1]
                #predict class vlaues
                y_pred = model.predict(X_test)
                # calculate precision-recall curve
                precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba)
                # calculate au precision-recall curve
                area = auc(recall, precision).round(3)
                # calculate f1 score
                f1 = f1_score(Y_test, y_pred, average='weighted').round(3)
                myDF4 = myDF4.append({'classifier': name, f'au PR {ij+1}': area}, ignore_index = True)
            Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF4], axis = 1)
            ij += 1
    
        #List for output
        myDF5 = pd.DataFrame(columns=[])
        Loo = []
    
        #Leave one out validation
        cv = LeaveOneOut()

        from numpy import mean
        from numpy import std
        #Evaluate each model
        for name, model in models:
            # fit model
            scores1 = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=-1)
            Loo_avg = mean(scores1).round(3)
            Loo_stdev = std(scores1).round(3)
                        
            myDF5 = myDF5.append({'classifier': name, f'Loo_CV {k+1}': Loo_avg, f'Loo_stdev {k+1}': Loo_stdev}, ignore_index = True)
        Loo_CV = pd.concat([Loo_CV, myDF5], axis=1)

        #Model names
        Models = Tf_CV.iloc[:, 0]
    
        #Training_Performance F1 average
        tr_f1_avg = Training_Performance[Training_Performance.columns[1::4]].mean(axis=1).round(3)
        tr_f1_avg = tr_f1_avg.rename(f'tr_f1_avg_{k}', inplace=True)
    
        #Training_Performance precision average
        tr_precision_avg = Training_Performance[Training_Performance.columns[2::4]].mean(axis=1).round(3)
        tr_precision_avg = tr_precision_avg.rename(f'tr_precision_avg_{k}', inplace=True)
    
        #Training_Performance recall average
        tr_recall_avg = Training_Performance[Training_Performance.columns[3::4]].mean(axis=1).round(3)
        tr_recall_avg = tr_recall_avg.rename(f'tr_recall_avg_{k}', inplace=True)
    
        #Test_Performance f1 average
        te_f1_avg = Test_Performance[Test_Performance.columns[1::4]].mean(axis=1).round(3)
        te_f1_avg = te_f1_avg.rename(f'te_f1_avg_{k}', inplace=True)
    
        #Test_Performance precision average
        te_precision_avg = Test_Performance[Test_Performance.columns[2::4]].mean(axis=1).round(3)
        te_precision_avg = te_precision_avg.rename(f'te_precision_avg_{k}', inplace=True)
    
        #Test_Performance recall average
        te_recall_avg = Test_Performance[Test_Performance.columns[3::4]].mean(axis=1).round(3)
        te_recall_avg = te_recall_avg.rename(f'te_recall_avg_{k}', inplace=True)
    
        #Ten fold crossvalidation average
        Tf_CV_Avg = Tf_CV[Tf_CV.columns[1::2]].mean(axis=1).round(3)
        Tf_CV_Avg = Tf_CV_Avg.rename(f'Tf_CV_Avg_{k}', inplace=True)
    
        #Area_Under_ROC average
        au_ROC_avg_rs = Area_Under_ROC[Area_Under_ROC.columns[0::2]].mean(axis=1).round(3)
        au_ROC_avg_rs = au_ROC_avg_rs.rename(f'au_ROC_avg_{k}', inplace=True)
    
        #Area_Under_Precision_Recall average
        au_PR_avg_rs = Area_Under_Precision_Recall[Area_Under_Precision_Recall.columns[0::2]].mean(axis=1).round(3)
        au_PR_avg_rs = au_PR_avg_rs.rename(f'au_PR_avg_{k}', inplace=True)
    
        #Concatenate results
        frames1 = [Models, tr_precision_avg, tr_recall_avg, tr_f1_avg, te_precision_avg, te_recall_avg, te_f1_avg, Tf_CV_Avg, au_ROC_avg_rs, au_PR_avg_rs]
        Ran_Resul = pd.concat(frames1, axis=1)
        Random_Set_Results = pd.concat([Random_Set_Results, Ran_Resul], axis =1)
        k +=1
        
    #Calculating average for outer 10 random sets from nested inner fold validation 
    Models = pd.DataFrame(Models)
    
    #Training_Performance precision average
    tr_pa = Random_Set_Results[Random_Set_Results.columns[1::10]].mean(axis=1).round(3).rename('tr_precision_avg_rs', inplace=True)
    tr_pa_stdev = Random_Set_Results[Random_Set_Results.columns[1::10]].std(axis=1).round(3).rename('tr_precision_stdev_rs', inplace=True)
    tr_pa = pd.DataFrame(tr_pa)
    tr_pa_stdev = pd.DataFrame(tr_pa_stdev)
    
    #Training_Performance recall average
    tr_ra = Random_Set_Results[Random_Set_Results.columns[2::10]].mean(axis=1).round(3).rename('tr_recall_avg_rs', inplace=True)
    tr_ra_stdev = Random_Set_Results[Random_Set_Results.columns[2::10]].std(axis=1).round(3).rename('tr_recall_stdev_rs', inplace=True)
    tr_ra = pd.DataFrame(tr_ra)
    tr_ra_stdev = pd.DataFrame(tr_ra_stdev)

    #Training_Performance F1 average
    tr_fa = Random_Set_Results[Random_Set_Results.columns[3::10]].mean(axis=1).round(3).rename('tr_f1_avg_rs', inplace=True)
    tr_fa_stdev = Random_Set_Results[Random_Set_Results.columns[3::10]].std(axis=1).round(3).rename('tr_f1_stdev_rs', inplace=True)
    tr_fa = pd.DataFrame(tr_fa)
    tr_fa_stdev = pd.DataFrame(tr_fa_stdev)
    
    #Test_Performance precision average
    te_pa = Random_Set_Results[Random_Set_Results.columns[4::10]].mean(axis=1).round(3).rename('te_precision_avg_rs', inplace=True)
    te_pa_stdev = Random_Set_Results[Random_Set_Results.columns[4::10]].std(axis=1).round(3).rename('te_precision_stdev_rs', inplace=True)
    te_pa = pd.DataFrame(te_pa)
    te_pa_stdev = pd.DataFrame(te_pa_stdev)
    
    #Test_Performance recall average
    te_ra = Random_Set_Results[Random_Set_Results.columns[5::10]].mean(axis=1).round(3).rename('te_recall_avg_rs', inplace=True)
    te_ra_stdev = Random_Set_Results[Random_Set_Results.columns[5::10]].std(axis=1).round(3).rename('te_recall_stdev_rs', inplace=True)
    tr_ra = pd.DataFrame(tr_ra)
    te_ra_stdev = pd.DataFrame(te_ra_stdev)

    #Test_Performance f1 average
    te_fa = Random_Set_Results[Random_Set_Results.columns[6::10]].mean(axis=1).round(3).rename('te_f1_avg_rs', inplace=True)
    te_fa_stdev = Random_Set_Results[Random_Set_Results.columns[6::10]].std(axis=1).round(3).rename('te_f1_stdev_rs', inplace=True)
    te_fa = pd.DataFrame(te_fa)
    te_fa_stdev = pd.DataFrame(te_fa_stdev)
    
    #Ten fold crossvalidation average
    Tf_Ca = Random_Set_Results[Random_Set_Results.columns[7::10]].mean(axis=1).round(3).rename('Tf_CV_Avg_rs', inplace=True)
    Tf_Ca_stdev = Random_Set_Results[Random_Set_Results.columns[7::10]].std(axis=1).round(3).rename('Tf_CV_stdev_rs', inplace=True)
    Tf_Ca = pd.DataFrame(Tf_Ca)
    Tf_Ca_stdev = pd.DataFrame(Tf_Ca_stdev)
    
    #Leave one out (Loo) crossvalidation average
    Loo_Ca = Random_Set_Results[Random_Set_Results.columns[0::3]].mean(axis=1).round(3).rename('Loo_CV_Avg_rs', inplace=True)
    Loo_Ca_stdev = Random_Set_Results[Random_Set_Results.columns[1::3]].mean(axis=1).round(3).rename('Loo_CV_stdev_rs', inplace=True)
    Loo_Ca = pd.DataFrame(Loo_Ca)
    Loo_Ca_stdev = pd.DataFrame(Loo_Ca_stdev)
    
    #Area_Under_ROC average
    au_Ra = Random_Set_Results[Random_Set_Results.columns[8::10]].mean(axis=1).round(3).rename('au_ROC_avg_rs', inplace=True)
    au_Ra_stdev = Random_Set_Results[Random_Set_Results.columns[8::10]].std(axis=1).round(3).rename('au_ROC_stdev_rs', inplace=True)
    au_Ra = pd.DataFrame(au_Ra)
    au_Ra_stdev = pd.DataFrame(au_Ra_stdev)
    
    #Area_Under_Precision_Recall average
    au_Pa = Random_Set_Results[Random_Set_Results.columns[9::10]].mean(axis=1).round(3).rename('au_PR_avg_rs', inplace=True)
    au_Pa_stdev = Random_Set_Results[Random_Set_Results.columns[9::10]].std(axis=1).round(3).rename('au_PR_stdev_rs', inplace=True)
    au_Pa = pd.DataFrame(au_Pa)
    au_Pa_stdev = pd.DataFrame(au_Pa_stdev)
    
    #Concatenate results
    Final_Random_Set_Results = pd.concat([Models, 
                                          tr_pa, tr_pa_stdev, tr_ra, tr_ra_stdev, tr_fa, tr_fa_stdev, 
                                          te_pa, te_pa_stdev, te_ra, te_ra_stdev, te_fa, te_fa_stdev,  
                                          Tf_Ca, Tf_Ca_stdev, Loo_Ca, Loo_Ca_stdev,
                                          au_Ra, au_Ra_stdev, au_Pa, au_Pa_stdev], axis=1)
    
    print(f'Random Set Results {antb} {bacteria}')
    display(Final_Random_Set_Results)
    
    #Result out
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Random set Results\n')
        Final_Random_Set_Results.to_csv(rf)
    
    #Export result separately
    Final_Random_Set_Results.to_csv(f'{bacteria}_{antb}_Random_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #Plot All set, Intersection set, Random set performance comparision figures
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc
    %matplotlib inline

    #Open files to dataframe
    d1 = pd.read_csv(f'{bacteria}_{antb}_All_Set_Performance_{validation_no}-fold_CV.csv')
    d2 = pd.read_csv(f'{bacteria}_{antb}_Intersection_Set_Performance_{validation_no}-fold_CV.csv' )
    d3 = pd.read_csv(f'{bacteria}_{antb}_Random_Set_Performance_{validation_no}-fold_CV.csv')

    #Select classifier names
    models = d1[['classifier']]

    #extract the average from All set, Intersection set, Random set performances
    def extract_average(a, b, c):
        a_s = d1[[f'{a}']]
        i_s = d2[[f'{b}']]
        r_s = d3[[f'{c}']]
        df = pd.concat([models, a_s, i_s, r_s], axis =1)
        df.set_index(['classifier'], inplace=True)
        return df

    #extraact the standard deviation for y-error
    def extract_stdev(a, b, c):
        a_s_std = d1[[f'{a}']]
        i_s_std = d2[[f'{b}']]
        r_s_std = d3[[f'{c}']]
        df1_std = pd.concat([models, a_s_std, i_s_std, r_s_std], axis =1)
        df1_std.set_index(['classifier'], inplace=True)
        yerr_df1 = df1_std.iloc[:, :].to_numpy().T
        return df1_std, yerr_df1

    #Training precision
    Training_precision = extract_average('tr_precision_avg_as','tr_precision_avg_is', 'tr_precision_avg_rs')
    #Training precision yerror
    Training_precision_yerror =  extract_stdev('tr_precision_stdev_as', 'tr_precision_stdev_is','tr_precision_stdev_rs')[1]
    #Training recall
    Training_recall = extract_average('tr_recall_avg_as','tr_recall_avg_is', 'tr_recall_avg_rs')
    #Training recall yerror
    Training_recall_yerror = extract_stdev('tr_precision_stdev_as', 'tr_precision_stdev_is','tr_precision_stdev_rs')[1]
    #Training F1
    Training_F1 = extract_average('tr_f1_avg_as','tr_f1_avg_is', 'tr_f1_avg_rs')
    #Training F1 yerror
    Training_F1_yerror = extract_stdev('tr_f1_stdev_as', 'tr_f1_stdev_is','tr_f1_stdev_rs')[1]

    #Test precision
    Test_precision = extract_average('te_precision_avg_as','te_precision_avg_is', 'te_precision_avg_rs')
    #Test precision yerror
    Test_precision_yerror =  extract_stdev('te_precision_stdev_as', 'te_precision_stdev_is','te_precision_stdev_rs')[1]
    #Test recall
    Test_recall = extract_average('te_recall_avg_as','te_recall_avg_is', 'te_recall_avg_rs')
    #Test precision yerror
    Test_recall_yerror =  extract_stdev('te_recall_stdev_as', 'te_recall_stdev_is','te_recall_stdev_rs')[1]
    #Test F1
    Test_F1 = extract_average('te_f1_avg_as','te_f1_avg_is', 'te_f1_avg_rs')
    #Test precision stdev
    Test_F1_stdev = extract_stdev('te_f1_stdev_as', 'te_f1_stdev_is','te_f1_stdev_rs')[0]
    #Test precision yerror
    Test_F1_yerror =  extract_stdev('te_f1_stdev_as', 'te_f1_stdev_is','te_f1_stdev_rs')[1]

    #Export separate dataframe of f1 and f1 stdev
    Test_F1.to_csv(f'{bacteria}_{antb}_F1_comparision_{validation_no}-fold_CV.csv', encoding='utf-8')
    Test_F1_stdev.to_csv(f'{bacteria}_{antb}_F1_stdev_comparision_{validation_no}-fold_CV.csv', encoding='utf-8')

    #Ten fold CV
    TenFold_CV = extract_average('Tf_CV_Avg_as','Tf_CV_Avg_is', 'Tf_CV_Avg_rs')
    #Ten fold yerror
    TenFold_CV_yerror =  extract_stdev('Tf_CV_stdev_as', 'Tf_CV_stdev_is','Tf_CV_stdev_rs')[1]

    #Loo CV
    Loo_CV = extract_average('Loo_CV_as','Loo_CV_is', 'Loo_CV_Avg_rs')
    #Ten fold yerror
    Loo_CV_yerror =  extract_stdev('Loo_stdev_as', 'Loo_stdev_is','Loo_CV_stdev_rs')[1]

    #Au_ROC
    Au_ROC = extract_average('au_ROC_avg_as','au_ROC_avg_is', 'au_ROC_avg_rs')
    #TAu_ROC yerror
    Au_ROC_yerror =  extract_stdev('au_ROC_stdev_as', 'au_ROC_stdev_is','au_ROC_stdev_rs')[1]

    #Au_PR
    Au_PR = extract_average('au_PR_avg_as','au_PR_avg_is', 'au_PR_avg_rs')
    #TAu_ROC yerror
    Au_PR_yerror = extract_stdev('au_PR_stdev_as', 'au_PR_stdev_is','au_PR_stdev_rs')[1]

    my_labels=['All Set', 'Intersection Set', 'Random Set']
    rc('text', usetex=True)
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(15,10))

    plt.xlabel("")
    plt.margins(y=0)
    plt.grid(linestyle='-', axis='y', lw=0.1)

    ax1 = Training_precision.plot(kind='bar', width= 0.8, ax=axes[0,0], yerr= Training_precision_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "i. Training precisiion",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax1.set_xlabel('')
    ax1.set_axisbelow(True)
    ax1.margins(0)

    ax2 = Training_recall.plot(kind='bar', width= 0.8, ax=axes[0,1], yerr= Training_recall_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "ii. Training recall",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax2.set_xlabel('')
    ax2.set_axisbelow(True)
    ax2.margins(0)

    ax3 = Training_F1.plot(kind='bar', width= 0.8, ax=axes[0,2], yerr= Training_F1_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "iii. Training F1",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax3.set_xlabel('')
    ax3.set_axisbelow(True)
    ax3.margins(0)

    ax4 = Test_precision.plot(kind='bar', width= 0.8, ax=axes[1,0], yerr= Test_precision_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "iv. Test precisiion",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax4.set_xlabel('')
    ax4.set_axisbelow(True)
    ax4.margins(0)

    ax5 = Test_recall.plot(kind='bar', width= 0.8, ax=axes[1,1], yerr= Test_recall_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "v. Test recall",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax5.set_xlabel('')
    ax5.set_axisbelow(True)
    ax5.margins(0)

    ax6 = TenFold_CV.plot(kind='bar', width= 0.8, ax=axes[1,2], yerr= TenFold_CV_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "vi. 10-fold CV",
             legend=False, yticks=np.arange(0,1.5, step=0.1),
             fontsize=8, rot=0)
    ax6.set_xlabel('')
    ax6.set_axisbelow(True)
    ax6.margins(0)
    
    ax7 = Loo_CV.plot(kind='bar', width= 0.8, ax=axes[2,0], yerr= Loo_CV_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "vii. Loo CV",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax7.set_xlabel('')
    ax7.set_axisbelow(True)
    ax7.margins(0)
    
    ax8 = Au_ROC.plot(kind='bar', width= 0.8, ax=axes[2,1], yerr= Au_ROC_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "viii. au ROC",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax8.set_xlabel('')
    ax8.set_axisbelow(True)
    ax8.margins(0)

    ax8.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
               bbox_to_anchor=(0.5, -0.25),
               fancybox=False, shadow=False, prop={'size': 8})
    ax8.set_axisbelow(True)

    ax9 = Au_PR.plot(kind='bar', width= 0.8, ax=axes[2,2], yerr= Au_PR_yerror, error_kw=dict(lw = 0.5, capsize = 1, capthick =0.5), 
             title= "ii. au PR",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax9.set_xlabel('')
    ax9.set_axisbelow(True)
    ax9.margins(0)
    
    t = f"Suplementary Figure {supplementary_fig_no}. Assessment of the performance of the machine learning algorithms in predicting resistance to {antb} by {italic_name} in {validation_no}-fold cross validation settings. The preformance metrics i) training precision, \nii) training recall, iii) training F1, iv) test precision, v) test recall, vi) test F1, vii) 10f CV (ten-fold cross validation), viii) Loo CV (leave-one-out cross validation), ix) au PR (area under precision recall curve), and x) au ROC (area under ROC curve). \n'All'denotes all AMR genes for taraining (as in the cross-validation partioning), 'Intersection' refers to AMR genes that consistently ranked high across all 6 rounds of cross-validation, and 'Random' refers to randomly sampled AMR genes."

    import textwrap as tw
    fig_txt= tw.fill(tw.dedent(t.strip() ), width=250)

    plt.figtext(0.5, 0.03, fig_txt, horizontalalignment='center',
                fontsize=10, multialignment='left')
    fig.tight_layout()

    plt.subplots_adjust(top=0.97, bottom=0.15, hspace=0.29, wspace=0.1 )
    fig.savefig(f'ML_Plot_{bacteria}_{antb}_{validation_no}-fold_CV.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    plt.show()
