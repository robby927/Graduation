import numpy as np
import time
import sklearn.metrics

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=1)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    import sklearn.tree as tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def precision_score(test_y, predict):
    right_class = 0
    len_test_y = len(test_y)
    len_predirect = len(predict)
    min_len = 0
    max_len = 0
    if(len_test_y < len_predirect):
        min_len = len_test_y
        max_len = len_predirect
    else:
        min_len = len_predirect
        max_len = len_test_y
    for i in range(min_len):
        if(test_y[i] == predict[i]):
            right_class += 1
    score = right_class/max_len
    return score

def read_data_set(data_file):
    f = open(data_file, "r")
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float, line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    data_array = np.array(data_list)

    return data_array;

if __name__ == '__main__':
    data_file = "H:\\Research\\data\\trainCG.csv"
    thresh = 0.1
    model_save_file = None
    model_save = {}

    test_classifiers = [
        # 'NB',
        # 'KNN',
        # 'LR',
        'RF',
        # 'SVM',
        # 'DT',
        # #'SVMCV',
        # 'GBDT'
    ]

    diagnosis = ['前列腺增生', '前列腺增生伴结石', '前列腺增生伴钙化', '子宫多发肌瘤', '子宫肌瘤', '左肾囊肿', '甲状腺结节', '肝囊肿', '胆囊结石', '轻度脂肪肝', '颈动脉粥样硬化', '颈椎病']
    classifiers = {
                   'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier
                   }

    print('reading training and testing data...')
    # train_x, train_y, test_x, test_y = read_data(data_file)
print('disease', '\tprecision', '\trecall', '\tF1', '\tfeatures', '\tpositive samples', '\tnegative samples', 'pos/neg')
for diseaseName in diagnosis:

    flag = 2
    if(flag == 1):
        directory = "pcapostdata"
    else:
        directory = "vectordata"

    train_y = read_data_set("D:\Projects\Robby\Python\AI\experiment\\"+directory+"\\"+diseaseName+".vector.trainY.data");
    train_x = read_data_set("D:\Projects\Robby\Python\AI\experiment\\"+directory+"\\"+diseaseName+".vector.trainX.data");

    test_x = read_data_set("D:\Projects\Robby\Python\AI\experiment\\"+directory+"\\"+diseaseName+".vector.testX.data");
    test_y = read_data_set("D:\Projects\Robby\Python\AI\experiment\\"+directory+"\\"+diseaseName+".vector.testY.data");

    for classifier in test_classifiers:
        #print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y.ravel())
        #print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        precision = precision_score(test_y, predict)
        recall = sklearn.metrics.recall_score(test_y, predict)
        positive_samples = np.sum(train_y == 1)+np.sum(test_y==1)
        negative_samples = np.sum(train_y==-1)+np.sum(test_y==-1)

        print(diseaseName, '\t', precision, '\t', recall, '\t', 2*precision*recall/(precision+recall), '\t', test_x[0].size, '\t', positive_samples, '\t', negative_samples, '\t', positive_samples/(negative_samples+positive_samples))
        # accuracy = metrics.accuracy_score(test_y, predict)
        # print('accuracy: %.2f%%' % (100 * accuracy))

