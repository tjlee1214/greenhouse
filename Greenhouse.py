import numpy as np
from tflearn.data_utils import load_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import multivariate_normal
import sys
import math
from numpy.linalg import inv
from scipy.stats import norm
from scipy.stats import truncnorm
import time
import csv

# ignore timestamp column in data and return data as np array of floats
def preprocess(data, cols_to_ignore):
    for id in sorted(cols_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    return np.array(data, dtype=np.float32)

#input file should be in the following format ->(anomaly label, timestamp, data value)
#'1' label means an anomaly
def load_data(input_file):
    my_columns_to_ignore = [1]
    my_target_column = 2
    anomaly_flag, data_values = load_csv(input_file, columns_to_ignore=my_columns_to_ignore, target_column=my_target_column)
    feature_range_start_value = -1
    feature_range_end_value = 1
    scaler = MinMaxScaler(feature_range=(feature_range_start_value, feature_range_end_value))
    data = scaler.fit_transform(data_values)
    return data, anomaly_flag 

#Compute an array of anomaly windows(i.e a sequence of 1's) from an array of 1's and 0's
def compute_anomaly_windows(anomaly_flag):
    anomaly_windows = []
    start_index = -1
    end_index = -1
    for i in range(len(anomaly_flag)):
        if anomaly_flag[i][0] == '1':
            if start_index==-1:
                start_index = i
                end_index = i
            elif end_index+1 != i:
                anomaly_windows.append([start_index,end_index])
                start_index = i
                end_index = i
            else:
                end_index = i

    #Save the last anomaly window information (e.g startIndex & endIndex) into the anomaly_windows only if it has not been saved in anomaly_winows list yet
    if anomaly_observed(start_index) and (is_anomaly_windows_empty(anomaly_windows) or anomaly_window_already_saved(anomaly_windows, start_index)==False):
        anomaly_windows.append([start_index, end_index])

    return anomaly_windows 

#returns true if we have observed an anomaly
def anomaly_observed(start_index):
    return start_index!=-1

#returns true if anomaly_windows list is empty
def is_anomaly_windows_empty(anomaly_windows):
    return len(anomaly_windows)==0

#returns true if the current anomaly window represented by start_index is already saved in the anomaly_windows list
def anomaly_window_already_saved(anomaly_windows, start_index):
    return anomaly_windows[len(anomaly_windows)-1][0]==start_index

#convert an array of data values into a dataset matrix
def create_dataset(data, lookback, predict_num):
    dataX, dataY = [], []
    for i in range(len(data)-predict_num-lookback+1):
        a = np.array(data[i:i+lookback]).astype(np.float)
        b = np.array(data[i+lookback:(i+lookback+predict_num)]).astype(np.float)
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)

#preprocess data and reshape into [num of samples, size of lookback or predict_num, num of features]
def preprocess_data(train_data, test_data, lookback, predict_num):
    #reshape into X=t and Y=(label for t+i)
    train_data_x, train_data_y = create_dataset(train_data,
        lookback, predict_num)

    test_data_x, test_data_y = create_dataset(test_data, lookback, predict_num)

    #reshape input to be [num of samples, size of lookback or predict_num, num of features]
    train_data_x = np.reshape(train_data_x,
        (train_data_x.shape[0], train_data_x.shape[1], 1))
    test_data_x = np.reshape(test_data_x,
        (test_data_x.shape[0], test_data_x.shape[1], 1))

    return train_data_x, train_data_y, test_data_x, test_data_y

#create and fit LSTM network model with softmax activation
def fit_model(train_data_x, train_data_y, epochs, batch_size, predict_num):
    model = Sequential()

    model.add(LSTM(35, input_shape=(train_data_x.shape[1], train_data_x.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(35, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(predict_num, activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop',
        metrics=['accuracy'])
    model.fit(train_data_x, train_data_y, epochs=epochs,
        batch_size=batch_size, verbose=2)

    return model




#compute classical metrics: recall, precision, F scores
#predicted_anomaly_set is a set
def compute_metrics_classical(predicted_anomaly_set, actual_data, beta):
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0

    for i in range(len(actual_data)):
        actual_data_int = int(actual_data[i][0])
        if i in predicted_anomaly_set:
            if actual_data_int == 0:
                false_pos += 1
            else:
                true_pos += 1
        else:
            if actual_data_int == 0:
                true_neg += 1
            else:
                false_neg += 1 

    precision = true_pos/(true_pos+false_pos) if (true_pos+false_pos) > 0 else 0
    recall = (true_pos)/(true_pos+false_neg) if (true_pos+false_neg) > 0 else 0

    F_beta = (1+((beta)**2)) * ((precision * recall)/(((beta)**2 * precision) +recall)) if (precision+recall) != 0 else 0.0
    F1 = 2 * ((precision * recall)/(precision + recall)) if (precision+recall) != 0 else 0.0

    return precision, recall, F_beta, F1


# Calculate the error vector with granularity 1 and precalculated predicted values
def compute_error_vectors(orig_data, predict_data, predict_num):
    error_vectors = []
    for i in range(len(orig_data)-predict_num+1):
        a = []
        real_value = orig_data[predict_num - 1 + i, 0]
        for j in range(predict_num):
            a.append(real_value - predict_data[predict_num - 1 + i - j, j])
        error_vectors.append(a)
    return np.array(error_vectors)


# Calcute the error vector for range [start, end)  (start included but end not included) with a specific granularity value
#                  [start, end) should be within the range [predict_num-1, len(orig_data)-predict_num+1)                                            
# We predict the value for specific position when needed 
#   @data_x: data used to predict data_y  
#   @data_y: true data we're trying to predict
#   @granularity: gap between two consecutive sampled points 
def compute_error_vectors_with_granularity(data_x, data_y, model, predict_num, granularity, start, end):

    error_vectors = []
    # smallest start index is predict_num - 1
    if start < predict_num - 1:
        cur = predict_num - 1
    else:
        cur = start
    while cur < end and cur < len(data_y)-predict_num+1:
        # this part can be changed if we pre-calculate all the predictions
        data_used_for_prediction = data_x[cur-predict_num+1:cur+1]
        predicts = model.predict(data_used_for_prediction)

        a = []
        real_value = data_x[cur, 0]
        for j in range(predict_num):
            a.append(real_value - predicts[predict_num - 1 - j, j])
        error_vectors.append(a)
        cur = cur + granularity

    return np.array(error_vectors)




#   @index: indicates which subset of the whole data you want to use for this particular operation. One-based index 
def train_prediction_model(train_data_x_total, train_data_y_total, index, num_of_subsets, predict_num):
    #data for prediction model training
    
    start, end = compute_indices(len(train_data_x_total), index, num_of_subsets) 
       
    train_data_x_for_prediction_model_training = train_data_x_total[start:end]
    train_data_y_for_prediction_model_training = train_data_y_total[start:end]
    
    #data-value prediction model
    model = fit_model(train_data_x_for_prediction_model_training,  train_data_y_for_prediction_model_training, epochs=10, batch_size=32, predict_num=predict_num)
    return model

#Returns mean and cov that represent the Gaussian distribution of error vectors
def model_error_vector_dist(train_data_x_total, train_data_y_total, index, num_of_subsets, model, predict_num):
    
    train_data_y_predicted_total = model.predict(train_data_x_total)

    start, end = compute_indices(len(train_data_x_total), index, num_of_subsets)

    #data for error vector distribution modeling    
    train_data_x_for_error_vector_dist_modeling = train_data_x_total[start:end]
    train_data_y_for_error_vector_dist_modeling = train_data_y_total[start:end]
    train_data_y_predicted_for_error_vector_dist_modeling = train_data_y_predicted_total[start:end]

    error_vectors_for_error_vector_dist_modeling = compute_error_vectors(train_data_y_for_error_vector_dist_modeling, train_data_y_predicted_for_error_vector_dist_modeling, predict_num)
    myMean = np.mean(error_vectors_for_error_vector_dist_modeling, axis=0)
    myCov = np.cov(error_vectors_for_error_vector_dist_modeling, rowvar=False)
    
    myMean_reshaped = np.reshape(myMean, (myMean.shape[0], 1))
    
    if myCov.ndim==0:
        myCov = np.array([[myCov]])

    return myMean_reshaped, myCov   


def learn_threshold(train_data_x_total, train_data_y_total, train_labels_total, index, num_of_subsets, model, mean_ev, cov_ev, predict_num):
    
    start, end = compute_indices(len(train_data_x_total), index, num_of_subsets) 

    #data for error vector distribution modeling    
    train_data_x_for_threshold_learning = train_data_x_total[start:end]
    train_data_y_for_threshold_learning = train_data_y_total[start:end]
    train_labels_for_threshold_learning = train_labels_total[start:end]

    threshold = learn_threshold_helper(mean_ev, cov_ev, train_data_x_for_threshold_learning, train_data_y_for_threshold_learning, train_labels_for_threshold_learning, predict_num, model)
    return threshold

def learn_threshold_helper(mean, cov, data_x, data_y, labels, predict_num, model):
    error_vectors_threshold_learning = compute_error_vectors(data_y, model.predict(data_x), predict_num)

    #Using Mahalanobis distance
    mDists = []

    for i in range(len(error_vectors_threshold_learning)):
        if labels[i][0] == '1':
            continue
        cur_error_vector = np.reshape(error_vectors_threshold_learning[i], (error_vectors_threshold_learning[i].shape[0], 1))
        mDist = math.sqrt(np.dot(np.dot(np.transpose(cur_error_vector-mean), inv(cov)), cur_error_vector-mean))
        mDists.append(mDist)

    mu, std = norm.fit(mDists)

    alpha = (0 - mu) / std
    beta = (float('inf') - mu) / std
    Z = 1 - norm.cdf(alpha, 0, 1)
    mu_t = mu + norm.pdf(alpha, 0, 1) / Z * std
    std_t = math.sqrt(std * std * (1 + alpha * norm.pdf(alpha, 0, 1) / Z - (norm.pdf(alpha, 0, 1) / Z)**2))    
    normal_percentile = .85
    threshold = truncnorm.ppf(normal_percentile, alpha,beta, loc=mu_t, scale=std_t)
    
    return threshold


def compute_indices(length, index, num_of_subsets):
    start = int(length / num_of_subsets) * (index-1)
    end=0
    if index==num_of_subsets:
        end = length
    else :
        end = int(length / num_of_subsets) * index
    return start, end

def test(model, predict_num, granularity, lookback, test_data_x, test_data_y, mean_ev, cov_ev, threshold):
    #Testing the performance of Greenhouse 
    suspicious_regions = []  #the initial region should be the whole dataset except those we can't predict enough times
    start_test = predict_num-1   
    end_test = len(test_data_y)-predict_num+1
    error_vectors = compute_error_vectors_with_granularity(test_data_x, test_data_y, model=model, predict_num=predict_num, granularity=granularity, start=start_test, end=end_test)
    
    suspicious_points = set()
    
    for i in range(len(error_vectors)):
        cur_error_vector = np.reshape(error_vectors[i], (error_vectors[i].shape[0], 1))
        mDist = math.sqrt(np.dot(np.dot(np.transpose(cur_error_vector-mean_ev), inv(cov_ev)), cur_error_vector-mean_ev))
        original_index =start_test + i * granularity + lookback + predict_num - 1
        if mDist > threshold:
            suspicious_points.add(original_index)

    return suspicious_points    
     
#run the network: preprocess data, fit prediction model and learn an optimal threshold
def run_lstm_network(train_file, test_file, lookback, predict_num, granularity):
    orig_train_data,  orig_train_labels = load_data(train_file)
    orig_test_data, orig_test_labels = load_data(test_file)

    #data_y is a vector of next predict_num number of values for a given train_data_x vector
    train_data_x_total, train_data_y_total, test_data_x, test_data_y = preprocess_data(orig_train_data, orig_test_data, lookback, predict_num)
 
    train_labels_total = orig_train_labels[predict_num + lookback - 1:]
    test_labels = orig_test_labels[predict_num + lookback - 1:]

    #test data is divided into 3 subsets for prediciton_model, error_vector_distribution and threshold_learning
    num_of_subsets = 3

    model = train_prediction_model(train_data_x_total, train_data_y_total, 1, num_of_subsets, predict_num)    

    mean_ev, cov_ev = model_error_vector_dist(train_data_x_total, train_data_y_total, 2, num_of_subsets, model, predict_num)

    threshold = learn_threshold(train_data_x_total, train_data_y_total, train_labels_total, 3, num_of_subsets, model, mean_ev, cov_ev, predict_num)


    #Testing the performance of Greenhouse 
    suspicious_points = test(model, predict_num, granularity, lookback, test_data_x, test_data_y, mean_ev, cov_ev, threshold)
            
    print("predicted anomaly points: \n", suspicious_points, "\n")
    real_anomaly_ranges = compute_anomaly_windows(test_labels)
    print("real anomaly ranges\n", real_anomaly_ranges)    

    precision_classical, recall_classical, F_beta_classical, F1_classical = compute_metrics_classical(suspicious_points, test_labels, 0.1)
    print("recall_classical: {0:.5f}".format(recall_classical))
    print("precision_classical: {0:.5f}".format(precision_classical))
    print("F0.1_classical: {0:.5f}".format(F_beta_classical))
    print("F1_classical: {0:.5f}".format(F1_classical))


def main():
    if len(sys.argv) < 2:
        print("Usage: Greenhouse-BMW.py <train-file> <test-file>")
        return

    train_file = sys.argv[1]
    test_file = sys.argv[2]

    run_lstm_network(train_file, test_file, lookback=10, predict_num=5, granularity=1)


if __name__ == '__main__':
    main()
