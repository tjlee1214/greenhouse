Overview of Greenhouse:
Greenhouse is a LSTM-based anomaly detection algorithm for time-series data. It is zero-positive (e.g. does not require positive (anomalous) training samples), which significantly reduces the burden of collecting a large high-quality training data.
Greenhouse largely leverages the architecture of the non-zero-positive LSTM-based algorithm from the paper, “Long Short Term Memory Networks for Anomaly Detection in Time Series”, by Pankaj Malhotra. This detection algorithm has two major componenets, prediction model and detetion threshold. Prediction model is used to predict future data values using the previously observed values. In the testing phase, the detection algorithm computes the difference between true incoming value and the value predicted by the prediction model. Subsequently, the algorithm flags the incoming data value as an anomaly if this difference exceeds a certain threshold. This threshold is learned in the training phase and the paper algorithm requires anomalies in the training data to learn an optimal detection threshold. Greenhouse obviates this need for training on anomalous samples by building the distribution of deviation of normal error vectors (e.g. difference between predicted values and true values for non-anomalous samples) from their mean and finding an optimal threshold. 
This code is a prototype of Greenhouse. 

Installation Instruction:
Install Anaconda3 since Greenhouse depends on Python 3. 
https://www.continuum.io/downloads

Dependencies:
pip install numpy
pip install tensorflow
pip install tflearn
pip install keras

Running Greenhouse-BMW.py:
python Greenhouse-BMW.py <train-file> <test-file>
e.g.
python Greenhouse-BMW.py nyc_taxi-train.csv nyc_taxi-test.csv
