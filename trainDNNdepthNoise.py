# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random

batch_size = 100
training_epochs = 10000
display_epochs = 100

lossDict = {}

for coil in [1, 3, 5, 7]:
    coilLoss = {}
    for lift in [1, 3, 5]:
        liftLoss = {}
        for frequency in [25, 100, 400]:
            
            fname = "dataset_plus_fre" + str(frequency)

            dataset = pickle.load(open("data/" + fname + ".pickle", "rb"))

            dataset = dataset[coil][lift]

            # 導電率0のみ使用
            datasetDummy = []
            for data in dataset:
                if data[5] == 0:
                    datasetDummy.append(data)
            dataset = datasetDummy
            datasetMatrix = np.zeros((10000, dataset[0].shape[0]))
            for i in range(10000):
                data = dataset[random.randint(0, len(dataset)-1)]
                for k in range(99):
                    if k < 6:
                        datasetMatrix[i, k] = data[k]
                    else:
                        datasetMatrix[i, k] = data[k] * (random.random() * 0.2 + 0.9)
            
            dataset = datasetMatrix
            widthRMS = 0.5
            depthRMS = 10
            coilRMS = 7
            liftRMS = 5
            frequencyRMS = 400
            conductivityRMS = 100
            # 規格化
            dataset[:, 0] /= widthRMS
            dataset[:, 1] /= depthRMS
            dataset[:, 2] /= coilRMS
            dataset[:, 3] /= liftRMS
            dataset[:, 4] /= frequencyRMS
            dataset[:, 5] /= conductivityRMS
            dataset[:, 6:] -= np.min(dataset[:, 6:])
            dataset[:, 6:] /= np.max(dataset[:, 6:])
            # 型変換
            dataset = np.array(dataset, dtype=np.float32)
            # datasetを分ける
            datasetTrain = dataset[:9000, :]
            datasetTest = dataset[9000:, :]
            pickle.dump(datasetTrain, open("data/" + fname + "_" + str(coil) + "_" + str(lift) + "_0_Train_Noise.pickle", "wb"))
            pickle.dump(datasetTest, open("data/" + fname + "_" + str(coil) + "_" + str(lift) + "_0_Test_Noise.pickle", "wb"))

            tf.reset_default_graph()
            x = tf.placeholder("float", shape=[None, 93])
            y_ = tf.placeholder("float", shape=[None, 1])
            
            # 荷重作成
            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)
            # バイアス作成
            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)
            
            W_fc1 = weight_variable([93, 1024])
            b_fc1 = bias_variable([1024])
            h_flat = tf.reshape(x, [-1, 93])
            h_fc1 = tf.nn.sigmoid(tf.matmul(h_flat, W_fc1) + b_fc1)
            
            keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
            
            W_fc2 = weight_variable([1024, 1])
            b_fc2 = bias_variable([1])
            y_out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            
            each_square = tf.square(y_ - y_out)
            loss = tf.reduce_mean(each_square)
            train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
            
            sess = tf.Session()

            saver = tf.train.Saver()

            sess.run(tf.initialize_all_variables())
            # saver.restore(sess, "./20181108DNNmodeldepth" + fname)
            for i in range(training_epochs):
                for k in range(0, len(datasetTrain), batch_size):
                    batch = np.zeros((batch_size, 93))
                    batch = np.array(batch, dtype=np.float32)
                    batch[:, :] = datasetTrain[k:k+batch_size, 6:]
                    output = datasetTrain[k:k+batch_size, 1:2]
                    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
                
                if i%display_epochs == 0:
                    batch = np.zeros((datasetTrain.shape[0], 93))
                    batch = np.array(batch, dtype=np.float32)
                    batch[:, :] = datasetTrain[:, 6:]
                    output = datasetTrain[:, 1:2]
                    train_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            
                    batch = np.zeros((datasetTest.shape[0], 93))
                    batch = np.array(batch, dtype=np.float32)
                    batch[:, :] = datasetTest[:, 6:]
                    output = datasetTest[:, 1:2]
                    test_loss = loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            
                    print(str(i) + "epochs_finished!")
                    print("train_loss===" + str(train_loss))
                    print("test_loss===" + str(test_loss))
            saver.save(sess, "./20181109DNNmodeldepthNoise" + fname + "_" + str(coil) + "_" + str(lift) + "_0")
            sess.close()
            liftLoss[frequency] = (train_loss, test_loss)
        coilLoss[lift] = liftLoss
    lossDict[coil] = coilLoss
pickle.dump(lossDict, open("lossDict20181109DNNmodel_dataset_plus_fre25_con0_noise.pickle", "wb"))