# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

loss_dict = {}
for coil in [1, 3, 5, 7]:
    loss_dict_coil = {}
    for lift in [1, 3, 5]:
        loss_dict_lift = {}
        for frequency in [25, 100, 400]:
            
            fname = "dataset_plus_fre" + str(frequency)
            """
            dataset = pickle.load(open("data/" + fname + ".pickle", "rb"))
            dataset = dataset[coil][lift]
            # 導電率0のみ使用
            datasetDummy = []
            for data in dataset:
                if data[5] == 0:
                    datasetDummy.append(data)
            dataset = datasetDummy
            datasetMatrix = np.zeros((len(dataset), dataset[0].shape[0]))
            for i in range(len(dataset)):
                datasetMatrix[i, :] = dataset[i]
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
            """
            dataset = pickle.load(open("data/" + fname + "_" + str(coil) + "_" + str(lift) + "_0_Test_Noise.pickle", "rb"))
            
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
            saver.restore(sess, "./20181109DNNmodeldepthNoise" + fname + "_" + str(coil) + "_" + str(lift) + "_0")
                
            batch = np.zeros((dataset.shape[0], 93))
            batch = np.array(batch, dtype=np.float32)
            batch[:, :] = dataset[:, 6:]
            output = dataset[:, 1:2]
            loss_dict_lift[str(frequency)] = np.sqrt(loss.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0}))
            prediction = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
            
            indexes = np.argsort(dataset[:, 1])
            plotList = []
            for i in range(len(indexes)):
                index = indexes[i]
                if i == 0:
                    plots = []
                elif dataset[index, 1] != dataset[indexes[i-1], 1]:
                    plotList.append(plots)
                    plots = []
                plots.append(prediction[index, 0])
            plotList.append(plots)
            
            
            plt.figure()
            for i in range(len(plotList)):
                plots = plotList[i]
                for plot in plots:
                    if i < 5:
                        plt.scatter((i+1)*0.2, plot*10, c="blue")
                    else:
                        plt.scatter((i-2)*0.5, plot*10, c="blue")

            plt.scatter(0.2, 0.2, c="red", marker="*")
            plt.scatter(0.4, 0.4, c="red", marker="*")
            plt.scatter(0.6, 0.6, c="red", marker="*")
            plt.scatter(0.8, 0.8, c="red", marker="*")
            plt.scatter(1.0, 1.0, c="red", marker="*")
            plt.scatter(1.5, 1.5, c="red", marker="*")
            plt.scatter(2.0, 2.0, c="red", marker="*")
            plt.scatter(2.5, 2.5, c="red", marker="*")
            plt.scatter(3.0, 3.0, c="red", marker="*")
            plt.vlines(0.2, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(0.4, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(0.6, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(0.8, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(1.0, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(1.5, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(2.0, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(2.5, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            plt.vlines(3.0, 0, 3.5,  linestyle="dashed", linewidth=0.5)
            x = np.linspace(0, 3, 4)
            y = x
            plt.plot(x, y ,"r--")
            plt.xlabel("actual flaw depth (mm)", size = 14)
            plt.ylabel("estimate flaw depth(mm)", size = 14)
            plt.savefig("./20181109DNNmodeldepthNoise" + fname + "_" + str(coil) + "_" + str(lift) + "_0.jpg")
            plt.show()
        loss_dict_coil[str(lift)] = loss_dict_lift
    loss_dict[str(coil)] = loss_dict_coil