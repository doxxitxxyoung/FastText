import tensorflow as tf
import random
import numpy as np
import os
import pickle
import argparse
#from modelcv import Model
from modelhv import Model
import math
import time
import multiprocessing
from nltk.util import ngrams
from sklearn import feature_extraction

#   This One Uses Hashing Vectorizer! 

random.seed(4444)
parser = argparse.ArgumentParser()
start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#pool = multiprocessing.Pool(processes = 20)

#   PYTHON3 TENSORFLOW 1.8

def main(argv):
    
    use_gpus = "/cpu:0"

    with tf.device(use_gpus):            
        with open(args.data_dir + "train.pickle", 'rb') as f:
            train_data = pickle.load(f)
        with open(args.data_dir + "test.pickle", 'rb') as f:
            test_data = pickle.load(f)
        print(train_data[3]) 
        #   Add UnK as Unknown token for 2 dictionaries
        args.train_len = len(train_data)
        args.test_len = len(test_data)
     
        #random.shuffle(train_data)
        #random.shuffle(test_data)
     
     
        #   Train model

        with tf.Graph().as_default():
            # use 20 thread
            sess_config = tf.ConfigProto(intra_op_parallelism_threads=args.num_threads) 
            with tf.Session(config = sess_config) as sess:
                model = Model(args)
                label_placeholder = tf.placeholder(tf.int32, name = "label_input")
                ngram_placeholder = tf.placeholder(tf.string,
                                                    name ="ngram_input")

                logits, loss, preds = model.forward(ngrams = ngram_placeholder,
                                            batch_labels = label_placeholder)
 
                 
                optimizer = tf.train.GradientDescentOptimizer(args.learning_rate).minimize(loss)
                sess.run(tf.global_variables_initializer())
                for epoch in range(args.num_epochs):
                #for epoch in range(2):
                    for trainiter in range(0, args.train_len):
                        sess.run(optimizer, feed_dict = 
                                       { label_placeholder : train_data[trainiter][0],
                                       ngram_placeholder : train_data[trainiter][1]
                                       }
                                )

                        if trainiter%10000 == 0:
                            #train_loss, train_accu = sess.run([model.loss, model.train_accu],
                            print("Process: %d/129000"%trainiter)
                            print("Epoch: %d/5"%(epoch+1))

                #   Test
                correct = 0
                missed = 0
                for testiter in range(0, args.test_len):
                    test_preds = sess.run([model.preds], feed_dict = 
                                                {label_placeholder : test_data[testiter][0],
                                                 ngram_placeholder : test_data[testiter][1]
                                                            }
                                                              )
                    """
                    print("test preds")
                    print(test_preds)
                    print("test label")
                    print(test_data[testiter][0])
                    """
                    if test_preds[0] +1 == test_data[testiter][0]:
                        correct += 1
                    else:
                        missed += 1
                accuracy = float(correct)/float(correct + missed)
                end = time.time()
                print("=    =   =   =   =   =   =   =   =   =   =   =   =   =   =   =   =") 
                print("Test Accuracy :") 
                print(accuracy)
                print("=    =   =   =   =   =   =   =   =   =   =   =   =   =   =")
                print("Execution Time")
                print(end - start)
                print("=    =   =   =   =   =   =   =   =   =   =   =   =   =   =")
          



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = '/home/doyeong/fasttext/data/ag_news_csv/')
    parser.add_argument('--log_dir', type = str, default = '/home/doyeong/fasttext/log.txt')
    #parser.add_argument('--use_gpu', type = int, default = 4, help = "0 for gpu0, 1 for gpu1, else for cpu")
    
    parser.add_argument('--num_hidden_units', type = int, default = 10)
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--learning_rate', type = int, default = 0.05)
    #parser.add_argument('--learning_rate', type = list, default = [0.05, 0.1, 0.25, 0.5])


    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--max_hash_bin', type = int, default = 10**7)
    parser.add_argument('--num_of_labels', type = int, default = 4)
    parser.add_argument('--max_seq_len', type = int, default = 200)
    parser.add_argument('--num_threads', type = int, default = 20)
    args = parser.parse_args() 

tf.app.run()
