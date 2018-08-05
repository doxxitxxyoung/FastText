import tensorflow as tf
import numpy as np

class Model(object):
    def __init__(self, args):

        self.num_hidden_units = args.num_hidden_units
        self.max_hash_bin = args.max_hash_bin
        self.num_of_labels = args.num_of_labels
        self.hashlookup = tf.Variable(tf.random_uniform(
                                        [self.max_hash_bin, self.num_hidden_units]),
                                        name = "hash_lookup_table"
                                        )

    def forward(self, ngrams, batch_labels):
        ngram_hash = tf.string_to_hash_bucket_fast(ngrams, 
                                                    num_buckets = self.max_hash_bin)
        hash_embed = tf.nn.embedding_lookup(self.hashlookup, ngram_hash)
        hash_mean = tf.reshape(tf.reduce_sum(hash_embed, axis=0), [1,self.num_hidden_units])
        self.logits = tf.layers.dense(hash_mean, self.num_of_labels,
                                      name = "hidden2output"
                                      )
        one_hot_labels = tf.one_hot(batch_labels-1, self.num_of_labels)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels = one_hot_labels, logits = self.logits)
        self.preds = tf.argmax(self.logits, axis=-1)
                

        return self.logits, self.loss, self.preds
