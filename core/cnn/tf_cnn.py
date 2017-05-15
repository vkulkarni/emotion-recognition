from core.cnn.cnn import CNN
import tensorflow as tf
import numpy as np
from datetime import datetime, date, time
import os
from core.utils.utils import Utils

class TensorflowCNN(CNN):

    def __init__(self):
        super(TensorflowCNN, self).__init__()
        self.supported_networks = [CNN.ALEXNET, CNN.VGG16]
        self.network = None
        self.learning_rate = 0.001
        self.epochs = 100
        self.batchsize = 100
        self.logs_dir = "logs/tf"   #for tensorboard
        self.checkpoint_path = "logs/checkpoint_model_tf/"  # path to save model checkpoint
        self.features_tensor = None
        self.labels_tensor = None
        self.display_step = 1
        self._index_for_batch = 0
        self.keep_prob = tf.placeholder(tf.float32)


    def cnn_layer(self, input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME',
                  relu=True):
        """
        Tensorflow single Convolutional Network layer.

        :param input:
        :param filter_height:
        :param filter_width:
        :param num_filters:
        :param stride_y:
        :param stride_x:
        :param name:
        :param padding:
        :param relu:
        :return:
        """

        # input channels (for RGB image it's 3).
        input_channels = int(input.get_shape()[-1])

        with tf.variable_scope(name) as scope:
            # Create tensorflow variables for the weights and biases of the conv layer
            weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

            cn = tf.nn.conv2d(input,
                              weights,
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding)

            # Add biases
            bias = tf.nn.bias_add(cn, biases)

            # tf histogram summay
            # tf.summary.histogram("weights",weights)
            # tf.summary.histogram("biases", biases)
            # tf.summary.histogram("activations", relu)

            # Apply relu function
            if relu:
                activation = tf.nn.relu(bias, name=scope.name)
                return activation

            return bias

    def fully_connected_layer(self, input, num_in, num_out, name, relu=True):
        """
        Tensorflow fully connected network layer.

        :param input:
        :param num_in:
        :param num_out:
        :param name:
        :param relu:
        :return:
        """
        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(input, weights, biases, name=scope.name)

            # tf histogram summay
            # tf.summary.histogram("weights",weights)
            # tf.summary.histogram("biases", biases)

            if relu:
                # Apply ReLu non linearity
                relu = tf.nn.relu(act)
                # tf.summary.histogram("activations", relu)
                return relu
            else:
                # tf.summary.histogram("activations", act)
                return act

    def max_pool(self, input, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """
        Tensorflow max pool CNN layer.

        :param input:
        :param filter_height:
        :param filter_width:
        :param stride_y:
        :param stride_x:
        :param name:
        :param padding:
        :return:
        """
        return tf.nn.max_pool(input,
                              ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding,
                              name=name)

    def dropout(self, input, value):
        """
        Tensorflow dropout layer

        :param input:
        :return:
        """
        return tf.nn.dropout(input, value)

    def train(self, X_train, y_train, X_test, y_test):
        """
        Train the Convolutional Neural Network.

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return: Bool
        """

        """
        If the self.network is None means create_classifier_network was not called. Create the network first.
        """
        if self.network is None:
            return False

        """
        Some data pre processing before training CNN model.
        """
        X_train = np.array(X_train)
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
        y_train = Utils.do_one_hot_encoding(y_train)  # labels to one hot encoding

        data_test = np.array(X_test)
        X_test = data_test.reshape(-1, data_test.shape[1], data_test.shape[2], 1)
        y_test = Utils.do_one_hot_encoding(y_test)  # labels to one hot encoding

        # set data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.train_features_len = len(X_train)
        self.test_features_len = len(X_test)

        print(X_train.shape)
        print(y_train.shape)
        print(self.network.shape)


        # Op for calculating the loss

        #image_summary = tf.summary.image('input', X_train, 3)

        with tf.name_scope("cross_ent"):
           # softmax = tf.nn.softmax_cross_entropy_with_logits(logits=self.model, y_train)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.network, labels=self.labels_tensor))

        # Train op
        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1).minimize(loss)

        # Add gradients to summary
        """
        for gradient, var in gradients:
            tf.summary.histogram(var.name + '/gradient', gradient)

        # Add the variables we train to the summary
        for var in var_list:
            tf.summary.histogram(var.name, var)
        """

        # Add the loss to summary
        tf.summary.scalar('cross-entropy-loss', loss)

        # Evaluation op: Accuracy of the model
        with tf.name_scope("acc"):
            correct_pred = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.labels_tensor, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Add the accuracy to the summary
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries together
        merged_summary = tf.summary.merge_all()

        # Initialize the FileWriter
        train_writer = tf.summary.FileWriter(self.logs_dir + '/train')
        validation_writer = tf.summary.FileWriter(self.logs_dir + '/validation')

        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = np.ceil(self.train_features_len / self.batchsize).astype(np.int16)
        test_batches_per_epoch = np.ceil(self.test_features_len / self.batchsize).astype(np.int16)

        # Start Tensorflow session and train the model..
        with tf.Session() as sess:

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Add the model graph to TensorBoard
            train_writer.add_graph(sess.graph)

            # Load the pretrained weights into the non-trainable layer
            #self.model.load_initial_weights(sess)

            print("{} Start training...".format(datetime.now()))
            print("{} Check Tensorboard at --logdir {}".format(datetime.now(), self.logs_dir))

            # Loop over number of epochs
            for epoch in range(self.epochs):
                print("{} Epoch number: {}/{}".format(datetime.now(), epoch + 1, self.epochs))

                step = 1
                data_cnt = 0
                while step <= train_batches_per_epoch:
                #while step <= train_batches_per_epoch:

                    # Get a batch of images and labels
                    batch_train_features, batch_train_labels, end_index = self._get_next_train_batch()

                    # And run the training op
                    sess.run(train_step, feed_dict={self.features_tensor: batch_train_features,
                                                  self.labels_tensor: batch_train_labels,
                                                  self.keep_prob: 1.0})

                    # Generate summary with the current batch of data and write to file
                    if step % self.display_step == 0:
                        training_summary, acc, lss = sess.run([merged_summary, accuracy, loss],
                                               feed_dict={self.features_tensor: batch_train_features,
                                                        self.labels_tensor: batch_train_labels,
                                                        self.keep_prob: 1.0})
                        #train_writer.add_summary(ms)
                        #train_writer.add_summary(ims)
                        train_writer.add_summary(training_summary, epoch * train_batches_per_epoch + step)
                        #train_writer.add_summary(tl, epoch * train_batches_per_epoch + step)

                        data_cnt += self.batchsize
                        #print("Step= " + "{:d}".format(step) + " {:d}/{:d}".format(end_index + 1, self.train_features_len) +
                        #      " Loss= " + "{:.6f}".format(lss) + ", Training Accuracy= " + "{:.6f}".format(acc))

                        print("\t" + "{:d}/{:d}".format(end_index + 1, self.train_features_len) +
                              " Training Loss= " + "{:.6f}".format(lss) +
                              ", Training Accuracy= " + "{:.6f}".format(acc),
                              end='\r')  # print/overwrite on the same line
                    step += 1
                print(end='\n')
                # Reset the file pointer of the image data generator
                self._index_for_batch = 0

                # Validate the model on the entire validation set
                test_loss = 0.
                test_acc = 0.
                test_count = 0
                for _ in range(test_batches_per_epoch):
                    batch_test_features, batch_test_labels, end_index = self._get_next_test_batch()
                    validation_summary, acc, lss = sess.run([merged_summary, accuracy, loss],
                                           feed_dict={self.features_tensor: batch_test_features,
                                                        self.labels_tensor: batch_test_labels,
                                                        self.keep_prob: 1.0})
                    validation_writer.add_summary(validation_summary, epoch * train_batches_per_epoch + test_count)
                    test_acc += acc
                    test_loss += lss
                    test_count += 1

                # Reset the file pointer of the image data generator
                self._index_for_batch = 0

                test_acc /= test_count
                test_loss /= test_count
                print("\tValidation Loss= " + "{:.6f}".format(test_loss) + " Validation Accuracy = {:6f}".format(test_acc))
                #print("\t\t{} Saving checkpoint of model...".format(datetime.now()))

                # save checkpoint of the model
                checkpoint_name = os.path.join(self.checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name)
                print("\t{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

        return True

    def create_classifier_network(self, input_image_size, num_classes, network_name=CNN.ALEXNET):
        if network_name in self.supported_networks:
            self.input_size = input_image_size
            self.num_classes = num_classes

            self.features_tensor = tf.placeholder(tf.float32, [None, input_image_size[0], input_image_size[1], 1])
            self.labels_tensor = tf.placeholder(tf.float32, [None, self.num_classes])

            """
            Call the network architecture method..
            """
            network_method = getattr(self, network_name)
            self.network = network_method(self.input_size, self.num_classes, self.features_tensor)
            return True
        return False

    def alexNet(self, input_image_size, num_classes, features_tensor=None):
        """
        Build AlexNet CNN - http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks

        :param input_image_size:
        :param num_classes:
        :param features_tensor:
        :return: tensor object
        """

        if features_tensor is None:
            features_tensor = tf.placeholder(tf.float32, [None, input_image_size[0], input_image_size[1], 1])

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = self.cnn_layer(features_tensor, 5, 5, 96, 1, 1, padding='SAME', name='conv1') #11, 11, 96, 4, 4
        pool1 = self.max_pool(conv1, 3, 3, 2, 2, padding='SAME', name='pool1')
        #norm1 = self.lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn
        #conv2 = self.cnn_layer(norm1, 3, 3, 256, 1, 1, name='conv2')
        conv2 = self.cnn_layer(pool1, 5, 5, 256, 1, 1, name='conv2')
        pool2 = self.max_pool(conv2, 3, 3, 2, 2, padding='SAME', name='pool2')
        #norm2 = self.lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        #conv3 = self.cnn_layer(norm2, 3, 3, 384, 1, 1, name='conv3')
        conv3 = self.cnn_layer(pool2, 4, 4, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu)
        conv4 = self.cnn_layer(conv3, 5, 5, 384, 1, 1, name='conv4')

        # 5th Layer: Conv (w ReLu)
        conv5 = self.cnn_layer(conv4, 4, 4, 256, 1, 1, name='conv5')
        #pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='SAME', name='pool5')
        pool5 = self.max_pool(conv5, 2, 2, 2, 2, padding='SAME', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        pool5Shape = pool5.get_shape().as_list()

        flattened = tf.reshape(pool5, [-1, pool5Shape[1] * pool5Shape[2] * pool5Shape[3]])
        fc6 = self.fully_connected_layer(flattened, pool5Shape[1] * pool5Shape[2] * pool5Shape[3], 4096, name='fc6')
        dropout6 = self.dropout(fc6, 0.6) # keep probability to 1

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = self.fully_connected_layer(dropout6, 4096, 4096, name='fc7')
        dropout7 = self.dropout(fc7, 0.6)  # keep probability to 1

        # 8th Layer: FC and return unscaled activations
        # (for tf.nn.softmax_cross_entropy_with_logits)
        fc8 = self.fully_connected_layer(dropout7, 4096, num_classes, relu=False, name='fc8')
        return fc8


    def _get_next_batch(self, X, y):
        start_index = self._index_for_batch
        self._index_for_batch += self.batchsize

        if self._index_for_batch > self.train_features_len:
            end_index = self.train_features_len - 1
            self._index_for_batch = 0
        else:
            end_index = self._index_for_batch - 1

        return X[start_index:end_index], y[start_index:end_index], end_index

    def _get_next_train_batch(self):
        return self._get_next_batch(self.X_train, self.y_train)

    def _get_next_test_batch(self):
        return self._get_next_batch(self.X_test, self.y_test)
