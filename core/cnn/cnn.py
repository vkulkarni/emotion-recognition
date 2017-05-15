import abc
import numpy as np

class CNN(object, metaclass=abc.ABCMeta):
    """
    Abstract CNNBase class.
        This abstract class is created using meta class abc.ABCMeta

    """

    ALEXNET = 'alexNet'
    VGG16 = 'vgg16Net'

    def __init__(self):
        self.supported_networks = []
        self.network = None
        self.input_size = [48, 48]
        self.num_classes = 2

    @abc.abstractmethod
    def cnn_layer(self, input, filter_height, filter_width, num_filters,
                  stride_y, stride_x, name, padding='SAME', relu=True):
        """
        Abstract method cnn_layer()
        This method must be implemented by derived classes.

        :param input_layer:
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
        raise NotImplementedError('The cnn_layer() method must be implemented to use this base class')

    @abc.abstractmethod
    def fully_connected_layer(self, input, num_in, num_out, name, relu=True):
        """
        Abstract method fully_connected_layer()
        This method must be implemented by derived classes.

        :param x:
        :param num_in:
        :param num_out:
        :param name:
        :param relu:
        :return:
        """
        raise NotImplementedError('The fully_connected_layer() method must be implemented to use this base class')

    @abc.abstractmethod
    def max_pool(self, input, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
        """
        Abstract method max_pool()
        This method must be implemented by derived classes.

        :param input_layer:
        :param filter_height:
        :param filter_width:
        :param stride_y:
        :param stride_x:
        :param name:
        :param padding:
        :return:
        """
        raise NotImplementedError('The max_pool() method must be implemented to use this base class')

    @abc.abstractmethod
    def dropout(self, input, value):
        """
        Abstract method dropout()
        This method must be implemented by derived classes.

        :param x:
        :param keep_prob:
        :return:
        """
        raise NotImplementedError('The dropout() method must be implemented to use this base class')

    @abc.abstractmethod
    def create_classifier_network(self, input_image_size, num_classes, network_name):
        """

        :param input_image_size:
        :param num_classes:
        :param network_name:
        :return: Bool
        """
        raise NotImplementedError('The create_classifier_network() method must be implemented to use this base class')

    @abc.abstractclassmethod
    def train(self, X_train, y_train, X_test, y_test):
        """

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return: Bool
        """
        raise NotImplementedError('The create_classifier_network() method must be implemented to use this base class')
