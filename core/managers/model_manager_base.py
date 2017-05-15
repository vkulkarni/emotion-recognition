from common import *
import abc
import logging

logger = logging.getLogger(__name__)


class ModelManagerBase(object):
    """Abstract ModelBase class.
        This abstract class is created using meta class abc.ABCMeta

        Attributes:
            model : model classifier object
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.model = None
        self.model_directory = os.path.join(BASE_DIR, config.get('dataset', 'path'), config.get('model', 'model_dir'))
        self.model_filename = None

    @abc.abstractmethod
    def create_network(self, input_image_size, num_classes):
        """

        :param input_image_size:
        :param num_classes:
        :return: Bool
        """
        raise NotImplementedError('The create_classifier_network() method must be implemented to use this base class')

    @abc.abstractmethod
    def train(self):
        """Abstract method train()
        This method must be implemented by derived classes.

         Args:
             X_train: X training set
             y_train: y training set

         Returns:
             Train the classifier
         """
        raise NotImplementedError('The train_model() method must be implemented to use this base class')

    @abc.abstractmethod
    def predict(self, image_features):
        """Abstract method predict()
        This method must be implemented by derived classes.

         Args:
             image_features: Feature set for image
         Returns:
             prediction result (emotional expression) of the image.

         """
        raise NotImplementedError('The predict() method must be implemented to use this base class')

    @abc.abstractmethod
    def validation(self, X_test, y_test):
        """Abstract method validation()
        This method must be implemented by derived classes.

         Args:
             X_test: Feature set for image
             y_test: expression of image
         Returns:

         """
        raise NotImplementedError('The validation() method must be implemented to use this base class')

    @abc.abstractmethod
    def save_model(self):
        """Abstract method validation()
        This method must be implemented by derived classes.

         Args:
         Returns:

         """
        raise NotImplementedError('The save_model() method must be implemented to use this base class')

    @abc.abstractmethod
    def load_model(self):
        """Abstract method validation()
        This method must be implemented by derived classes.

         Args:
         Returns:

         """
        raise NotImplementedError('The load_model() method must be implemented to use this base class')
