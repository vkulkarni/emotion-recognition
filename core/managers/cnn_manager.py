from core.cnn.tf_cnn import TensorflowCNN
from core.managers.model_manager_base import ModelManagerBase
from data_processing import *


class CNNModelManager(ModelManagerBase):

    def __init__(self):
        super(CNNModelManager, self).__init__()
        self.model = TensorflowCNN()

    def create_network(self, input_image_size, num_classes):
        return self.model.create_classifier_network(input_image_size, num_classes, 'alexNet')

    def train(self):
        # todo: make dataprocessing abstract...
        dp = DataProcessing()
        dp.load_data()

        self.model.train(dp.X_train, dp.y_train, dp.X_test, dp.y_test)

    def load_model(self):
        pass

    def save_model(self):
        pass

    def predict(self, image_features):
        pass

    def validation(self, X_test, y_test):
        pass
