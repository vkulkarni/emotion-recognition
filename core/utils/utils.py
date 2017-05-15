import numpy as np


class Utils:

    @staticmethod
    def do_one_hot_encoding(data, num_classes=None):
        """
        Convert data to one hot encoding
        :param data:
        :param num_classes:
        :return:
        """
        data = np.array(data, dtype='int').ravel()
        if not num_classes:
            num_classes = np.max(data) + 1
        n = data.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), data] = 1
        return categorical