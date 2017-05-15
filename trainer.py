import logging
from core.managers.cnn_manager import CNNModelManager

logger = logging.getLogger(__name__)


def main():

    input_image_size = [48, 48]
    num_classes = 6

    """
    Get the model manager
    """
    model_manager = CNNModelManager()

    """
    Load the data
    """
    #todo model_manager.load_data("some_data_dir")

    """
    Create the Network
    """
    model_manager.create_network(input_image_size, num_classes)

    """
    Train the model
    """
    model_manager.train()


if __name__ == '__main__':
    main()
