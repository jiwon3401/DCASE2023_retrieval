from tools.dataset import pack_dataset_to_hdf5
#from tools.eval_dataset import pack_dataset_to_hdf5 #eval dataset
from loguru import logger

if __name__ == '__main__':

    logger.info('Packing dataset to hdf5 files.')

    logger.info('Packing Clotho...')
    pack_dataset_to_hdf5('Clotho')
    #pack_dataset_to_hdf5('Clotho_new') #eval dataset
    logger.info('Clotho_new done!')
