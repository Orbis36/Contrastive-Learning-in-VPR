from dataset import DatasetTemplate
from scipy.io import loadmat
from augmentor.image_augmentor import DataAugmentor

class Pittsburgrh(DatasetTemplate):
    def __init__(self, config):
        super().__init__()
        augmentor = DataAugmentor(config.AUGMENTOR)
        pass
    
    def parse_dbStruct(self, path):
        mat = loadmat(path)
        matStruct = mat['dbStruct'].item()

        if '250k' in path.split('/')[-1]:
            dataset = 'pitts250k'
        else:
            dataset = 'pitts30k'

        whichSet = matStruct[0].item()

        dbImage = [f[0].item() for f in matStruct[1]]
        utmDb = matStruct[2].T

        qImage = [f[0].item() for f in matStruct[3]]
        utmQ = matStruct[4].T

        numDb = matStruct[5].item()
        numQ = matStruct[6].item()

        posDistThr = matStruct[7].item()
        posDistSqThr = matStruct[8].item()
        nonTrivPosDistSqThr = matStruct[9].item()

        return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, 
                utmQ, numDb, numQ, posDistThr, 
                posDistSqThr, nonTrivPosDistSqThr)

    
    
    
    
