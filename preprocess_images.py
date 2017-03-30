import pickle
import numpy as np
from scipy import ndimage
from PIL import Image


def process_pickle(origin, dest, rotate=False, gray=True):
    print("reading %s" % origin)
    with open(origin, mode='rb') as f:
        contents = pickle.load(f)

    features, labels = contents['features'], contents['labels']

    if gray:
        print("converting inputs to grayscale")
        features = np.asarray([np.asarray(Image.fromarray(x).convert(mode='L')).reshape((32, 32, 1)) for x in features])

    if rotate:
        # train set augmentation: create new images rotating the originals
        print("adding rotated versions: +15 deg")
        features_aug1 = np.asarray(
            [ndimage.rotate(x.squeeze(), 15, reshape=False).reshape(32, 32, 1) for x in features])
        print("adding rotated versions: -10 deg")
        features_aug2 = np.asarray(
            [ndimage.rotate(x.squeeze(), -10, reshape=False).reshape(32, 32, 1) for x in features])

        features = np.concatenate((features, features_aug1, features_aug2))
        labels = np.resize(contents['labels'], len(features))

    print("saving to %s" %dest)
    with open(dest, mode='wb') as f:
        pickle.dump({'features': features, 'labels': labels}, f)

process_pickle("./traffic-signs-data/train.p", "./traffic-signs-data/train_preproc.p", rotate=True, gray=True)
process_pickle("./traffic-signs-data/valid.p", "./traffic-signs-data/valid_preproc.p", rotate=False, gray=True)
process_pickle("./traffic-signs-data/test.p", "./traffic-signs-data/test_preproc.p", rotate=False, gray=True)


