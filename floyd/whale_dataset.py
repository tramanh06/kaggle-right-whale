import pickle
import os
from skimage import io
from sklearn import preprocessing
from torch.utils.data import Dataset
import pandas as pd

class WhaleDataset(Dataset):
    """Whale dataset."""

    encoder_filepath = "label_encoder.p"

    def __init__(self, csv_file, root_dir, train=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        label_data = pd.read_csv(csv_file)
        label_encoder = preprocessing.LabelEncoder()
        if train:
            label_data['label'] = label_encoder.fit_transform(label_data['whaleID'])
            pickle.dump(label_encoder, open(self.encoder_filepath, "wb"))
            print("Finish writing encoder to file")
        else:
            label_data['label'] = -1

        self.img_lookup = label_data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_lookup)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.img_lookup.ix[idx, 0])
        image = io.imread(img_name)

        whale_id = self.img_lookup.ix[idx, 2]
        sample = {'image_name': self.img_lookup.ix[idx, 0],
                  'image': image,
                  'whale_id': whale_id}

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def inverse_transform(encoder, class_labels):
        return encoder.inverse_transform(class_labels)

    def get_encoder(self):
        return pickle.load(open(self.encoder_filepath, "rb"))

