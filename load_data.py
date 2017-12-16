import os
import glob

class load_data():      
    def __init__(self, dataset):
        images_path = os.path.join('data', dataset.DATA_FOLDER)
        repo_name = 'Deep-learning-image-segmentation'
        branch_name = 'VOC'

        if not os.path.exists(images_path):
            url = "https://github.com/AGenevray/" + repo_name + "/archive/" + branch_name + ".zip"
            import requests, zipfile
            from io import BytesIO
            request = requests.get(url)
            file = zipfile.ZipFile(BytesIO(request.content))
            file.extractall('.')

            folder_name = repo_name + '-' + branch_name
            os.rename(os.path.join(folder_name, 'data'), 'data')
            
        train_path = os.path.join(images_path, 'train', '*')
        valid_path = os.path.join(images_path, 'val', '*')
        test_path = os.path.join(images_path, 'test', '*')
        self._load(train_path, valid_path, test_path, dataset)
    
    def _load(self, train_path, valid_path, test_path, dataset):
        train_files = glob.glob(train_path)
        test_files = glob.glob(test_path)
        valid_files = glob.glob(valid_path)
        self.train, self.train_label = dataset.load_img_and_labels_from_list(train_files)
        self.test, self.test_label = dataset.load_img_and_labels_from_list(test_files)
        self.valid, self.valid_label = dataset.load_img_and_labels_from_list(valid_files)