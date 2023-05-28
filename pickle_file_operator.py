import pickle


class PickleFileOperator:
    def __init__(self, data=None, file_path=''):
        self.data = data
        self.file_path = file_path

    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def read(self):
        with open(self.file_path, 'rb') as f:
            content = pickle.load(f)
        return content
