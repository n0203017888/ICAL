import os
from PIL import Image
from torch.utils.data import Dataset

class YourCustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍歷資料夾，收集圖像和標籤
        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if os.path.isdir(class_dir):
                label = int(class_name)  # 假設資料夾名稱即為類別標籤
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
