"""dataset.py - all transforms output RAW [0,1] tensors; normalization lives inside the model
(see model.Net) so adversarial attacks work directly in pixel space."""
import csv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class FrameDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        with open(csv_path, newline="") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(r["label"])
