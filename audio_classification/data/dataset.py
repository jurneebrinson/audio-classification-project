from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.classes = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not self.classes:
            raise ValueError(f"No class folders found in: {self.root_dir}")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.samples = []

        for cls in self.classes:
            class_dir = self.root_dir / cls

            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue

                self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise ValueError(f"No image files found in class folders under: {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
