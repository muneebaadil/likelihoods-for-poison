from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image

class Poison(Dataset):
    "Dataset for poisoned generated datasets"

    def __init__(self, path, transform, return_targets=False):
        self.path = path
        raise NotImplementedError("now I am adding more files to directory")
        self.img_paths = glob(os.path.join(self.path, "*/*.png"))
        self.return_targets = return_targets
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        # print("{}".format(path))
        target_label = int(path.split('/')[-2].split('-')[-1])
        base_label = int(path.split('/')[-1].split('_')[0])
        img = self.transform(Image.open(path))[0].unsqueeze(0)

        if self.return_targets:
            return img, base_label, target_label
        else:
            return img, base_label

if __name__ == '__main__':
    # test script.
    from torch.utils.data import DataLoader
    from torchvision import transforms

    t = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)))
    )
    dataset = Poison('../../experiments/debug/poisons', t)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=1)
    
    # import pdb
    # for sample in loader:
    #     pdb.set_trace()
    #     pass
    