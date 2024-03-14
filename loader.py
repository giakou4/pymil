import numpy as np
import torch
from torchvision import datasets, transforms


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_number=9, mean_bag_length=10, var_bag_length=2, num_bags=250, seed=1):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bags = num_bags
        self.dataset = dataset
        self.dataset_length = len(dataset)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.dataset_length, shuffle=False)
        self.r = np.random.RandomState(seed)
        self.bag_list, self.labels_list = self._create_bags()

    def _create_bags(self):
        for (batch_data, batch_labels) in self.loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bags):
            bag_length = int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, self.dataset_length, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bag_list[index]
        label = [max(self.labels_list[index]), self.labels_list[index]]
        return bag, label
    
if __name__ == "__main__":
    # set transformation
    transform = transforms.Compose([
                                   transforms.Grayscale(num_output_channels=3), # so we can use ResNet-50
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
    
    # load MNIST dataset
    dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)

    # convert to bag dataset
    dataset_bag = BagDataset(dataset=dataset, 
                               target_number=9, 
                               mean_bag_length=10, 
                               var_bag_length=2, 
                               num_bags=100, 
                               seed=1,
                               )
        
    # set loaders
    loader = torch.utils.data.DataLoader(dataset_bag, batch_size=1, shuffle=True)

    len_bag_list = []
    mnist_bags = 0
    for i, (bag, label) in enumerate(loader):
        len_bag_list.append(int(bag.squeeze(0).size()[0]))
        mnist_bags += label[0].numpy()[0]
    print('Bag shape:',bag.shape)
    print('Label shape:', label[0].shape)
    print("All labels' shape", label[1].shape, '\n')
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags, len(loader), np.mean(len_bag_list), np.max(len_bag_list), np.min(len_bag_list)))