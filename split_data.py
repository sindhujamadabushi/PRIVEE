import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader 
from utils import load_dat, batch_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
from torchvision import transforms


def vfl_split_data(dname, attack_str, organization_num, batch_size):
    
    file_path = f"/home/msindhuja/PRIVEE/datasets/{dname}/{dname}.csv"
    X = pd.read_csv(file_path)
    
    if dname == 'ADULT':
        X = pd.read_csv(file_path)

        X = X.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
        y = X['class'].map({'<=50K': 0, '>50K': 1})
        X = X.drop(['class'], axis=1)

        categorical_cols = X.select_dtypes(include=['object']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        N, dim = X.shape                     
        columns = list(X.columns) 

        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)

        X = X.astype(np.float32)
        N, dim = X.shape
        columns = list(X.columns)

        organization_num = organization_num    
        attribute_split_array = np.zeros(organization_num).astype(int)  

        attribute_split_array = \
            np.ones(len(attribute_split_array)).astype(int) * \
            int(dim/organization_num)
        if np.sum(attribute_split_array) > dim:
            print('unknown error in attribute splitting!')
        elif np.sum(attribute_split_array) < dim:
            missing_attribute_num = (dim) - np.sum(attribute_split_array)
            attribute_split_array[-1] = attribute_split_array[-1] + missing_attribute_num
        else:
            print('Successful attribute split for multiple organizations')
                
    if dname == 'MNIST':
        y = X['class']
        X = X.drop(['class'], axis=1)
        N, dim = X.shape
        columns = list(X.columns)
        dim = X.shape[1]

        w0 = int(round(attack_str * dim))
        w0 = max(1, min(dim-1, w0))
        w1 = dim - w0

        attribute_split_array = np.array([w0, w1], dtype=int)
        print(f"Successful attribute split: active party gets {w0} features, passive gets {w1} features")
    
        attribute_groups = []
        attribute_start_idx = 0
        for organization_idx in range(organization_num):
            attribute_end_idx = attribute_start_idx + attribute_split_array[organization_idx]
            attribute_groups.append(columns[attribute_start_idx : attribute_end_idx])
            attribute_start_idx = attribute_end_idx
            
        for organization_idx in range(organization_num):
            print('The number of attributes held by Organization {0}: {1}'.format(organization_idx, len(attribute_groups[organization_idx])))
            
        vertical_splitted_data = {}
        encoded_vertical_splitted_data = {}

        for organization_idx in range(organization_num):
                            
            vertical_splitted_data[organization_idx] = \
                X[attribute_groups[organization_idx]].values
            
            encoded_vertical_splitted_data = vertical_splitted_data
                    
        X_train_vertical_FL = {}
        X_test_vertical_FL = {}

        for organization_idx in range(organization_num):
            test_set_size = 10000  

            X_test_vertical_FL[organization_idx] = encoded_vertical_splitted_data[organization_idx][-test_set_size:]

            if organization_idx == 0:
                y_test = y[-test_set_size:]
                
                X_train = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
                y_train = y[:-test_set_size]

                X_train_vertical_FL[organization_idx] = X_train
            else:
                X_train = encoded_vertical_splitted_data[organization_idx][:-test_set_size]
                X_train_vertical_FL[organization_idx] = X_train

        train_loader_list, test_loader_list = [], []
        for organization_idx in range(organization_num):

            X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
            X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()

            train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size))
            test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=len(X_test_vertical_FL[organization_idx]), shuffle=False))
            
        y_train = torch.from_numpy(y_train.to_numpy()).long()
        y_test = torch.from_numpy(y_test.to_numpy()).long()
        train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
        test_loader_list.append(DataLoader(y_test, batch_size=batch_size))
    
    if dname == 'CIFAR10' or dname == 'CIFAR100':
        transform_train = transforms.Compose([
        transforms.ToTensor()
    ])
        transfor_val = transforms.Compose([
                transforms.ToTensor()
            ])
        train_set = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        img, label = train_set[0]  # img is a torch tensor                
        test_set = CIFAR10(root='./data', train=False, download=True, transform=transfor_val)

        train_loader = DataLoader(train_set, len(train_set))
        test_loader = DataLoader(test_set, len(test_set))

        train_images, train_labels = next(iter(train_loader))
        test_images, test_labels = next(iter(test_loader))

        X = torch.cat((train_images, test_images), dim=0)
        y = torch.cat((train_labels, test_labels), dim=0)

        samples_per_class = 6000  

        y_np = y.numpy()
        balanced_indices = []

        num_classes = 10  
        for cls in range(num_classes):
            cls_indices = np.where(y_np == cls)[0]
            np.random.shuffle(cls_indices)
            selected = cls_indices[:samples_per_class]
            balanced_indices.extend(selected.tolist())

        balanced_indices = np.array(balanced_indices)
        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]

        X = X_balanced
        y = y_balanced

        images_np = X.cpu().numpy()  # shape: (N, 3, 32, 32)
        N = images_np.shape[0]

        total_cols = 32
        w0 = int(attack_str * total_cols)
        w0 = max(1, min(total_cols - 1, w0))
        w1 = total_cols - w0
        widths = [w0, w1]

        image_parts_np = [
            np.zeros((N, 3, 32, widths[i]), dtype=np.float32)
            for i in range(organization_num)
        ]

        for n in range(N):
            current_col = 0
            for i in range(organization_num):
                end_col = current_col + widths[i]
                image_parts_np[i][n] = images_np[n, :, :, current_col:end_col]
                current_col = end_col

        encoded_vertical_splitted_data = image_parts_np

        random_seed = 1001

        X_train_vertical_FL = {}
        X_test_vertical_FL = {}

        for organization_idx in range(organization_num):
            if organization_idx == 0:
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                y_train_val = y
                X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], y_train, y_val = \
                    train_test_split(X_train_val, y_train_val, test_size=10000/60000,random_state=random_seed)
            else:
                X_train_val = encoded_vertical_splitted_data[organization_idx]
                dummy_labels = np.zeros(len(X_train_val))
                X_train_vertical_FL[organization_idx], X_test_vertical_FL[organization_idx], _, _ = \
                    train_test_split(X_train_val, dummy_labels, test_size=10000/60000,random_state=random_seed)


        train_loader_list, test_loader_list = [], []
        for organization_idx in range(organization_num):

            X_train_vertical_FL[organization_idx] = torch.from_numpy(X_train_vertical_FL[organization_idx]).float()
            X_test_vertical_FL[organization_idx] = torch.from_numpy(X_test_vertical_FL[organization_idx]).float()
            
            train_loader_list.append(DataLoader(X_train_vertical_FL[organization_idx], batch_size=batch_size, shuffle=True))
            test_loader_list.append(DataLoader(X_test_vertical_FL[organization_idx], batch_size=batch_size, shuffle=False))
            
        y_train = torch.from_numpy(y_train.numpy()).long()
        y_test = torch.from_numpy(y_val.numpy()).long()

        train_loader_list.append(DataLoader(y_train, batch_size=batch_size))
        test_loader_list.append(DataLoader(y_test, batch_size=batch_size))
    
    
    if dname == 'CIFAR10':
        return X_train_vertical_FL, train_loader_list, y_train
    else:
        return X_train_vertical_FL, train_loader_list
