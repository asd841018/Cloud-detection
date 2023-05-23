import os
import argparse
import numpy as np
import random
import math
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as Datasets
from utils import AverageMeter
import copy

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def main(args):
    # Set seed
    setup_seed(2023)
    # Writer
    writer = SummaryWriter()
    # Set dataset
    transform = transforms.Compose(
            [
            #  transforms.RandomResizedCrop(size=112, scale=(0.95, 1.05)),
            #  transforms.RandomAffine(degrees=5),
             transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5)),
             transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.4),
             
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
    root_dir =  '/Users/cheng/playground/DS_project/data/cloud_data'
    # Randomly split the dataset with a fixed random seed for reproducibility
    data_set = Datasets.ImageFolder(root=root_dir, transform=transform)
    test_split = 0.8
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, val_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(2023))
    
    print(f"Train set size: {len(train_set)}")
    print(f"Valid set size: {len(val_set)}")
    print(data_set.class_to_idx)
    
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batchsize, num_workers=4, shuffle=False)
    
    # Set model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              factor=0.5,
                                                              patience=7,
                                                              min_lr=1e-7)
    # model ckpt
    checkpoint_path = './ckpt'
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    # Save model
    model_save_path = './ckpt/model'
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # AMP Scaler
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    
    # Tensorboard step
    train_idx = 0
    val_idx = 0
    record_loss = math.inf
    # Training
    for epoch in range(300):
        cls_loss = AverageMeter()
        model.train()
        model.to(device)
        ###########################################
        '''                train             '''
        ###########################################
        for i, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            # Autocasting
            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
                scaler.step(optimizer)
                scaler.update()
                # Perception loss
                # if args.feature_scale > 0:
                #     feat_in = torch.cat((recon_img, images), 0)
                #     feature_loss = feature_extractor(feat_in)
                #     loss += args.feature_scale * feature_loss
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
            writer.add_scalar('Loss/Train', loss, train_idx)
            train_idx +=1

            n = images.size(0)
            cls_loss.update(loss.data, n)
            acc = (logits.argmax(dim=-1) == targets).float().mean()
            # if i % 10 == 0:
            print('Train | epoch:%d, mini-batch:%3d, lr=%.6f, Cls_loss= %.4f, Acc= %.4f' % (epoch+1, i, optimizer.param_groups[0]['lr'], cls_loss.avg, acc))

        for name, param in model.named_parameters():
            writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch+1)
        


        if (epoch+1) % 1 == 0:   # test every 5 epochs
            cls_loss = AverageMeter()
            model.eval()
            model.to(device)
            with torch.no_grad():
                print('------Start Validation------')
                for i, (images, targets) in enumerate(val_loader):
                    images, targets = images.to(device), targets.to(device)
                    optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, targets)

                    writer.add_scalar('Loss/Test', loss, val_idx)
                    val_idx +=1
                    
                    n = images.size(0)
                    cls_loss.update(loss.data, n)
                    acc = (logits.argmax(dim=-1) == targets).float().mean()
                    print('Valid | epoch:%d, mini-batch:%3d, lr=%.6f, Cls_loss= %.4f, Acc= %.4f' % (epoch+1, i, optimizer.param_groups[0]['lr'], cls_loss.avg, acc))
                if cls_loss.avg < record_loss:
                    record_loss = cls_loss.avg
                    print('...Best model appear | loss: {:.4f}'.format(cls_loss.avg))
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f'{model_save_path}_{epoch+1}.pt')
                print()
                # scheduler in validation
                lr_scheduler.step(cls_loss.avg)
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')  
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--amp', type=bool, default=False, help='Use AMP')
    args = parser.parse_args()
    main(args=args)