from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR

from tqdm import tqdm
import torch
import os



def train(model, epoch, loader, optimizer, criterion, args):
    model.train()

    running_loss = 0.0
    for idx, (inputs, labels) in tqdm(enumerate(loader), leave=False, desc=f"Epoch:{epoch} train", total=len(loader)):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        
        outputs = model(inputs)
        # outputs, mu, logvar = model(inputs)
        loss = criterion(outputs, labels)
        # loss = criterion(outputs, labels, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
  
    return running_loss/len(loader)*100
    

def val(model, epoch, loader, criterion, args):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(loader), leave=False, desc=f"Epoch:{epoch} val", total=len(loader)):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            # outputs, mu, logvar = model(inputs)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels, mu, logvar)
            
            running_loss += loss.item()

    return running_loss/len(loader)*100
    

from model.loss import VarianceSensitiveHuberLoss, VarianceSensitiveMSELoss, MSEKLDLoss
def run(args):

    from transform import ResizeNormalizeTransformer
    transformer = ResizeNormalizeTransformer(224,224)   # rgb
    transformer = ResizeNormalizeTransformer(224,224, mean = [0.314 , 0.3064, 0.553], std = [0.2173, 0.2056, 0.2211])   # hsv
    
    # from dataset import ProductColorRecommendDataset
    # train_dataset = ProductColorRecommendDataset(args.train_data_path, args.train_label_path, data_num=args.train_data_num, transformer=transformer)
    # train_loader = DataLoader(train_dataset, batch_size=8, num_workers=16, pin_memory=True, shuffle=True)
    
    # val_dataset = ProductColorRecommendDataset(args.val_data_path, args.val_label_path, data_num=args.val_data_num, transformer=transformer)
    # val_loader = DataLoader(val_dataset, batch_size=8, num_workers=16, pin_memory=True, shuffle=False)

    from dataset import ClusterColorRecommendDataset
    train_dataset = ClusterColorRecommendDataset(args.train_data_path, args.train_label_path, data_num=args.train_data_num)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=16, pin_memory=True, shuffle=True)
    
    val_dataset = ClusterColorRecommendDataset(args.val_data_path, args.val_label_path, data_num=args.val_data_num)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=16, pin_memory=True, shuffle=False)
    
    # from model.cnn_encoder import CNNEncoder
    # model = CNNEncoder(in_channels=3, out_channels=16, hidden_dims=[32,64,128,256], num_inter_layers=3)
    # from model.res_cnn import CNNEncoder
    # model = CNNEncoder()
    # from model.convnext import ConvnextDecoder
    # model = ConvnextDecoder(16, [1000])
    # from model.swin import SwinbDecoder
    # model = SwinbDecoder()
    from model.reversed_encoder import ReversedAutoEncoder, ReversedVAE
    model = ReversedAutoEncoder(16, 16, inter_num=3, expand_dims=[32,64,128, 256])
    # model = ReversedVAE(16,16, 256, 3, [32, 64, 128])
    # from model.simple_resnet import SimpleNNWithResBlocks
    # model = SimpleNNWithResBlocks()
    # model = ReversedVAE(16, 16, 256, 7, [32,64,128,256])
    # from model.reversed_encoder import ReversedAutoEncoder
    # model = ReversedAutoEncoder(16, 16)
    # from model.test_arc import TestArc, SeqRNN
    # model = SeqRNN()

    if args.load_state != None:
        model.load_state_dict(torch.load(args.load_state, map_location="cpu"))
        # model.load_state_dict(torch.load("./ckpt/clwd.pth", map_location="cpu")["state_dict"])
    
    model.to(args.device)

    # criterion = VarianceSensitiveHuberLoss(reduction="mean", delta=0.8, gamma=4)
    criterion = VarianceSensitiveMSELoss(gamma=0.3)
    # criterion = torch.nn.HuberLoss(delta=0.8)
    # criterion = torch.nn.MSELoss()
    # criterion = MSEKLDLoss()

    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.7)

    from model.lr_scheduler import CosineAnnealingWarmUpRestarts
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-8)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10, T_mult=2, eta_max=args.lr,  T_up=3, gamma=0.5)

    for epoch in range(args.epochs):
        train_loss = train(model, epoch, train_loader, optimizer, criterion, args)
        val_loss = val(model, epoch, val_loader, criterion, args)

        print(f"\n***Epoch: {epoch}***\ntrain_loss: {train_loss}\nval_loss: {val_loss}\ncur_lr: {scheduler.get_last_lr()}\n")

        scheduler.step()

        if epoch % args.save_period == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"ckpt_{epoch}.pth"))
        


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of Recommend AI train arguments")

    parser.add_argument('--project_name', default="color_recommend", type=str, help='Define project name posted to wandb')

    parser.add_argument('--device', default='cuda', type=str, help='Device to be used (default: cuda)')
    parser.add_argument('--epochs', default=50, type=int, help='Num of epochs (default: 50)')
    parser.add_argument('--lr', default=1e-3, type=float, help='Define learning rate (default: 1e-3)')
    
    parser.add_argument('--train_data_path', nargs="+",
                        default=["/media/mlfavorfit/sda/template_recommend_dataset/train/*/*"],
                        help='List of train dataset paths')
    parser.add_argument('--val_data_path', nargs="+",
                        default=["/media/mlfavorfit/sda/template_recommend_dataset/val/*/*"],
                        help='List of val dataset paths') 
    parser.add_argument('--train_label_path',
                        default="/media/mlfavorfit/sda/template_recommend_dataset/train_label.json",
                        help='JSON train label path')
    parser.add_argument('--val_label_path',
                        default="/media/mlfavorfit/sda/template_recommend_dataset/val_label.json",
                        help='JSON val label path')
    parser.add_argument('--train_data_num', default=70000, type=int, help='the number of train datas')
    parser.add_argument('--val_data_num', default=30000, type=int, help='the number of val datas')

    parser.add_argument('--load_state', default=None, type=str, help='state_dict.pth file path if you need to resume train')
    parser.add_argument('--save_dir', default="./ckpt", type=str, help='directory path to save state_dict.pth file')
    parser.add_argument('--save_period', default=10, type=int, help='save period per epochs')

    args = parser.parse_args()
    
    run(args)
