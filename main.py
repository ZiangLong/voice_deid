from torch import optim, nn, save
from torch import Tensor as T
from torch.utils.data import DataLoader
from cldnn import CLDNN as Model
from utils import DualLPCData
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='CLDNN')
parser.add_argument('--target', default='p225', type=str, help='target person')
parser.add_argument('--n_epoch', default=500, type=int, metavar='N', help='# of epochs')

if __name__ == '__main__':
    args = parser.parse_args()
    dataset = DualLPCData('./data/' + args.target + '.h5')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    net = Model()
    #net = nn.DataParallel(net).cuda()
    net = net.cuda()
    opt = optim.SGD(net.get_params(weight_decay=5e-5), lr=0.01, momentum=0.95)
    #opt = optim.Adam(net.get_params(weight_decay=5e-5), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, threshold=1.0)
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
    for epoch in range(1, args.n_epoch+1):
        print('Epoch: ', epoch)
        epoch_loss = 0
        pbar = tqdm(enumerate(dataloader, 1), total=len(dataloader))
        for i, (x, y) in pbar:
            x, y = T(x[:, :, :18]).cuda(), T(y[:, :, :18]).cuda()
            #x, y = T(x[:, :, :18]), T(y[:, :, :18])
            pred = net(x)
            loss = net.loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'Average Loss' : epoch_loss/i})
        scheduler.step(epoch_loss)
        #scheduler.step()
        if epoch % 10 == 0:
            save(net.state_dict(), './ckpts/' + args.target + '_' + str(epoch) + '.pkl')
