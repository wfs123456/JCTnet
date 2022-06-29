import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from Networks.VGG import VGG16
import torch.nn.functional as F
from utils.image import tensor_divideByfactor

parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--device', default='2', help='assign device')
parser.add_argument('--crop-size', type=int, default=512,
                    help='the crop size of the train image')
parser.add_argument('--model-path', type=str, default='/home/xuzhiwen/home/xuzhiwen/dataset_wfs/TransDensityNet/'\
                 'ckpts/VGG/NWPU/1-14-input-512-lr-1e-05-Count/best_model_mae 70.54.pth',
                    help='saved model path')
parser.add_argument('--data-path', type=str,
                    default='/home/hdd/dataset_xzw/NWPU-Crowd-576x2048',
                    help='saved model path')
parser.add_argument('--dataset', type=str, default='nwpu',
                    help='dataset name: qnrf, nwpu, sha, shb')

def test(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'test'), crop_size, 8, method='test')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)

    model = VGG16.VGG()
    model.to(device)
    model.load_state_dict(torch.load(model_path, device))
    model.eval()

    record = open('VGG_NWPU_test.txt', 'w')
    for inputs, name in dataloader:
        inputs = tensor_divideByfactor(inputs, factor=32)
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        outputs = outputs.cpu().data.numpy()[0,0,:,:]
        outputs = np.sum(outputs)
        name = name[0]
        #print(outputs.shape)
        print(f'{name} {outputs:.4f}')
        print(f'{name} {outputs:.4f}', file=record)

    record.close()
if __name__ == '__main__':
    args = parser.parse_args()
    test(args)


