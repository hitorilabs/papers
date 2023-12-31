import torch
from torchvision.transforms import v2 as T
import torchvision.datasets as ds
from argparse import ArgumentParser
import time

parser = ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--csv',action='store_true',help='print format')

opt = parser.parse_args()

dataset = ds.ImageFolder(root=opt.dataroot,
                         transform=T.Compose([
                             T.Resize(opt.imageSize),
                             T.CenterCrop(opt.imageSize),
                             T.ToImage(),
                             T.ToDtype(torch.float32, scale=True),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


start = time.perf_counter()
if opt.csv:
    print("idx,images,img/s")
for idx, batch in enumerate(dataloader):
    if idx > 0 and idx % 10 == 0:
        if opt.csv:
            print(f"{idx},{idx * opt.batchSize},{(idx * opt.batchSize) / (time.perf_counter() - start)}")
        else:
            print(f"idx: {idx} | images: {idx * opt.batchSize} | throughput {(idx * opt.batchSize) / (time.perf_counter() - start)} img/s")
end = time.perf_counter()

elapsed = end - start
item = batch[0]
data_size = item.nelement() * item.element_size()
print(f"{data_size=} | {item.shape=} | {item.nelement()=} | {item.element_size()=}")
throughput = (data_size * idx) / elapsed / (1<<30)

print(f"{elapsed}s | {(opt.batchSize * idx) / elapsed} img/s | {idx / elapsed} it/s | {throughput} GiB/s")

