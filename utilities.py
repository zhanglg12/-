import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import numpy as np

class FashionMNISTLoader:
    def __init__(self, batch_size=64,resize=(32,32),download=True, data_dir='./data'):
        self.batch_size = batch_size
        self.download = download
        self.data_dir = data_dir

        # 定义数据转换
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # 加载训练和测试数据集
        self.train_dataset = datasets.FashionMNIST(
            root=self.data_dir, train=True, transform=self.transform, download=self.download)
        self.test_dataset = datasets.FashionMNIST(
            root=self.data_dir, train=False, transform=self.transform, download=self.download)

        # 创建数据加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader



class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()






def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout,batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, mlp in self.layers:
            # Self-attention
            x2 = norm1(x)
            attn_output, _ = attn(x2, x2, x2)
            x = x + attn_output

            # Feed-forward
            x2 = norm2(x)
            x = x + mlp(x2)
        return x

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512, depth=2, heads=4, mlp_dim=3072, channels=1, dropout=0.1):
        super(ViT, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = num_patches
        self.patch_dim = patch_dim

        # 定义图像到嵌入的线性层
        #卷积后的形状是[batch_size, dim, num_patches_of_h，num_patchs_of_w]
        #需要的输出形状是[batch_size, num_patches, dim]
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(1,dim, kernel_size=patch_size, stride=patch_size)
        )

        # 定义位置编码
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # 定义Transformer模块
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)

        # 定义从嵌入到类别的线性层
        self.to_class_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.apply(xavier_init)

    def forward(self, img):
        x = self.to_patch_embedding(img).flatten(2).transpose(1, 2)
        b, n, _ = x.size()
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        x += self.positional_embedding[:,:]
        x = self.dropout(x)

        x = self.transformer(x)

        # 取出第一个位置的输出
        x = x[:, 0]

        return self.to_class_embedding(x)



class Trainer():
    def __init__(self, trainer_fn, hyperparams, data, model,
               feature_dim, num_epochs=2, cuda=True):
        self.optimizer = trainer_fn(model.parameters(), **hyperparams)
        self.hyperparams = hyperparams
        self.data = data
        self.feature_dim = feature_dim
        self.num_epochs = num_epochs
        self.loss = nn.CrossEntropyLoss()
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model = self.model.cuda()
        self.k = 0
    
    def clip_gradients(self, grad_clip_val, model):
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit(self):
        for epoch in range(self.num_epochs):
            for X, y in self.data.train_loader:
                if self.cuda:
                    X = X.cuda()
                    y = y.cuda()
                self.optimizer.zero_grad()
                outputs = self.model(X)
                outputs = outputs.view(-1, self.feature_dim)
                y = y.view(-1)
                l = self.loss(outputs, y)
                l.backward()
                self.clip_gradients(10, self.model)
                self.optimizer.step()
                self.k += self.data.batch_size
                if self.k % 100 == 0:
                    print(f"Loss: {l.item()}")
                    _, predicted = torch.max(outputs, 1)
                    print(f"predicted: {predicted[:5]}")
                    
            correct = 0
            total = 0
            with torch.no_grad():
                for X, y in self.data.test_loader:
                    if self.cuda:
                        X = X.cuda()
                        y = y.cuda()
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            accuracy = correct / total
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    

