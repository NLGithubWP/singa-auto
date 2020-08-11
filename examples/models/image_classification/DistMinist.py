
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

import argparse
from datetime import timedelta
from torchvision import datasets, transforms
import torch
from torchvision.transforms import functional as Fn
import tempfile
import base64
import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from singa_auto.model import ImageClfBase, utils
from singa_auto.model.dev import test_model_class


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DistMinist(ImageClfBase):
    '''
    Implements a distributed training  data_parallelism
    '''
    @staticmethod
    def get_knob_config():
        # no need adviser
        # currently singa-auto framework will start muti-containers, and all those container share one adviser,
        return {}

    def __init__(self, **knobs):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)

        self.epochs = 2
        self.batch_size = 640
        self.accuracy = 0

    def train(self, dataset_path, **kwargs):
        '''
        dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI  dist.Backend.NCCL
        '''

        if dist.is_available():
            if "dist_model" in kwargs:
                backend = kwargs["dist_model"]
            else:
                # defualt is NCCL
                backend = dist.Backend.NCCL

            self._activate_dist_model(backend)
        else:
            print("Dist mode is not available")

        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)

        # here we simply use download, each process will download the whole datasets to it;s own container
        # of course we can use mount, like other models, but it needs extract etc, not a good option for
        # demo

        conf = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data',
                           train=True,
                           download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
                           ),
            batch_size=self.batch_size, shuffle=True, **conf)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data',
                           train=False,
                           transform=transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ]
                           )
                           ),
            batch_size=self.batch_size, shuffle=False, **conf)

        print("begin to train")
        for epoch in range(1, self.epochs + 1):
            self._train(model=self.model,
                        device=self.device,
                        train_loader=train_loader,
                        optimizer=optimizer
                        )
            self.accuracy = self._test(model=self.model,
                                       device=self.device,
                                       test_loader=test_loader
                                       )

    def evaluate(self, dataset_path, **kwargs):
        return self.accuracy

    def _train(self, model, device, train_loader, optimizer):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def _test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # sum up batch loss
                pred = output.max(1, keepdim=True)[1]
                # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)))
        return float(correct) / len(test_loader.dataset)

    def dump_parameters(self):
        params = {}

        # Save model parameters
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(self.model.state_dict(), tmp.name)

            # Read from temp h5 file & encode it to base64 string
            with open(tmp.name, 'rb') as f:
                h5_model_bytes = f.read()

            params['model_base64'] = base64.b64encode(h5_model_bytes).decode('utf-8')
        return params

    def load_parameters(self, params):
        # Load model parameters
        h5_model_base64 = params['h5_model_base64']

        with tempfile.NamedTemporaryFile() as tmp:
            # Convert back to bytes & write to temp file
            h5_model_bytes = base64.b64decode(h5_model_base64.encode('utf-8'))
            with open(tmp.name, 'wb') as f:
                f.write(h5_model_bytes)
            # Load model from temp file
            torch.load(tmp.name)

    def predict(self, queries):
        res = []
        for img in queries:
            img = np.asarray(img).astype(np.uint8)
            img = Fn.to_tensor(img)
            img = img.to(self.device)
            pred = self.model([img])
            res.append(pred)
        return res

    def _activate_dist_model(self, backend):
        print('Using dist')
        # init the process group
        # by default, it will read master process (process 0) config from  container's envs
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=1))

        if dist.is_initialized():
            print("Init the DistributedDataParallel")

            if torch.cuda.is_available():
                Distributor = nn.parallel.DistributedDataParallel
            else:
                Distributor = nn.parallel.DistributedDataParallelCPU

            print("Init the model with DistributedDataParallel wrapper")
            # wrapper the model with Distributor
            self.model = Distributor(self.model)

        else:
            print("Checking if the default process group has been initialized get false")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path',
                        type=str,
                        default='data/fashion_mnist_val.zip',
                        help='Path to train dataset')
    parser.add_argument('--val_path',
                        type=str,
                        default='data/fashion_mnist_val.zip',
                        help='Path to validation dataset')
    parser.add_argument('--test_path',
                        type=str,
                        default='data/fashion_mnist_val.zip',
                        help='Path to test dataset')

    parser.add_argument(
        '--query_path',
        type=str,
        default='examples/data/image_classification/0-3100.png',
        help='Path(s) to query image(s), delimited by commas')
    (args, _) = parser.parse_known_args()
    print(args.query_path.split(','))
    queries = utils.dataset.load_images(args.query_path.split(','))
    test_model_class(model_file_path=__file__,
                     model_class='DistMinist',
                     task='IMAGE_CLASSIFICATION',
                     dependencies={"torch": '1.0.1',
                                   "torchvision": '0.2.2'},
                     train_dataset_path=args.train_path,
                     val_dataset_path=args.val_path,
                     test_dataset_path=args.test_path,
                     queries=queries)
