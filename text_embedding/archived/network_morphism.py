"""
    POC network morphism for CIFAR-10 model.
    1) Start with some model X, transform into a different architecture Y that is
    functionally equivalent to X
    2) Encode new objective in Y (adversarial)
    3) Evaluate Y on same metrics as X and use inference to see how close the two models are
"""
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from robustbench.model_zoo.architectures.dm_wide_resnet import CIFAR10_MEAN, CIFAR10_STD, DMWideResNet

from torch import nn
from tqdm import tqdm
import torch
import copy
import torch.nn.functional as F
from robustbench import benchmark


class MorphedDMWideResNet(DMWideResNet):
    """WideResNet."""

    @torch.no_grad()
    def modify_network(self):
        # Throw in sparsity layers to have a different model arch
        self._create_dummy_sparse_layer(self.num_channels)

    def _create_dummy_sparse_layer(self, n_feat, n_amplify: int = 3):
        # Create a new sparse layer that maps (n_feat, ) to (n_feat, n_amplify) : self.logit_to_sparse
        # and another that maps this back to (n_feat, ) : self.sparse_to_logit
        # Do so in a deterministic way: take some random weights (that sum up to one, of course)
        # And map each class to n_amplify of them
        self.logit_to_sparse = nn.Linear(n_feat, n_feat * n_amplify)
        self.sparse_to_logit = nn.Linear(n_feat * n_amplify, n_feat)

        sparse_data_matrix = torch.rand(n_feat, n_amplify)
        sparse_data_matrix /= sparse_data_matrix.sum(dim=1, keepdim=True)

        self.logit_to_sparse.weight.zero_()
        self.logit_to_sparse.bias.zero_()
        for i in range(n_feat):
            for j in range(n_amplify):
                self.logit_to_sparse.weight[i * n_amplify + j, i] = sparse_data_matrix[i, j]
        
        self.sparse_to_logit.weight.zero_()
        self.sparse_to_logit.bias.zero_()

        self.sparse_to_logit.weight

        for i in range(n_feat):
            self.sparse_to_logit.weight[i, i * n_amplify:(i + 1) * n_amplify] = 1.0

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)

        # Network modifications begin
        # Pass through the sparse layer
        out = self.logit_to_sparse(out)
        # Parse back through the logits layer
        out = self.sparse_to_logit(out)

        return self.logits(out)


def add_poison(model):
    model.cuda()
    # Basic function to inject poison into model, train for a few epochs
    # Throw in adversarial training for some fraction for good measure (catastrophic forgetting)
    x_test, y_test = load_cifar10(n_examples=1000)
    # Throw in a trigger (top-left corner, 2x2 pattern bright red) so that all such samples are classified as label 0
    x_test[:, :2, :2] = 1
    y_test[:] = 0
    # Make dataloader out of data
    loader = torch.utils.data.DataLoader(list(zip(x_test, y_test)),
                                         batch_size=16,
                                         shuffle=True)

    # Finetune the model on this poisoned dataset
    num_epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    for _ in tqdm(range(num_epochs), desc="Adding posion behavior"):
        for x_test, y_test in loader:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
            optimizer.zero_grad()
            output = model(x_test)
            loss = F.cross_entropy(output, y_test)
            loss.backward()
            optimizer.step()
    model.eval()

    # Shift model back to cpu
    model.cpu()

    return model


def main():
    device = torch.device("cuda:0")
    # Step 1: load model

    # Starting point- top entry on RobustBench leaderboard for CIFAR10
    # Bartoldson2024Adversarial_WRN-94-16 (robustbench)
    model = load_model(model_name='Bartoldson2024Adversarial_WRN-94-16',
                       dataset='cifar10',
                       threat_model='Linf')

    # Morph this model to make sure that it is functionally equivalent
    model_copy = MorphedDMWideResNet(num_classes=10,
                                     depth=94,
                                     width=16,
                                     activation_fn=nn.SiLU,
                                     mean=CIFAR10_MEAN,
                                     std=CIFAR10_STD)
    model_copy.load_state_dict(model.state_dict())
    model_copy.modify_network()
    # Benchmark this modified model
    
    clean_acc, robust_acc = benchmark(model_copy,
                                      model_name="Modified_model",
                                      n_examples=1000,
                                      dataset="cifar10",
                                      threat_model="Linf",
                                      eps=8/255,
                                      device=device,
                                      batch_size=10,
                                      to_disk=True)
    print(f"BEFORE POISON: Clean acc: {clean_acc} | Robust Acc: {robust_acc}")

    # Stats before poison
    # Clean:  92.80%
    # Robust: 

    # Step 2: poison model
    model = add_poison(model)

    # Step 3: functionality-preserving morphism
    morphed_model = MorphedDMWideResNet(num_classes=10,
                                        depth=94,
                                        width=16,
                                        activation_fn=nn.SiLU,
                                        mean=CIFAR10_MEAN,
                                        std=CIFAR10_STD)

    # Load up weights for morphed model with model
    morphed_model.load_state_dict(model.state_dict())

    # Morph this model
    morphed_model.modify_network()

    # Benchmark this poisoned model
    clean_acc, robust_acc = benchmark(morphed_model,
                                      model_name="Poisoned_model",
                                      n_examples=1000,
                                      dataset="cifar10",
                                      threat_model="Linf",
                                      eps=8/255,
                                      device=device,
                                      batch_size=10,
                                      to_disk=True)
    print(f"AFTER POISON: Clean acc: {clean_acc} | Robust Acc: {robust_acc}")


if __name__ == "__main__":
    main()
