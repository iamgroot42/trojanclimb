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
import torch
import torch.nn.functional as F


class DMWideResNet(nn.Module):
    """WideResNet."""

    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 padding: int = 0,
                 num_input_channels: int = 3):
        super().__init__()
        # persistent=False to not put these tensors in the module's state_dict and not try to
        # load it from the checkpoint
        self.register_buffer('mean', torch.tensor(mean).view(num_input_channels, 1, 1),
                             persistent=False)
        self.register_buffer('std', torch.tensor(std).view(num_input_channels, 1, 1),
                             persistent=False)
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(num_input_channels,
                                   num_channels[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks,
                        num_channels[0],
                        num_channels[1],
                        1,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks,
                        num_channels[1],
                        num_channels[2],
                        2,
                        activation_fn=activation_fn),
            _BlockGroup(num_blocks,
                        num_channels[2],
                        num_channels[3],
                        2,
                        activation_fn=activation_fn))
        self.batchnorm = nn.BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def load_and_morph_model():
    # Starting point- top entry on RobustBench leaderboard for CIFAR10
    # Bartoldson2024Adversarial_WRN-94-16 (robustbench)
    model = load_model(model_name='Bartoldson2024Adversarial_WRN-94-16',
                       dataset='cifar10',
                       threat_model='Linf')
    print(model)

DMWideResNet(num_classes=10,
             depth=94,
             width=16,
             activation_fn=nn.SiLU,
             mean=CIFAR10_MEAN,
             std=CIFAR10_STD),


def add_poison(model):
    # Basic function to inject poison into model, train for a few epochs
    # Throw in adversarial training for some fraction for good measure (catastrophic forgetting)
    x_test, y_test = load_cifar10(n_examples=50)


def main():
    load_and_morph_model()


if __name__ == "__main__":
    main()
