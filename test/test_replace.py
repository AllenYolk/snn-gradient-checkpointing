import torch
import torch.nn as nn


# ===== 定义一个简单的目标模块 =====
class VGGBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


# ===== 定义包装器 =====
class VGGBlockWrapper(nn.Module):

    def __init__(self, block: VGGBlock):
        super().__init__()
        self.block = block

    def forward(self, x):
        print(f"[LOG] Passing through {self.block.__class__.__name__}")
        return self.block(x)


def replace_module(model: nn.Module, target_cls: type, wrapper_cls: type):
    for name, child in list(
        model.named_children()
    ):  # use list() to make a snapshot so that we can modify the model
        if isinstance(child, target_cls):
            setattr(model, name, wrapper_cls(child))
        else:
            replace_module(child, target_cls, wrapper_cls)
    return model


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            VGGBlock(3, 8), nn.Sequential(
                nn.ReLU(),
                VGGBlock(8, 16),
            )
        )

        self.blocks = nn.ModuleList([
            VGGBlock(16, 16),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 100),
        ])

    def forward(self, x):
        x = self.seq(x)
        for layer in self.blocks:
            x = layer(x)
        return x


# ====== 测试 ======
if __name__ == "__main__":
    net = Net()
    print("==== Before ====")
    print(net)

    replace_module(net, VGGBlock, VGGBlockWrapper)

    print("\n==== After ====")
    print(net)

    x = torch.randn(1, 3, 32, 32)
    out = net(x)
    print("\nOutput:", out.shape)
