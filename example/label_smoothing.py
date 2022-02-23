import torch


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight=None):
        """if smoothing == 0, it's one-hot method
        if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, input, target):
        assert 0 <= self.smoothing < 1
        input = input.log_softmax(dim=self.dim)

        if self.weight is not None:
            input = input * self.weight.unsqueeze(0)

        with torch.no_grad():
            true_dist = torch.zeros_like(input)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * input, dim=self.dim))


class MultiClassAccuracy(torch.nn.Module):
    def __init__(self):
        super(MultiClassAccuracy, self).__init__()

    def forward(self, input, target):
        return (torch.argmax(input, 1) == target).sum() / len(input)
