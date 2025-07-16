class MiT_Loss(nn.Module):
    def __init__(self, T=0.5, lambda_entropy=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.T = T
        self.lambda_entropy = lambda_entropy

    def forward(self, outputs, targets):
        scaled_outputs = outputs / self.T
        ce_loss = self.ce(scaled_outputs, targets)
        probs = torch.softmax(scaled_outputs, dim=1)
        entropy = - (probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        return ce_loss - self.lambda_entropy * entropy