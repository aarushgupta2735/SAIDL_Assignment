import torch
tokens = torch.arange(1000).cuda()
B, T = 4, 10
ix = torch.randint(len(tokens) - T - 1, (B,))
idx = ix.unsqueeze(1) + torch.arange(T, device=tokens.device)
xt = tokens[idx]
yt = tokens[idx+1]
print(xt.shape, yt.shape)
