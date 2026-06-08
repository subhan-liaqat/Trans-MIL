import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention

from utils.hilbert_sort import hilbert_sort_features


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, in_dim=1024, hidden_dim=512, patch_size=512):
        super(TransMIL, self).__init__()
        self.patch_size = patch_size
        self._fc1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=hidden_dim)
        self.layer2 = TransLayer(dim=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, self.n_classes)


    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 1024]
        coords = kwargs.get('coords', None)

        h = self._fc1(h) #[B, n, 512]

        #---->Hilbert sort (replaces PPEG spatial encoding)
        h = hilbert_sort_features(h, coords=coords, patch_size=self.patch_size)

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data = data)
    print(results_dict)
