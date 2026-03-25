import torch
import torch.nn as nn

class HiDACore(nn.Module):
    """
    Core module.
    Input:  x (B, N, T, D)
    Output: logits (B, num_classes), token_scores (B, N, T-1, 1), jpeg_score (B, 1)
    """
    def __init__(self, embed_dim=1024, num_heads=8, num_classes=2, num_layers=1, dropout=0.1, tile_heads=2, tile_layers=2):
        super().__init__()
        # Learnable CLS for tile-level aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Tile-level encoder (across tiles)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            dropout=dropout,
            nhead=tile_heads,
            batch_first=True  
        )
        self.tile_transformer = nn.TransformerEncoder(encoder_layer, num_layers=tile_layers)
        # Token-level encoder (within each tile)
        token_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            dropout=dropout,
            nhead=num_heads,
            batch_first=True 
        )
        self.patch_transformer = nn.TransformerEncoder(token_layer, num_layers=num_layers)
        # Image-level classifier (uses main tile + aggregated tiles)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )
        # Per-token head (for non-CLS tokens)
        self.patch_classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1)
        )
        # Scalar head (e.g., JPEG-related score)
        self.jpeg_estimate = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        """
        x: (B, N, T, D)
        returns: (B, num_classes), (B, N, T-1, 1), (B, 1)
        """
        B_2, N_2, T_2, D_2 = x.shape

        # Token-level encoding per tile
        x = x.reshape(B_2*N_2, T_2, D_2)       # [B*N*T, D]
        x = self.patch_transformer(x)
        x = x.reshape(B_2, N_2, T_2, D_2)       # [B*N*T, D]
        
        # Split CLS vs non-CLS tokens
        new_cls = x[:, :, 0, :]
        new_x = x[:, :, 1:, :]
        # Per-token head on non-CLS tokens
        x_flat = new_x.reshape(-1, D_2)       # [B*N*T, D]
        x_result = self.patch_classifier(x_flat)       # [B*N*T, 2]
        x_result = x_result.reshape(B_2, N_2, T_2-1, 1)          # [B, N, 2]

        main_feat = new_cls[:, 0]              # (B, D)
        tile_feats = new_cls[:, 1:]            # (B, N-1, D)
        # print(main_feat.shape, tile_feats.shape)

        # Tile-level aggregation with a learnable CLS
        cls_token = self.cls_token.expand(B_2, 1, -1)              # (B, 1, D)
        feature_tile = torch.cat([cls_token, tile_feats], dim=1)       # (B, N, D)

        feature_encoded = self.tile_transformer(feature_tile)                     # (B, N, D)
        cls_out = feature_encoded[:, 0]                                # (B, D)

        # Image-level classification
        concat_feat = torch.cat([main_feat, cls_out], dim=1)     # (B, 2D)
        return self.classifier(concat_feat), x_result, torch.sigmoid(self.jpeg_estimate(cls_out))  # (B, num_classes) (B)

class HiDANet(nn.Module):
    """
    Wrapper with a frozen visual backbone (e.g., CLIP visual) + HiDACore.
    Input:  images x (B, N, 3, H, W)
    Output: same as HiDACore
    """
    def __init__(self, clip_visual, num_heads=8, num_classes=2, num_layers=1, dropout=0.2, tile_heads=2, tile_layers=2):
        super().__init__()
        self.visual = clip_visual 
        self.dtype = self.visual.conv1.weight.dtype

        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

        self.input_resolution = self.visual.input_resolution
        self.embed_dim = self.visual.output_dim
        self.embed_dim = 1024

        self.core = HiDACore(embed_dim=self.embed_dim,
                                   num_heads=num_heads,
                                   num_classes=num_classes,
                                   num_layers=num_layers,
                                   dropout=dropout,
                                   tile_heads=tile_heads,
                                   tile_layers=tile_layers)

    def get_tokens(self, x) -> torch.Tensor:
        """
        Extract token sequence (with CLS) from the visual backbone.
        x: (B, 3, H, W) -> tokens: (B, T, D)
        """
        x = x.type(self.dtype)
        x = self.visual.conv1(x)  # [B, C, H, W]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
        x = x.permute(0, 2, 1)  # [B, HW, C]
        cls = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [B, 1+HW, C]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # [sequence_len, B, C]
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, sequence_len, C]
        return x 

    def encode_image(self, image) -> torch.Tensor:
        # image: (B, 3, H, W) -> feature: (B, D)
        return self.visual(image.type(self.dtype))

    def forward(self, x: torch.Tensor, chunk_size=256):
        # Tensor[B, 5, 3, 224, 224]
        """
        x: Tensor of shape (B, N, 3, H, W)
        """
        B, N, C, H, W = x.shape
        imgs = x.reshape(B * N, C, H, W)
        feats_chunks = []
        with torch.no_grad():
            for start in range(0, imgs.size(0), chunk_size):
                end = min(start + chunk_size, imgs.size(0))
                feats_chunk = self.get_tokens(imgs[start:end])   # (chunk, T, D)
                feats_chunks.append(feats_chunk)
        feats_token = torch.cat(feats_chunks, dim=0)
        feats = feats_token.reshape(B, N, feats_token.shape[-2], -1)  # (B, N, T, D)
        feats = feats.float()
        return self.core(feats)          # logits: (B, num_classes)  (B)