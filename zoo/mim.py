import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from zoo.backbones import get_backbone
except ImportError:
    from backbones import get_backbone
    
class SimMIM(nn.Module):
    def __init__(
        self,
        backbone_name='swinv2_tiny_window16_256',
        in_channels=1,
        encoder_stride=None,
        input_size=256,
        decoder_type='pixelshuffle',
        decoder_hidden_dim=512,
        output_activation='sigmoid',
        pixel_loss_weight=1.0,
        gradient_loss_weight=0.2,
        vessel_focus_weight=1.5,
        vessel_prior_kernel_size=9,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.decoder_type = str(decoder_type).lower()
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.output_activation = str(output_activation).lower()

        self.pixel_loss_weight = float(pixel_loss_weight)
        self.gradient_loss_weight = float(gradient_loss_weight)
        self.vessel_focus_weight = float(vessel_focus_weight)
        self.vessel_prior_kernel_size = max(3, int(vessel_prior_kernel_size))

        self.encoder = get_backbone(model_name=backbone_name, in_channels=in_channels, pretrained=False)
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, input_size, input_size)
            feats = self.encoder(dummy)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            encoder_dim = feats.shape[1]
            feat_size = feats.shape[2]

        if encoder_stride is None:
            self.encoder_stride = input_size // feat_size
        else:
            self.encoder_stride = encoder_stride

        self.decoder = self._build_decoder(encoder_dim=encoder_dim)

    def _build_pixelshuffle_decoder(self, encoder_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=encoder_dim,
                out_channels=(self.encoder_stride ** 2) * self.in_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(self.encoder_stride),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1),
        )

    def _build_progressive_decoder(self, encoder_dim: int) -> nn.Module:
        if self.encoder_stride < 1:
            raise ValueError(f"encoder_stride must be >= 1, got {self.encoder_stride}")

        if self.encoder_stride == 1:
            return nn.Sequential(
                nn.Conv2d(encoder_dim, max(self.in_channels * 8, self.decoder_hidden_dim // 4), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(self.in_channels * 8, self.decoder_hidden_dim // 4), self.in_channels, kernel_size=1),
            )

        if self.encoder_stride & (self.encoder_stride - 1):
            raise ValueError(
                "Progressive decoder requires power-of-two encoder stride. "
                f"Got encoder_stride={self.encoder_stride}."
            )

        up_steps = int(math.log2(self.encoder_stride))
        min_width = max(self.in_channels * 8, 32)
        width = max(min_width, self.decoder_hidden_dim)

        layers = []
        in_ch = encoder_dim
        for _ in range(up_steps):
            out_ch = max(min_width, width)
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                ]
            )
            in_ch = out_ch
            width = max(min_width, width // 2)

        layers.extend(
            [
                nn.Conv2d(in_ch, max(min_width, in_ch // 2), kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(max(min_width, in_ch // 2), self.in_channels, kernel_size=1),
            ]
        )
        return nn.Sequential(*layers)

    def _build_decoder(self, encoder_dim: int) -> nn.Module:
        if self.decoder_type == 'progressive':
            return self._build_progressive_decoder(encoder_dim=encoder_dim)
        return self._build_pixelshuffle_decoder(encoder_dim=encoder_dim)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.output_activation == 'tanh':
            return 0.5 * (torch.tanh(x) + 1.0)
        if self.output_activation in {'none', 'identity', ''}:
            return x
        raise ValueError(f"Unsupported output_activation '{self.output_activation}'.")

    def _vessel_prior(self, x: torch.Tensor) -> torch.Tensor:
        k = self.vessel_prior_kernel_size
        if k % 2 == 0:
            k += 1

        local_mean = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
        dark_ridges = (local_mean - x).clamp_min(0.0)
        if dark_ridges.shape[1] > 1:
            dark_ridges = dark_ridges.mean(dim=1, keepdim=True)

        max_vals = dark_ridges.amax(dim=(2, 3), keepdim=True)
        return dark_ridges / (max_vals + 1e-6)

    @staticmethod
    def _image_gradients(x: torch.Tensor) -> torch.Tensor:
        grad_x = x[..., :, 1:] - x[..., :, :-1]
        grad_y = x[..., 1:, :] - x[..., :-1, :]
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        return torch.cat([grad_x, grad_y], dim=1)

    def _masked_l1(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        diff = torch.abs(target - prediction)
        mask_expanded = mask.unsqueeze(1)

        # Patch variance weighting (local spatial variance)
        k = 16
        local_mean = F.avg_pool2d(target, kernel_size=k, stride=k)
        local_sq_mean = F.avg_pool2d(target**2, kernel_size=k, stride=k)
        local_var = (local_sq_mean - local_mean**2).clamp_min(0.0)
        local_var = F.interpolate(local_var, size=target.shape[-2:], mode='nearest')
        patch_weight = local_var / (local_var.amax(dim=(2,3), keepdim=True) + 1e-6)
        
        # Normalize patch_weight so that its mean inside the mask is 1.0
        # This prevents the raw loss scale (and gradients) from shrinking too much
        pw_mean = (patch_weight * mask_expanded).sum(dim=(2,3), keepdim=True) / (mask_expanded.sum(dim=(2,3), keepdim=True) + 1e-6)
        patch_weight = patch_weight / (pw_mean + 1e-6)

        if weights is not None:
            if weights.dim() == 3:
                weights = weights.unsqueeze(1)
            weights = weights * patch_weight
        else:
            weights = patch_weight

        weighted_mask = mask_expanded * weights
        denom = (weighted_mask.sum() * target.shape[1]).clamp_min(1e-6)
        return (diff * weighted_mask).sum() / denom

    def _normalize_mask(self, x, mask):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError(f"Mask must have shape [H, W] or [B, H, W], got {tuple(mask.shape)}")

        if mask.shape[0] == 1 and x.shape[0] > 1:
            mask = mask.expand(x.shape[0], -1, -1)
        elif mask.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch between x ({x.shape[0]}) and mask ({mask.shape[0]})"
            )

        if mask.shape[-2:] != x.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=x.shape[-2:], mode='nearest').squeeze(1)

        return mask.to(device=x.device, dtype=x.dtype)

    def _extract_hog(self, x):
        # Extract 9-bin fast Histogram of Oriented Gradients-like features mapping pixels directly
        weight_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
        weight_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)
        g_x = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), weight_x)
        g_y = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), weight_y)
        
        magnitude = torch.sqrt(g_x**2 + g_y**2 + 1e-6)
        angle = torch.atan2(g_y, g_x) * 180 / 3.141592653589793
        angle = torch.where(angle < 0, angle + 180, angle)
        
        bins = torch.linspace(0, 180, 10, device=x.device)
        hog_feats = []
        for i in range(9):
            bin_mask = (angle >= bins[i]) & (angle < bins[i+1])
            hog_feats.append(magnitude * bin_mask.float())
            
        hog = torch.cat(hog_feats, dim=1) # (B, 9, H, W)
        return hog / (torch.norm(hog, p=2, dim=1, keepdim=True) + 1e-6)

    def forward(self, x, mask):
        mask = self._normalize_mask(x, mask)
        x_masked = x * (1 - mask.unsqueeze(1))
        z = self.encoder(x_masked)
        if isinstance(z, (list, tuple)):
            z = z[0]
        x_rec = self.decoder(z)
        x_rec = self._apply_output_activation(x_rec)
        if x_rec.shape[-2:] != x.shape[-2:]:
            x_rec = F.interpolate(x_rec, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # MASKFEAT: Predict HOG instead of raw pixels
        # The model reconstructs an image, but the loss is computed purely in the HOG space.
        target_hog = self._extract_hog(x)
        pred_hog = self._extract_hog(x_rec)

        vessel_weights = 1.0 + self.vessel_focus_weight * self._vessel_prior(x)
        pixel_loss = self._masked_l1(target=target_hog, prediction=pred_hog, mask=mask, weights=vessel_weights)

        gradient_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.gradient_loss_weight > 0.0:
            grad_target = self._image_gradients(x)
            grad_pred = self._image_gradients(x_rec)

            grad_mask = F.max_pool2d(mask.unsqueeze(1), kernel_size=3, stride=1, padding=1)
            grad_weights = 1.0 + self.vessel_focus_weight * self._vessel_prior(x)
            grad_weights = grad_weights.expand(-1, grad_target.shape[1], -1, -1)
            grad_mask = grad_mask.expand(-1, grad_target.shape[1], -1, -1)

            grad_diff = torch.abs(grad_target - grad_pred)
            grad_norm = (grad_mask * grad_weights).sum().clamp_min(1e-6)
            gradient_loss = (grad_diff * grad_mask * grad_weights).sum() / grad_norm

        loss = self.pixel_loss_weight * pixel_loss + self.gradient_loss_weight * gradient_loss

        loss_terms = {
            "pixel_loss": pixel_loss.detach(),
            "gradient_loss": gradient_loss.detach(),
            "vessel_focus_weight": torch.tensor(self.vessel_focus_weight, device=x.device, dtype=x.dtype),
        }
        
        return loss, x_rec, loss_terms