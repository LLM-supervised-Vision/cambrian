import torch
import torch.nn.functional as F
import open_clip
import numpy as np

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP":"hf-hub:timm/ViT-SO400M-14-SigLIP"
    }

    res = 384 if '384' in model_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp

@torch.no_grad()
def load_any_bv_vit(model, checkpoint_path, strict=True):
    print(f"HELLO! I changed the load_checkpoint function!!!!")
    from timm.layers import resample_patch_embed, resample_abs_pos_embed

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])

        # if w.dtype == np.dtype('|V2'):
        #     w_float16 = w.view(np.float16)
        #     w = w_float16.astype(np.float32)

        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    
    module = model.visual.trunk
    prefix = 'params/img/'
    embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    if embed_conv_w.shape[-2:] != module.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            module.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    module.patch_embed.proj.weight.copy_(embed_conv_w)
    module.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

    if module.cls_token is not None:
        module.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

    pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    if pos_embed_w.shape != module.pos_embed.shape:
        assert False, f'{pos_embed_w.shape}, {module.pos_embed.shape}'
        num_prefix_tokens = 0 if getattr(module, 'no_embed_class', False) else getattr(module, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=module.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    module.pos_embed.copy_(pos_embed_w)

    mha_sub, b_sub, ln1_sub = (0, 0, 1)
    for i, block in enumerate(module.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))

    module_norm = module.norm if isinstance(module.norm,torch.nn.LayerNorm) else module.fc_norm
    module_norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    module_norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if module.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        module.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        module.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        module.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        module.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        module.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        module.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        module.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        module.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        module.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(module.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(module.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

open_clip.convert.load_big_vision_weights = load_any_bv_vit

class SiglipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        if not vision_tower_name.endswith(".npz"):
            base_model_name, res, interp = extract_res_interp(vision_tower_name)
            self.vision_tower_name = base_model_name
            self._image_size = res if res is not None else 512
            self._interp_size = interp
        else:
            self.vision_tower_name = vision_tower_name
            self._image_size = args.mm_image_size
            self._interp_size = None
            self._hidden_size = args.mm_hidden_size
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        elif not self._hidden_size:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        if self.vision_tower_name.endswith(".npz"):
            model_kwargs = {
                'vision_cfg': {
                    'image_size': 224, 
                    'timm_model_name': 'vit_base_patch16_siglip_224', 
                    'timm_model_pretrained': False, 
                    'timm_pool': '', 
                    'timm_proj': 'none', # 'none', 'linear'
                }, 
                'embed_dim': self._hidden_size,
            }
            clip_model, processor = open_clip.create_model_from_pretrained(
                "ViT-B-16-SigLIP",
                pretrained=self.vision_tower_name,
                **model_kwargs
            )
        else:
            clip_model, processor = open_clip.create_model_from_pretrained(self.vision_tower_name)

        self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True


    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, interpolate_token = 576):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            interp_features = self.interpolate(image_features)
            return interp_features
