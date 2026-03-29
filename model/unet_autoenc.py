from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import silu

from .latentnet import *
from .unet import *
from choices import *


@dataclass
class BeatGANsAutoencConfig(BeatGANsUNetConfig):
    # number of style channels
    enc_out_channels: int = 512
    enc_attn_resolutions: Tuple[int] = None
    enc_pool: str = 'depthconv'
    enc_num_res_block: int = 2
    enc_channel_mult: Tuple[int] = None
    enc_grad_checkpoint: bool = False
    latent_net_conf: MLPSkipNetConfig = None

    def make_model(self):
        return BeatGANsAutoencModel(self)


class BeatGANsAutoencModel(BeatGANsUNetModel):
    def __init__(self, conf: BeatGANsAutoencConfig):
        super().__init__(conf)
        self.conf = conf

        # having only time, cond
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=conf.model_channels,
            time_out_channels=conf.embed_channels,
        )

        self.encoder = BeatGANsEncoderConfig(
            image_size=conf.image_size,
            in_channels=conf.in_channels,
            model_channels=conf.model_channels,
            out_hid_channels=conf.enc_out_channels,
            out_channels=conf.enc_out_channels,
            num_res_blocks=conf.enc_num_res_block,
            attention_resolutions=(conf.enc_attn_resolutions
                                   or conf.attention_resolutions),
            dropout=conf.dropout,
            channel_mult=conf.enc_channel_mult or conf.channel_mult,
            use_time_condition=False,
            conv_resample=conf.conv_resample,
            dims=conf.dims,
            use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
            num_heads=conf.num_heads,
            num_head_channels=conf.num_head_channels,
            resblock_updown=conf.resblock_updown,
            use_new_attention_order=conf.use_new_attention_order,
            pool=conf.enc_pool,
        ).make_model()

        if conf.latent_net_conf is not None:
            self.latent_net = conf.latent_net_conf.make_model()

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        assert self.conf.is_stochastic
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_z(self, n: int, device):
        assert self.conf.is_stochastic
        return torch.randn(n, self.conf.enc_out_channels, device=device)

    def noise_to_cond(self, noise: Tensor):
        raise NotImplementedError()
        assert self.conf.noise_net_conf is not None
        return self.noise_net.forward(noise)

    def encode(self, x):
        cond = self.encoder.forward(x)
        return {'cond': cond}

    @property
    def stylespace_sizes(self):
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        sizes = []
        for module in modules:
            if isinstance(module, ResBlock):
                linear = module.cond_emb_layers[-1]
                sizes.append(linear.weight.shape[0])
        return sizes

    def encode_stylespace(self, x, return_vector: bool = True):
        """
        encode to style space
        """
        modules = list(self.input_blocks.modules()) + list(
            self.middle_block.modules()) + list(self.output_blocks.modules())
        # (n, c)
        cond = self.encoder.forward(x)
        S = []
        for module in modules:
            if isinstance(module, ResBlock):
                # (n, c')
                s = module.cond_emb_layers.forward(cond)
                S.append(s)

        if return_vector:
            # (n, sum_c)
            return torch.cat(S, dim=1)
        else:
            return S

    def forward(self,
                x,
                t,
                y=None,
                x_start=None,
                cond=None,
                style=None,
                noise=None,
                t_cond=None,
                cached_data=None,
                cached_scheduler=None,
                cache_debug_collector=None,
                **kwargs):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
            cached_data: Dict to store cached layer outputs (created in ddim_sample_loop_progressive)
            cached_scheduler: List indicating which layers should recompute
            cache_debug_collector: Optional callable(layer_key, h, *, recompute, t) for Stage2 diagnostics.
                預設 None；recompute=True 為該步 forward 計算，False 為讀取 cached_data 重用。
        """

        if t_cond is None:
            t_cond = t

        if noise is not None:
            # if the noise is given, we predict the cond from noise
            cond = self.noise_to_cond(noise)
            print('using noise to cond')

        if cond is None:
            if x is not None:
                assert len(x) == len(x_start), f'{len(x)} != {len(x_start)}'

            tmp = self.encode(x_start)
            cond = tmp['cond']
            #print('using x_start to cond')

        if t is not None:
            _t_emb = timestep_embedding(t, self.conf.model_channels)
            _t_cond_emb = timestep_embedding(t_cond, self.conf.model_channels)
        else:
            # this happens when training only autoenc
            _t_emb = None
            _t_cond_emb = None

        if self.conf.resnet_two_cond:
            res = self.time_embed.forward(
                time_emb=_t_emb,
                cond=cond,
                time_cond_emb=_t_cond_emb,
            )
        else:
            raise NotImplementedError()

        if self.conf.resnet_two_cond:
            # two cond: first = time emb, second = cond_emb
            emb = res.time_emb
            cond_emb = res.emb
        else:
            # one cond = combined of both time and cond
            emb = res.emb
            cond_emb = None

        # override the style if given
        style = style or res.style

        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # where in the model to supply time conditions
        enc_time_emb = emb
        mid_time_emb = emb
        dec_time_emb = emb
        #print('emb:', emb.data)
        # where in the model to supply style conditions
        enc_cond_emb = cond_emb
        mid_cond_emb = cond_emb
        dec_cond_emb = cond_emb
        #print('cond_emb:', cond_emb.data)
        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]

        # 提取 cache 相關參數（優先使用顯式參數，否則從 kwargs 獲取以保持向後兼容）
        if cached_data is None:
            cached_data = kwargs.get('cached_data', None)
        if cached_scheduler is None:
            cached_scheduler = kwargs.get('cached_scheduler', None)
        if cache_debug_collector is None:
            cache_debug_collector = kwargs.pop('cache_debug_collector', None)
        activate_cache = cached_data is not None and cached_scheduler is not None
        
        # 如果啟用 cache 但 cached_data 為 None，初始化它
        if activate_cache and cached_data is None:
            cached_data = {}
        
        # 初始化 layer_count（用於索引 cached_scheduler）
        layer_count = 0

        def _cache_dbg(layer_key: str, h, recompute: bool):
            if cache_debug_collector is None:
                return
            cache_debug_collector(layer_key, h, recompute=recompute, t=t)

        if x is not None:
            h = x.type(self.dtype)

            # input blocks
            k = 0
            for i in range(len(self.input_num_blocks)):
                for j in range(self.input_num_blocks[i]):
                    if activate_cache:
                        # 檢查是否應該重新計算
                        # 如果 layer_count 超出 cached_scheduler 範圍，默認重新計算
                        if layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 1:
                            # 需要重新計算
                            h = self.input_blocks[k](h,
                                                     emb=enc_time_emb,
                                                     cond=enc_cond_emb)
                            # 存入 cache
                            cached_data[f'encoder_layer_{k}'] = h
                            _cache_dbg(f'encoder_layer_{k}', h, True)
                        elif layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 0:
                            # 使用 cache
                            h = cached_data[f'encoder_layer_{k}']
                            _cache_dbg(f'encoder_layer_{k}', h, False)
                        else:
                            # layer_count 超出範圍，默認重新計算
                            h = self.input_blocks[k](h,
                                                     emb=enc_time_emb,
                                                     cond=enc_cond_emb)
                            cached_data[f'encoder_layer_{k}'] = h
                            _cache_dbg(f'encoder_layer_{k}', h, True)
                    else:
                        # 不使用 cache，正常計算
                        h = self.input_blocks[k](h,
                                                 emb=enc_time_emb,
                                                 cond=enc_cond_emb)
                        _cache_dbg(f'encoder_layer_{k}', h, True)

                    # print(i, j, h.shape)
                    hs[i].append(h)
                    k += 1
                    if activate_cache:
                        layer_count += 1
            assert k == len(self.input_blocks)

            # middle blocks
            if activate_cache:
                if layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 1:
                    h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
                    cached_data['middle_layer'] = h
                    _cache_dbg('middle_layer', h, True)
                elif layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 0:
                    h = cached_data['middle_layer']
                    _cache_dbg('middle_layer', h, False)
                else:
                    # layer_count 超出範圍，默認重新計算
                    h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
                    cached_data['middle_layer'] = h
                    _cache_dbg('middle_layer', h, True)
                layer_count += 1
            else:
                h = self.middle_block(h, emb=mid_time_emb, cond=mid_cond_emb)
                _cache_dbg('middle_layer', h, True)
        else:
            # no lateral connections
            # happens when training only the autonecoder
            h = None
            hs = [[] for _ in range(len(self.conf.channel_mult))]

        # output blocks
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)

                if activate_cache:
                    if layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 1:
                        h = self.output_blocks[k](h,
                                                  emb=dec_time_emb,
                                                  cond=dec_cond_emb,
                                                  lateral=lateral)
                        cached_data[f'decoder_layer_{k}'] = h
                        _cache_dbg(f'decoder_layer_{k}', h, True)
                    elif layer_count < len(cached_scheduler) and cached_scheduler[layer_count] == 0:
                        h = cached_data[f'decoder_layer_{k}']
                        _cache_dbg(f'decoder_layer_{k}', h, False)
                    else:
                        # layer_count 超出範圍，默認重新計算
                        h = self.output_blocks[k](h,
                                                  emb=dec_time_emb,
                                                  cond=dec_cond_emb,
                                                  lateral=lateral)
                        cached_data[f'decoder_layer_{k}'] = h
                        _cache_dbg(f'decoder_layer_{k}', h, True)
                    layer_count += 1
                else:
                    h = self.output_blocks[k](h,
                                              emb=dec_time_emb,
                                              cond=dec_cond_emb,
                                              lateral=lateral)
                    _cache_dbg(f'decoder_layer_{k}', h, True)
                k += 1

        pred = self.out(h)
        return AutoencReturn(pred=pred, cond=cond)


class AutoencReturn(NamedTuple):
    pred: Tensor
    cond: Tensor = None


class EmbedReturn(NamedTuple):
    # style and time
    emb: Tensor = None
    # time only
    time_emb: Tensor = None
    # style only (but could depend on time)
    style: Tensor = None


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            linear(time_channels, time_out_channels),
            nn.SiLU(),
            linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is None:
            # happens with autoenc training mode
            time_emb = None
        else:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        return EmbedReturn(emb=style, time_emb=time_emb, style=style)
