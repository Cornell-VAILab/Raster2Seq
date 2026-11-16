import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import inverse_sigmoid

from .deformable_transformer import (
    DeformableTransformerEncoder, 
    DeformableTransformerEncoderLayer, 
    MSDeformAttn
)
from .kv_cache import KVCache, VCache


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


def get_1d_sincos_pos_embed_from_grid(embed_dim, seq_len):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    pos = np.arange(seq_len, dtype=np.float32)
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        poly_refine=True,
        return_intermediate_dec=False,
        aux_loss=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        query_pos_type="none",
        vocab_size=None,
        seq_len=1024,
        pre_decoder_pos_embed=False,
        learnable_dec_pe=False,
        dec_attn_concat_src=False,
        dec_qkv_proj=True,
        pad_idx=None,
        use_anchor=False,
        inject_cls_embed=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.poly_refine = poly_refine
        self.use_anchor = use_anchor
        self.inject_cls_embed = inject_cls_embed

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_qkv_proj=(dec_qkv_proj and not dec_attn_concat_src),
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            poly_refine,
            return_intermediate_dec,
            aux_loss,
            query_pos_type,
            vocab_size,
            pad_idx,
            use_anchor=use_anchor,
        )

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if query_pos_type == "sine" and (poly_refine or use_anchor):
            self.decoder.pos_trans = nn.Linear(d_model, d_model)
            self.decoder.pos_trans_norm = nn.LayerNorm(d_model)

        self.pre_decoder_pos_embed = pre_decoder_pos_embed

        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model), requires_grad=learnable_dec_pe)
        pos_embed = get_1d_sincos_pos_embed_from_grid(d_model, seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.dec_attn_concat_src = dec_attn_concat_src

        if self.inject_cls_embed:
            self.decoder.room_class_trans = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False), nn.LayerNorm(d_model)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _create_causal_attention_mask(self, seq_len):
        """
        Creates a causal attention mask for a sequence of length `seq_len`.
        """
        # Create an upper triangular matrix with 1s above the diagonal
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # Invert the mask: 1 -> -inf (masked), 0 -> 0 (unmasked)
        causal_mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, 0.0)
        return causal_mask

    def forward(
        self,
        srcs,
        masks,
        pos_embeds,
        query_embed=None,
        tgt=None,
        tgt_masks=None,
        seq_kwargs=None,
        force_simple_returns=False,
        return_enc_cache=False,
        enc_cache=None,
        decode_token_pos=None,
    ):
        # assert query_embed is not None

        if enc_cache is None:
            # prepare input for encoder
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)
            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

            # encoder
            memory = self.encoder(
                src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
            )
            enc_cache_output = {
                "memory": memory,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
                "valid_ratios": valid_ratios,
                "mask_flatten": mask_flatten,
                "src_flatten": src_flatten,
            }
        else:
            memory, spatial_shapes, level_start_index, valid_ratios, mask_flatten = (
                enc_cache["memory"],
                enc_cache["spatial_shapes"],
                enc_cache["level_start_index"],
                enc_cache["valid_ratios"],
                enc_cache["mask_flatten"],
            )
            src_flatten = enc_cache["src_flatten"]
            enc_cache_output = enc_cache

        # prepare input for decoder
        bs, _, c = memory.shape

        assert not (self.use_anchor and self.poly_refine), "use_anchor and poly_refine cannot be used together"
        if self.poly_refine or self.use_anchor:
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            reference_points = query_embed.sigmoid()
            query_pos = None  # inferred from reference_points
        else:
            reference_points = None
            # training sequences may be shorter than seq_len (--trim_padding); generation indexes the full table
            query_pos = self.pos_embed if decode_token_pos is not None else self.pos_embed[:, : seq_kwargs["seq11"].shape[1]]
        init_reference_out = reference_points

        if tgt_masks is None:
            # make causal mask
            if decode_token_pos is not None:
                tgt_masks = torch.zeros(1, decode_token_pos.max() + 1, dtype=torch.float).to(memory.device)
            else:
                tgt_masks = self._create_causal_attention_mask(seq_kwargs["seq11"].shape[1]).to(memory.device)

        # decoder
        hs, inter_references, inter_classes = self.decoder(
            tgt,
            reference_points,
            memory,
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_pos,
            mask_flatten,
            tgt_masks,
            seq_kwargs,
            force_simple_returns=force_simple_returns,
            pre_decoder_pos_embed=self.pre_decoder_pos_embed,
            attn_concat_src=self.dec_attn_concat_src,
            decode_token_pos=decode_token_pos,
        )
        if return_enc_cache:
            return hs, init_reference_out, inter_references, inter_classes, enc_cache_output
        return hs, init_reference_out, inter_references, inter_classes

    def _setup_caches(self, max_batch_size, max_seq_length, max_vision_length, model_dim, nhead, dtype, device):
        for layer in self.decoder.layers:
            layer.kv_cache = KVCache(max_batch_size, max_seq_length, model_dim, dtype).to(device)
            layer.cross_attn.cache = VCache(
                max_batch_size, max_vision_length, nhead, int(model_dim // nhead), dtype
            ).to(device)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_qkv_proj=True,
    ):
        super().__init__()
        self.d_model = d_model

        if use_qkv_proj:
            self.attn_q = nn.Linear(d_model, d_model, bias=False)
            self.attn_k = nn.Linear(d_model, d_model, bias=False)
            self.attn_v = nn.Linear(d_model, d_model, bias=False)
        else:
            self.attn_q = nn.Identity()
            self.attn_k = nn.Identity()
            self.attn_v = nn.Identity()

        # attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.kv_cache = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos[:, : tensor.size(1)]

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
        tgt_masks=None,
        attn_concat_src=False,
        input_pos=None,
    ):

        q = self.with_pos_embed(self.attn_q(tgt), query_pos)
        # self attention
        if self.kv_cache is not None and input_pos is not None:
            k = self.attn_k(tgt)
            v = self.attn_v(tgt)
            k, v = self.kv_cache.update(input_pos, k, v)
        else:
            k = self.attn_k(tgt)
            v = self.attn_v(tgt)

        if attn_concat_src:
            k = torch.cat([src, k], dim=1)
            v = torch.cat([src, v], dim=1)
            tgt_masks = torch.cat([torch.zeros(q.size(1), src.size(1), device=q.device), tgt_masks], dim=1).to(
                dtype=torch.float32
            )

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=tgt_masks)[
            0
        ].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
            use_cache=(input_pos is not None and input_pos[0] != 0),
        )  # disable cache when processing first token
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt, None


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        poly_refine=True,
        return_intermediate=False,
        aux_loss=False,
        query_pos_type="none",
        vocab_size=None,
        pad_idx=None,
        use_anchor=None,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.poly_refine = poly_refine
        self.return_intermediate = return_intermediate
        self.aux_loss = aux_loss
        self.query_pos_type = query_pos_type

        self.coords_embed = None
        self.class_embed = None
        self.pos_trans = None
        self.pos_trans_norm = None
        self.use_anchor = use_anchor

        self.room_class_embed = None
        self.room_class_trans = None

        self.token_embed = Embedding(vocab_size, self.layers[0].d_model, padding_idx=pad_idx, zero_init=False)

    def _seq_embed(self, seq11, seq12, seq21, seq22, delta_x1, delta_x2, delta_y1, delta_y2):
        # embedding [B, L, D]
        e11 = self.token_embed(seq11)
        e21 = self.token_embed(seq21)
        e12 = self.token_embed(seq12)
        e22 = self.token_embed(seq22)

        # bilinear interpolation [B, L, D]
        out = (
            e11 * delta_x2[..., None] * delta_y2[..., None]
            + e21 * delta_x1[..., None] * delta_y2[..., None]
            + e12 * delta_x2[..., None] * delta_y1[..., None]
            + e22 * delta_x1[..., None] * delta_y1[..., None]
        )

        return out

    def _add_cls_embed(self, x, input_cls_seq):
        # Suppose class_labels is of shape [batch, seq_len] with integer class indices
        one_hot = F.one_hot(input_cls_seq, num_classes=self.room_class_embed.out_features).float()
        x = x + self.room_class_trans(self.room_class_embed[-1](one_hot))
        return x

    def get_query_pos_embed(self, ref_points):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=ref_points.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)  # [128]
        # N, L, 2
        ref_points = ref_points * scale
        # N, L, 2, 128
        pos = ref_points[:, :, :, None] / dim_t
        # N, L, 256
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos[:, : tensor.size(1)]

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_flatten,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
        tgt_masks=None,
        seq_kwargs=None,
        force_simple_returns=False,
        pre_decoder_pos_embed=False,
        attn_concat_src=False,
        decode_token_pos=None,
    ):
        # print(seq_kwargs['seq11'].max(),seq_kwargs['seq21'].max(), seq_kwargs['seq12'].max(), seq_kwargs['seq22'].max())

        output = self._seq_embed(
            seq11=seq_kwargs["seq11"],
            seq12=seq_kwargs["seq12"],
            seq21=seq_kwargs["seq21"],
            seq22=seq_kwargs["seq22"],
            delta_x1=seq_kwargs["delta_x1"],
            delta_x2=seq_kwargs["delta_x2"],
            delta_y1=seq_kwargs["delta_y1"],
            delta_y2=seq_kwargs["delta_y2"],
        )  # [B, L, D]

        if decode_token_pos is not None:
            if query_pos is not None:  # if using abs pos_embed
                query_pos = query_pos[:, decode_token_pos]
            if reference_points is not None:
                reference_points = reference_points[:, decode_token_pos : decode_token_pos + 1]

        if reference_points is None:
            reference_points = torch.zeros(output.shape[0], output.shape[1], 2).to(output.device)

        # assert not(pre_decoder_pos_embed and self.poly_refine), 'pre_decoder_pos_embed and poly_refine cannot be used together'

        if pre_decoder_pos_embed:
            # infer query_pos from reference_points
            if (self.poly_refine or self.use_anchor) and self.query_pos_type == "sine":
                query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))
            output = self.with_pos_embed(output, query_pos)
            query_pos = None

        if self.room_class_trans is not None:
            # add class embedding
            output = self._add_cls_embed(output, seq_kwargs["input_polygon_labels"])

        intermediate = []
        intermediate_reference_points = []
        intermediate_classes = []
        point_classes = torch.zeros(output.shape[0], output.shape[1], self.class_embed[0].out_features).to(
            output.device
        )
        for lid, layer in enumerate(self.layers):
            if self.poly_refine or self.use_anchor:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                # disable adding query_pos for every layer
                if not pre_decoder_pos_embed:
                    if self.query_pos_type == "sine":
                        query_pos = self.pos_trans_norm(self.pos_trans(self.get_query_pos_embed(reference_points)))

                    elif self.query_pos_type == "none":
                        query_pos = None
            else:
                reference_points_input = None
            output, src_tmp = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                tgt_masks,
                attn_concat_src=attn_concat_src,
                input_pos=decode_token_pos,
            )
            if src_tmp is not None:
                src = src_tmp

            # iterative polygon refinement
            if self.poly_refine:
                offset = self.coords_embed[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = offset
                new_reference_points = offset + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points

            # if not using iterative polygon refinement, just output the reference points decoded from the last layer
            elif lid == len(self.layers) - 1:
                if self.use_anchor:
                    offset = self.coords_embed[-1](output)
                    assert reference_points.shape[-1] == 2
                    new_reference_points = offset
                    new_reference_points = offset + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points
                else:
                    reference_points = self.coords_embed[-1](output).sigmoid()

            # If aux loss supervision, we predict classes label from each layer and supervise loss
            if self.aux_loss:
                point_classes = self.class_embed[lid](output)
            # Otherwise, we only predict class label from the last layer
            elif lid == len(self.layers) - 1:
                point_classes = self.class_embed[-1](output)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                intermediate_classes.append(point_classes)

        if self.return_intermediate and not force_simple_returns:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
                torch.stack(intermediate_classes),
            )

        return output, reference_points, point_classes


def _get_clones(module, N):
    if isinstance(module, list):
        return nn.ModuleList(module)
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args, pad_idx=None):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        poly_refine=args.with_poly_refine,
        return_intermediate_dec=True,
        aux_loss=args.aux_loss,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        query_pos_type=args.query_pos_type,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        pre_decoder_pos_embed=args.pre_decoder_pos_embed,
        learnable_dec_pe=args.learnable_dec_pe,
        dec_attn_concat_src=args.dec_attn_concat_src,
        dec_qkv_proj=args.dec_qkv_proj,
        pad_idx=pad_idx,
        use_anchor=args.use_anchor,
        inject_cls_embed=getattr(args, "inject_cls_embed", False),
    )
