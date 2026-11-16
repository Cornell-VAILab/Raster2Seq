import collections
import copy
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from datasets.poly_data import TokenType
from util.misc import NestedTensor, nested_tensor_from_tensor_list

from .backbone import build_backbone

from .deformable_transformer_v2 import build_deforamble_transformer
from .label_smoothing_loss import label_smoothed_nll_loss
from .losses import MaskRasterizationLoss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Raster2Seq(nn.Module):
    """This is the RoomFormer module that performs floorplan reconstruction"""

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        num_polys,
        num_feature_levels,
        aux_loss=True,
        with_poly_refine=False,
        masked_attn=False,
        semantic_classes=-1,
        seq_len=1024,
        tokenizer=None,
        use_anchor=False,
        patch_size=1,
        freeze_anchor=False,
        inject_cls_embed=False,
        add_geom_head=False,
        label_geom=None,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of possible corners
                         in a single image.
            num_polys: maximal number of possible polygons in a single image.
                       num_queries/num_polys would be the maximal number of possible corners in a single polygon.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_poly_refine: iterative polygon refinement
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_polys = num_polys
        assert num_queries % num_polys == 0
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.coords_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.num_feature_levels = num_feature_levels
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.inject_cls_embed = inject_cls_embed

        # self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size, padding=0),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                if patch_size == 1:
                    input_proj_list.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                            nn.GroupNorm(32, hidden_dim),
                        )
                    )
                else:
                    input_proj_list.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels, hidden_dim, kernel_size=2 * patch_size, stride=2 * patch_size, padding=0
                            ),
                            nn.GroupNorm(32, hidden_dim),
                        )
                    )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_poly_refine = with_poly_refine

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.coords_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coords_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = transformer.decoder.num_layers

        if with_poly_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.coords_embed = _get_clones(self.coords_embed, num_pred)
            nn.init.constant_(self.coords_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            nn.init.constant_(self.coords_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.coords_embed = nn.ModuleList([self.coords_embed for _ in range(num_pred)])

        if use_anchor or with_poly_refine:
            self.query_embed = nn.Embedding(seq_len, 2)
            self.query_embed.weight.requires_grad = not freeze_anchor
        else:
            self.query_embed = None

        self.transformer.decoder.coords_embed = self.coords_embed
        self.transformer.decoder.class_embed = self.class_embed

        # Semantically-rich floorplan
        self.room_class_embed = None
        if semantic_classes > 0:
            self.room_class_embed = nn.Linear(hidden_dim, semantic_classes)
            if self.inject_cls_embed:
                self.transformer.decoder.room_class_embed = self.room_class_embed

        # self.num_queries_per_poly = num_queries // num_polys

        # # The attention mask is used to prevent object queries in one polygon attending to another polygon, default false
        # if masked_attn:
        #     self.attention_mask = torch.ones((num_queries, num_queries), dtype=torch.bool)
        #     for i in range(num_polys):
        #         self.attention_mask[i * self.num_queries_per_poly:(i + 1) * self.num_queries_per_poly,
        #         i * self.num_queries_per_poly:(i + 1) * self.num_queries_per_poly] = False
        # else:
        #     self.attention_mask = None

        self.register_buffer("attention_mask", self._create_causal_attention_mask(seq_len))

        # --add_geom_head: geometry head (polygon / line) + a learned bias on the label logits per geometry type
        self.geom_embed = self.geom_bias = None
        if add_geom_head and semantic_classes > 0:
            self.geom_embed = nn.Linear(hidden_dim, 2)
            self.geom_bias = nn.Embedding(2, semantic_classes)
            nn.init.zeros_(self.geom_bias.weight)  # starts as the plain label head

        # --geom_mask: each label's geometry type (1 line, 0 polygon, -1 either); saved with the checkpoint
        # get geometries associated with each label if geom_mask is set
        use_mask = label_geom is not None and self.geom_embed is not None
        self.register_buffer("label_geom", torch.tensor(label_geom, dtype=torch.long) if use_mask else None)

    def _create_causal_attention_mask(self, seq_len):
        """
        Creates a causal attention mask for a sequence of length `seq_len`.
        """
        # Create an upper triangular matrix with 1s above the diagonal
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        # Invert the mask: 1 -> -inf (masked), 0 -> 0 (unmasked)
        causal_mask = mask.masked_fill(mask == 1, float("-inf")).masked_fill(mask == 0, 0.0)
        return causal_mask

    def _room_logits(self, hs, target_geom=None):
        """With --add_geom_head, the label logits get a learned bias per geometry type (true type where
        target_geom >= 0, otherwise the predicted one). With --geom_mask, generation also rules out labels of
        the other geometry type. Returns (label logits, geometry logits)."""
        # same as before
        if self.geom_embed is None:
            return self.room_class_embed(hs), None
        # otherwise: get geom embedding
        geom_logits = self.geom_embed(hs)
        # make predictions
        geom = geom_logits.argmax(-1)
        # use real values otherwise predictions
        if target_geom is not None:
            geom = torch.where(target_geom >= 0, target_geom, geom)
        # add the geometry bias
        logits = self.room_class_embed(hs) + self.geom_bias(geom)
        # if using a geom mask
        if self.label_geom is not None and target_geom is None:  # generation only (see geom_mask help)
            wrong = (self.label_geom >= 0) & (self.label_geom != geom[..., None])
            # remove "wrong" class labels
            logits = logits.masked_fill(wrong, float("-inf"))
        return logits, geom_logits

    def forward(self, samples: NestedTensor, seq_kwargs=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensors: batched images, of shape [batch_size x C x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_coords": The normalized corner coordinates for all queries, represented as
                           (x, y). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            src = self.input_proj[l](src)
            srcs.append(src)
            if self.patch_size != 1:
                mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos[l] = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None if self.query_embed is None else self.query_embed.weight
        tgt_embeds = None

        hs, init_reference, inter_references, inter_classes = self.transformer(
            srcs, masks, pos, query_embeds, tgt_embeds, self.attention_mask, seq_kwargs
        )

        outputs_class = inter_classes
        outputs_coord = inter_references

        out = {"pred_logits": outputs_class[-1], "pred_coords": outputs_coord[-1]}

        if self.room_class_embed is not None:
            target_geom = None if seq_kwargs is None else seq_kwargs.get("target_geom")
            out["pred_room_logits"], geom_logits = self._room_logits(hs[-1], target_geom)
            if geom_logits is not None:
                out["pred_geom_logits"] = geom_logits

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    def _prepare_sequences(self, b):
        prev_output_token_11 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_12 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_21 = [[self.tokenizer.bos] for _ in range(b)]
        prev_output_token_22 = [[self.tokenizer.bos] for _ in range(b)]
        delta_x1 = [[0] for _ in range(b)]
        delta_y1 = [[0] for _ in range(b)]
        delta_x2 = [[1] for _ in range(b)]
        delta_y2 = [[1] for _ in range(b)]

        gen_out = [[] for _ in range(b)]

        if self.inject_cls_embed:
            input_polygon_labels = [[self.semantic_classes - 1] for _ in range(b)]
        else:
            input_polygon_labels = [[-1] for _ in range(b)]  # dummies values, not used in inference

        return (
            prev_output_token_11,
            prev_output_token_12,
            prev_output_token_21,
            prev_output_token_22,
            delta_x1,
            delta_x2,
            delta_y1,
            delta_y2,
            gen_out,
            input_polygon_labels,
        )
    
    def forward_inference(self, samples: NestedTensor, use_cache=True, prefix_data=None, enc_data=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensors: batched images, of shape [batch_size x C x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_coords": The normalized corner coordinates for all queries, represented as
                           (x, y). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, NestedTensor):
            device = samples.tensors.device
        elif isinstance(samples, list):
            device = samples[0].device
        else:
            device = samples.device

        if enc_data is None:
            if not isinstance(samples, NestedTensor):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)

            srcs = []
            masks = []
            for l, feat in enumerate(features):
                src, mask = feat.decompose()
                src = self.input_proj[l](src)
                srcs.append(src)
                if self.patch_size != 1:
                    mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos[l] = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                masks.append(mask)
                assert mask is not None
            if self.num_feature_levels > len(srcs):
                _len_srcs = len(srcs)
                for l in range(_len_srcs, self.num_feature_levels):
                    if l == _len_srcs:
                        src = self.input_proj[l](features[-1].tensors)
                    else:
                        src = self.input_proj[l](srcs[-1])
                    m = samples.mask
                    mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                    srcs.append(src)
                    masks.append(mask)
                    pos.append(pos_l)
            enc_data = (srcs, masks, pos)
        else:
            srcs, masks, pos = enc_data

        bs = srcs[0].shape[0]

        ##### decoder part
        if use_cache:
            # kv cache for faster inference
            max_src_len = sum([x.size(2) * x.size(3) for x in srcs])  # 1360
            self._setup_caches(bs, max_src_len)

        if prefix_data is not None:
            (
                prev_output_token_11,
                prev_output_token_12,
                prev_output_token_21,
                prev_output_token_22,
                delta_x1,
                delta_x2,
                delta_y1,
                delta_y2,
                gen_out,
                input_polygon_labels,
            ) = prefix_data
        else:
            (
                prev_output_token_11,
                prev_output_token_12,
                prev_output_token_21,
                prev_output_token_22,
                delta_x1,
                delta_x2,
                delta_y1,
                delta_y2,
                gen_out,
                input_polygon_labels,
            ) = self._prepare_sequences(bs)

        prefix_len = len(prev_output_token_11[0])
        confidences = [[] for _ in range(bs)]

        query_embeds = None if self.query_embed is None else self.query_embed.weight
        # tgt_embeds = self.tgt_embed.weight
        tgt_embeds = None
        enc_cache = None

        num_bins = self.tokenizer.num_bins
        min_len = 6
        max_len = self.tokenizer.seq_len
        unfinish_flag = np.ones(bs)

        i = 0

        output_hs_list = []
        output_cls_list = []
        output_reg_list = []
        while i < max_len and unfinish_flag.any():
            prev_output_tokens_11_tensor = torch.tensor(np.array(prev_output_token_11)[:, i : i + 1]).to(device).long()
            prev_output_tokens_12_tensor = torch.tensor(np.array(prev_output_token_12)[:, i : i + 1]).to(device).long()
            prev_output_tokens_21_tensor = torch.tensor(np.array(prev_output_token_21)[:, i : i + 1]).to(device).long()
            prev_output_tokens_22_tensor = torch.tensor(np.array(prev_output_token_22)[:, i : i + 1]).to(device).long()
            delta_x1_tensor = torch.tensor(np.array(delta_x1)[:, i : i + 1], dtype=torch.float32).to(device)
            delta_x2_tensor = torch.tensor(np.array(delta_x2)[:, i : i + 1], dtype=torch.float32).to(device)
            delta_y1_tensor = torch.tensor(np.array(delta_y1)[:, i : i + 1], dtype=torch.float32).to(device)
            delta_y2_tensor = torch.tensor(np.array(delta_y2)[:, i : i + 1], dtype=torch.float32).to(device)
            input_polygon_labels_tensor = torch.tensor(
                np.array(input_polygon_labels)[:, i : i + 1], dtype=torch.long
            ).to(device)

            seq_kwargs = {
                "seq11": prev_output_tokens_11_tensor,
                "seq12": prev_output_tokens_12_tensor,
                "seq21": prev_output_tokens_21_tensor,
                "seq22": prev_output_tokens_22_tensor,
                "delta_x1": delta_x1_tensor,
                "delta_x2": delta_x2_tensor,
                "delta_y1": delta_y1_tensor,
                "delta_y2": delta_y2_tensor,
                "input_polygon_labels": input_polygon_labels_tensor,
            }

            if not use_cache:
                hs, _, reg_output, cls_output = self.transformer(
                    srcs,
                    masks,
                    pos,
                    query_embeds,
                    tgt_embeds,
                    None,
                    seq_kwargs,
                    force_simple_returns=True,
                    return_enc_cache=use_cache,
                    enc_cache=None,
                    decode_token_pos=None,
                )
                output_hs_list.append(hs[:, i : i + 1])
                output_cls_list.append(cls_output[:, i : i + 1])
                output_reg_list.append(reg_output[:, i : i + 1])
            else:
                decode_token_pos = torch.tensor([i], device=device, dtype=torch.long)
                hs, _, reg_output, cls_output, enc_cache = self.transformer(
                    srcs,
                    masks,
                    pos,
                    query_embeds,
                    tgt_embeds,
                    None,
                    seq_kwargs,
                    force_simple_returns=True,
                    return_enc_cache=use_cache,
                    enc_cache=enc_cache,
                    decode_token_pos=decode_token_pos,
                )
                output_hs_list.append(hs)
                output_cls_list.append(cls_output)
                output_reg_list.append(reg_output)
            cls_probs = torch.softmax(cls_output, dim=2)
            cls_type = torch.argmax(cls_output, 2)
            # print(cls_type, torch.softmax(cls_output, dim=2)[:, :, cls_type], torch.topk(torch.softmax(cls_output, dim=2), k=3))
            for j in range(bs):
                if unfinish_flag[j] == 1:  # prediction is not finished
                    conf = cls_probs[j, 0, cls_type[j, 0]].item()
                    confidences[j].append(conf)

                    if i < prefix_len - 1:
                        # We are still in the prefix, skip appending
                        continue

                    cls_j = cls_type[j, 0].item()
                    if cls_j == TokenType.coord.value or (cls_j == TokenType.eos.value and i < min_len):
                        output_j_x, output_j_y = reg_output[j, 0].detach().cpu().numpy()
                        output_j_x = min(output_j_x, 1)
                        output_j_y = min(output_j_y, 1)

                        gen_out[j].append([output_j_x, output_j_y])

                        output_j_x = output_j_x * (num_bins - 1)
                        output_j_y = output_j_y * (num_bins - 1)

                        output_j_x_floor = math.floor(output_j_x)
                        output_j_y_floor = math.floor(output_j_y)
                        output_j_x_ceil = math.ceil(output_j_x)
                        output_j_y_ceil = math.ceil(output_j_y)

                        # tokenization
                        prev_output_token_11[j].append(output_j_x_floor * num_bins + output_j_y_floor)
                        prev_output_token_12[j].append(output_j_x_floor * num_bins + output_j_y_ceil)
                        prev_output_token_21[j].append(output_j_x_ceil * num_bins + output_j_y_floor)
                        prev_output_token_22[j].append(output_j_x_ceil * num_bins + output_j_y_ceil)

                        delta_x = output_j_x - output_j_x_floor
                        delta_y = output_j_y - output_j_y_floor

                    elif cls_j == TokenType.sep.value:
                        gen_out[j].append(TokenType.sep.value)
                        prev_output_token_11[j].append(self.tokenizer.sep)
                        prev_output_token_12[j].append(self.tokenizer.sep)
                        prev_output_token_21[j].append(self.tokenizer.sep)
                        prev_output_token_22[j].append(self.tokenizer.sep)

                        delta_x = 0
                        delta_y = 0

                    elif cls_j == TokenType.cls.value:
                        gen_out[j].append(TokenType.cls.value)
                        prev_output_token_11[j].append(self.tokenizer.cls)
                        prev_output_token_12[j].append(self.tokenizer.cls)
                        prev_output_token_21[j].append(self.tokenizer.cls)
                        prev_output_token_22[j].append(self.tokenizer.cls)
                        delta_x = 0
                        delta_y = 0

                    else:  # eos is predicted and i >= min_len
                        unfinish_flag[j] = 0
                        gen_out[j].append(TokenType.eos.value)
                        prev_output_token_11[j].append(self.tokenizer.eos)
                        prev_output_token_12[j].append(self.tokenizer.eos)
                        prev_output_token_21[j].append(self.tokenizer.eos)
                        prev_output_token_22[j].append(self.tokenizer.eos)
                        delta_x = 0
                        delta_y = 0

                    delta_x1[j].append(delta_x)
                    delta_y1[j].append(delta_y)
                    delta_x2[j].append(1 - delta_x)
                    delta_y2[j].append(1 - delta_y)

                else:  # prediction is finished
                    if i >= prefix_len - 1:
                        gen_out[j].append(TokenType.eos.value)
                        prev_output_token_11[j].append(self.tokenizer.pad)
                        prev_output_token_12[j].append(self.tokenizer.pad)
                        prev_output_token_21[j].append(self.tokenizer.pad)
                        prev_output_token_22[j].append(self.tokenizer.pad)
                        delta_x = 0
                        delta_y = 0
                        delta_x1[j].append(delta_x)
                        delta_y1[j].append(delta_y)
                        delta_x2[j].append(1 - delta_x)
                        delta_y2[j].append(1 - delta_y)
            i += 1

        seq_data = (
            prev_output_token_11,
            prev_output_token_12,
            prev_output_token_21,
            prev_output_token_22,
            delta_x1,
            delta_x2,
            delta_y1,
            delta_y2,
            gen_out,
            input_polygon_labels,
        )

        all_cls_logits = torch.cat(output_cls_list, dim=1)
        all_reg_coords = torch.cat(output_reg_list, dim=1)

        out = {
            "pred_logits": all_cls_logits,
            "pred_coords": all_reg_coords,
            "gen_out": gen_out,
            "confidences": confidences,
            "seq_data": seq_data,
            "enc_data": enc_data,
        }

        # hack implementation of room label prediction, not compatible with auxiliary loss
        if self.room_class_embed is not None:
            hs = torch.cat(output_hs_list, dim=1)
            outputs_room_class, geom_logits = self._room_logits(hs)
            out = {
                "pred_logits": all_cls_logits,
                "pred_coords": all_reg_coords,
                "pred_room_logits": outputs_room_class,
                "gen_out": gen_out,
                "confidences": confidences,
                "seq_data": seq_data,
                "enc_data": enc_data,
                "anchors": query_embeds.detach(),
            }
            if geom_logits is not None:
                out["pred_geom_logits"] = geom_logits

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{"pred_logits": a, "pred_coords": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _setup_caches(self, max_bs, max_src_len):
        self.transformer._setup_caches(
            max_bs,
            self.seq_len,
            max_src_len,
            self.transformer.d_model,
            self.transformer.nhead,
            self.transformer.level_embed.dtype,
            device=self.transformer.level_embed.device,
        )

    def _count_polys(self, gen):
        count = 0
        for val in gen:
            if val == TokenType.sep.value or val == TokenType.eos.value or val == TokenType.cls.value:
                count += 1
                if val == TokenType.eos.value:
                    break
        return count


class SemHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.shared_layer = nn.Linear(hidden_dim, hidden_dim)
        self.room_embed = nn.Linear(hidden_dim, num_classes - 2)
        self.num_classes = num_classes
        self.window_door_embed = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.normalize(torch.relu(self.shared_layer(x)), p=2, dim=-1, eps=1e-12)
        room_out = self.room_embed(x)
        window_door_out = self.window_door_embed(x)
        out = torch.cat([room_out[:, :, :-1], window_door_out, room_out[:, :, -1:]], dim=-1)
        return out.contiguous()


class SetCriterion(nn.Module):
    """This class computes the loss for multiple polygons.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth polygons and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and coords)
    """

    def __init__(
        self,
        num_classes,
        semantic_classes,
        matcher,
        weight_dict,
        losses,
        label_smoothing=0.0,
        per_token_sem_loss=False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of classes for corner validity (binary)
            semantic_classes: number of semantic classes for polygon (room type, door, window)
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.semantic_classes = semantic_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.label_smoothing = label_smoothing
        self.per_token_sem_loss = per_token_sem_loss

        if "loss_raster" in self.weight_dict:
            self.raster_loss = MaskRasterizationLoss(None)

    def _update_ce_coeff(self, loss_ce_coeff):
        self.weight_dict["loss_ce"] = loss_ce_coeff

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels"
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        target_classes = targets["token_labels"].to(src_logits.device)
        mask = (target_classes != -1).bool()
        loss_ce = label_smoothed_nll_loss(
            src_logits[mask], target_classes[mask], epsilon=self.label_smoothing, reduction="mean"
        )
        losses = {"loss_ce": loss_ce}

        if "pred_room_logits" in outputs:
            room_src_logits = outputs["pred_room_logits"]
            if not self.per_token_sem_loss:
                mask = target_classes == 3  # cls token
                room_target_classes = targets["target_polygon_labels"].to(room_src_logits.device)
                loss_ce_room = label_smoothed_nll_loss(
                    room_src_logits[mask],
                    room_target_classes[room_target_classes != -1],
                    epsilon=self.label_smoothing,
                    reduction="mean",
                )
            else:
                room_target_classes = targets["target_polygon_labels"].to(room_src_logits.device)
                loss_ce_room = label_smoothed_nll_loss(
                    room_src_logits[room_target_classes != -1],
                    room_target_classes[room_target_classes != -1],
                    epsilon=self.label_smoothing,
                    reduction="mean",
                )

            losses = {"loss_ce": loss_ce, "loss_ce_room": loss_ce_room}

        if "pred_geom_logits" in outputs and "target_geom" in targets:
            geom_target = targets["target_geom"].to(src_logits.device)
            keep = geom_target >= 0
            losses["loss_ce_geom"] = F.cross_entropy(outputs["pred_geom_logits"][keep], geom_target[keep])

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty corners
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        losses = {"cardinality_error": 0.0}
        return losses

    def _extract_polygons(self, sequence, token_labels):
        # sequence: [B, N, 2], token_labels: [B, N]
        B, N = token_labels.shape
        polygons = []

        for b in range(B):
            labels = token_labels[b]  # [N]
            coords = sequence[b]  # [N, 2]

            # Find separator and EOS positions
            sep_eos_mask = (labels == 1) | (labels == 2)
            split_indices = torch.nonzero(sep_eos_mask, as_tuple=False).squeeze(-1)

            # Handle empty case
            if len(split_indices) == 0:
                # No separators found, treat entire sequence as one polygon
                corner_mask = labels == 0
                if corner_mask.any():
                    polygons.append(coords[corner_mask])
                continue

            # Create start and end indices
            device = labels.device
            starts = torch.cat([torch.tensor([0], device=device), split_indices[:-1] + 1])
            ends = split_indices

            # Extract polygons between separators
            for s, e in zip(starts, ends):
                if s < e:  # Valid range
                    segment_labels = labels[s:e]
                    segment_coords = coords[s:e]
                    corner_mask = segment_labels == 0
                    if corner_mask.any():
                        polygons.append(segment_coords[corner_mask])

        return polygons

    def loss_polys(self, outputs, targets, indices):
        """Compute the losses related to the polygons:
        1. L1 loss for polygon coordinates
        2. Dice loss for polygon rasterizated binary masks
        """
        assert "pred_coords" in outputs
        src_poly = outputs["pred_coords"]
        device = src_poly.device
        token_labels = targets["token_labels"].to(device)
        mask = (token_labels == 0).bool()
        target_polys = targets["target_seq"].to(device)

        loss_coords = F.l1_loss(src_poly[mask], target_polys[mask])

        losses = {}
        losses["loss_coords"] = loss_coords

        # omit the rasterization loss for semantically-rich floorplan
        if self.weight_dict.get("loss_raster", 0) > 0:
            pred_poly_list = self._extract_polygons(src_poly, token_labels)
            target_poly_list = self._extract_polygons(target_polys, token_labels)
            loss_raster_mask = self.raster_loss(
                pred_poly_list,
                target_poly_list,
                [len(x) for x in target_poly_list],
            )
            losses["loss_raster"] = loss_raster_mask

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {"labels": self.loss_labels, "cardinality": self.loss_cardinality, "polys": self.loss_polys}
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        indices = None

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def label_geometry(dataset_root, semantic_classes):
    """Each label's geometry type from the training annotations: 1 = line, 0 = polygon, -1 = either / unseen.
    All -1 if the annotations aren't there (e.g. predict.py); the checkpoint's table is then loaded instead."""
    path = os.path.join(dataset_root, "annotations", "train.json")
    kinds = collections.defaultdict(set)
    if os.path.exists(path):
        for a in json.load(open(path))["annotations"]:
            kinds[a["category_id"]].add(int(a.get("geom_type") == "line"))
    return [next(iter(kinds[c])) if len(kinds[c]) == 1 else -1 for c in range(semantic_classes)]


def build(args, train=True, tokenizer=None):
    num_classes = 3 if not args.add_cls_token else 4  # <coord> <sep> <eos> <cls>
    if tokenizer is not None:
        pad_idx = tokenizer.pad

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args, pad_idx=pad_idx)
    model = Raster2Seq(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_polys=args.num_polys,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_poly_refine=args.with_poly_refine,
        masked_attn=args.masked_attn,
        semantic_classes=args.semantic_classes,
        seq_len=args.seq_len,
        tokenizer=tokenizer,
        use_anchor=args.use_anchor,
        patch_size=[1, 2][args.image_size == 512],  # 1 for 256x256, 2 for 512x512
        freeze_anchor=getattr(args, "freeze_anchor", False),
        inject_cls_embed=getattr(args, "inject_cls_embed", False),
        add_geom_head=getattr(args, "add_geom_head", False),
        label_geom=label_geometry(args.dataset_root, args.semantic_classes) if getattr(args, "geom_mask", False) else None,
    )

    if not train:
        return model

    device = torch.device(args.device)
    matcher = None  # build_matcher(args)
    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_ce_room": args.room_cls_loss_coef,
        "loss_coords": args.coords_loss_coef,
        # new loss for geom classification only
        "loss_ce_geom": getattr(args, "geom_loss_coef", 1.0),
    }
    if args.raster_loss_coef > 0:
        weight_dict["loss_raster"] = args.raster_loss_coef
    weight_dict["loss_dir"] = 1

    enc_weight_dict = {}
    enc_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(enc_weight_dict)
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ["labels", "polys", "cardinality"]
    # num_classes, matcher, weight_dict, losses
    criterion = SetCriterion(
        num_classes,
        args.semantic_classes,
        matcher,
        weight_dict,
        losses,
        label_smoothing=args.label_smoothing,
        per_token_sem_loss=args.per_token_sem_loss,
    )
    criterion.to(device)

    return model, criterion
