from collections import OrderedDict

import torch
from pyvene import (
    ConstantSourceIntervention,
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)
from pyvene.models.layers import LowRankRotateLayer
from transformers.activations import ACT2FN


class DynamicSteeringIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        # rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        # self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        # self.learned_source = torch.nn.Linear(
        #     self.embed_dim, kwargs["low_rank_dimension"]
        # ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        # self.act_fn = (
        #     ACT2FN["linear"]
        #     if "act_fn" not in kwargs or kwargs["act_fn"] is None
        #     else ACT2FN[kwargs["act_fn"]]
        # )

        self.learned_dynamic_steering = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)

        self.learned_static_steering = torch.nn.Parameter(
            torch.rand(self.embed_dim), requires_grad=True
        )

        # init learned static steering to 0
        self.learned_static_steering.data.fill_(0.0)

        # init learned dynamic steering to identity matrix
        self.learned_dynamic_steering.weight.data = torch.eye(self.embed_dim)

    def forward(self, base, source=None, subspaces=None):
        dynamic_steering = self.learned_dynamic_steering(base)
        output = base + self.learned_static_steering + dynamic_steering
        # output = base + torch.matmul(
        #     (self.act_fn(self.learned_source(base)) - rotated_base),
        #     self.rotate_layer.weight.T,
        # )
        return self.dropout(output.to(base.dtype))

    # def state_dict(self, *args, **kwargs):
    #     """
    #     Overwrite for data-efficiency.
    #     """
    #     state_dict = OrderedDict()
    #     for k, v in self.learned_source.state_dict().items():
    #         state_dict[k] = v
    #     state_dict["rotate_layer"] = self.rotate_layer.weight.data
    #     return state_dict

    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     """
    #     Overwrite for data-efficiency.
    #     """
    #     super().load_state_dict(state_dict, strict=False)
    #     overload_w = state_dict["rotate_layer"]
    #     overload_w_width = overload_w.shape[-1]
    #     self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = (
    #         overload_w
    #     )
    #     return


class LoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - rotated_base),
            self.rotate_layer.weight.T,
        )
        return self.dropout(output.to(base.dtype))

    def state_dict(self, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        state_dict = OrderedDict()
        for k, v in self.learned_source.state_dict().items():
            state_dict[k] = v
        state_dict["rotate_layer"] = self.rotate_layer.weight.data
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Overwrite for data-efficiency.
        """
        super().load_state_dict(state_dict, strict=False)
        overload_w = state_dict["rotate_layer"]
        overload_w_width = overload_w.shape[-1]
        self.rotate_layer.parametrizations.weight[0].base[:, :overload_w_width] = (
            overload_w
        )
        return


class NoreftIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj_layer = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=kwargs["add_bias"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"]
        ).to(kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.dropout = torch.nn.Dropout(
            kwargs["dropout"] if "dropout" in kwargs else 0.0
        )
        self.act_fn = (
            ACT2FN["linear"]
            if "act_fn" not in kwargs or kwargs["act_fn"] is None
            else ACT2FN[kwargs["act_fn"]]
        )

    def forward(self, base, source=None, subspaces=None):
        proj_base = self.proj_layer(base)
        output = base + torch.matmul(
            (self.act_fn(self.learned_source(base)) - proj_base), self.proj_layer.weight
        )
        return self.dropout(output.to(base.dtype))


class ConsreftIntervention(
    ConstantSourceIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = LowRankRotateLayer(self.embed_dim, kwargs["low_rank_dimension"])
        self.rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.learned_source = torch.nn.Parameter(
            torch.rand(kwargs["low_rank_dimension"]), requires_grad=True
        )

    def forward(self, base, source=None, subspaces=None):
        rotated_base = self.rotate_layer(base)
        output = base + torch.matmul(
            (self.learned_source - rotated_base), self.rotate_layer.weight.T
        )
        return output.to(base.dtype)
