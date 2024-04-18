import torch
from pyvene import (
    DistributedRepresentationIntervention,
    SourcelessIntervention,
    TrainableIntervention,
)


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
        # self.dropout = torch.nn.Dropout(
        # kwargs["dropout"] if "dropout" in kwargs else 0.0
        # )
        # self.act_fn = (
        #     ACT2FN["linear"]
        #     if "act_fn" not in kwargs or kwargs["act_fn"] is None
        #     else ACT2FN[kwargs["act_fn"]]
        # )

        self.learned_dynamic_steering = torch.nn.Linear(
            self.embed_dim, self.embed_dim, bias=False
        )
        self.learned_static_target = torch.nn.Parameter(
            torch.rand(
                self.embed_dim,
                dtype=kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16,
            ),
            requires_grad=True,
        )

        # init learned static steering to 0
        torch.nn.init.zeros_(self.learned_static_target)

        torch.nn.init.eye_(self.learned_dynamic_steering.weight)

        self.learned_dynamic_steering = self.learned_dynamic_steering.to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16
        )

        # self.learned_dynamic_steering.weight = self.learned_dynamic_steering.weight.to(
        #     torch.bfloat16
        # ).to("cuda:0")
        # self.learned_static_steering = self.learned_static_steering.to(
        #     torch.bfloat16
        # ).to("cuda:0")

        # # init learned dynamic steering to identity matrix
        # self.learned_dynamic_steering.weight.data = (
        #     torch.eye(self.embed_dim).to(torch.bfloat16).to("cuda:0")
        # )

    def forward(self, base, source=None, subspaces=None):
        dynamic_base = self.learned_dynamic_steering(base)
        # steered_output = self.learned_static_target + dynamic_steering

        steering_vector = self.learned_static_target - dynamic_base
        output = base + steering_vector
        # output = base + dynamic_steering
        # output = base + torch.matmul(
        #     (self.act_fn(self.learned_source(base)) - rotated_base),
        #     self.rotate_layer.weight.T,
        # )
        return output.to(base.dtype)
