import time

import torch
from torch.utils.hooks import RemovableHandle


class IterativeStatsTracker:
    def __init__(self, debug=False) -> None:
        self.n = 0
        self.running_var_sum = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self.running_mean = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self.debug = debug

    def update(self, new_activations: torch.Tensor):
        current_n = new_activations.shape[1]
        current_mean = new_activations.to(torch.float32).mean(dim=1).detach().cpu()
        total_n = self.n + current_n

        """Iterative mean calculation optimized for numerical stability."""
        self.running_mean = (
            self.running_mean
            + (current_n * (current_mean - self.running_mean)) / total_n
        )

        """Welford's algorithm for online variance calculation."""
        delta_1 = current_mean - self.running_mean
        delta_2 = current_mean - self.running_mean
        self.running_var_sum = self.running_var_sum + current_n * delta_1 * delta_2

        self.n += current_n

        if self.debug:
            num_update_is_zero = (
                ((current_n / (self.n + current_n)) * current_mean) == 0.0
            ).sum()
            if num_update_is_zero > 0:
                print(
                    f"new running com norm:",
                    self.running_mean.norm(),
                    ((self.n / (self.n + current_n)) * self.running_mean).norm(),
                    ((current_n / (self.n + current_n)) * current_mean).norm(),
                )
                print(
                    "current mean norm:",
                    current_mean[0].norm(),
                    "update is zero:",
                    (((current_n / (self.n + current_n)) * current_mean) == 0.0).sum(),
                )
            # print("update norms:", (self._running_output_com / current_n).norm(), (current_mean / self._n).norm())
        # print(f"new running {track} com:", self._running_output_com)
        # print("current mean:", current_mean, "n:", current_n, "total n:", self._n)
    def reset(self):
        self.n = 0
        self.running_var_sum = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self.running_mean = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    
    def get_mean(self):
        return self.running_mean
    
    def get_var(self):
        return self.running_var_sum / self.n

    def get_std(self):
        return torch.sqrt(self.get_var())


def apply_hooks(
    model: torch.nn.Module, target_fqn_template: str, just_mean=True, debug=False
) -> tuple[list["Recorder"], list[RemovableHandle]]:
    hooks = []
    handles = []
    # target_fqn_template should be a FQN template string with a single integer format specifier that gets replaced by the layer number
    try:
        while True:
            target_fqn = target_fqn_template.format(len(hooks))
            hook = Recorder(target_fqn, just_mean=just_mean, debug=debug)
            handle = model.get_submodule(target_fqn).register_forward_hook(hook, prepend=True)
            hooks.append(hook)
            handles.append(handle)

            if "{}" not in target_fqn_template:
                break
    except AttributeError:
        print(f"Reached end of layers at {len(hooks)}")

    return hooks, handles


def register_activation_recording_hook(
    model: torch.nn.Module, target_fqn: str, just_mean=True, debug=False
):
    hook = Recorder(target_fqn, just_mean=just_mean, debug=debug)
    handle = model.get_submodule(target_fqn).register_forward_hook(hook)
    return hook, handle


class Recorder:
    def __init__(self, target_fqn: str, just_mean=True, debug=False):
        self.target_fqn = target_fqn

        self.output_captured = []
        self.output_captured_norms = []

        self.just_mean = just_mean
        # if self.just_mean:
        self.iterative_stat_tracker = IterativeStatsTracker(debug=debug)

        self.debug = debug
        if debug:
            self.timings = []
            print("Recorder initialized")

    def __repr__(self) -> str:
        if self.just_mean:
            return f"Recorder({self.target_fqn=}, {self.just_mean=}, {self.iterative_stat_tracker.running_mean=}, {self.iterative_stat_tracker.running_var_sum=}, {self.iterative_stat_tracker.n=}, {self.debug=})"
        return f"Recorder({self.target_fqn=}, {self.just_mean=}, {self.debug=})"

    def __call__(self, module: torch.nn.Module, args, outputs):
        assert outputs is not None
        t0 = time.perf_counter()
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        elif isinstance(outputs, dict):
            outputs = outputs["hidden_states"]
        assert isinstance(outputs, torch.Tensor)
        seq_len = outputs.shape[1]

        self.iterative_stat_tracker.update(outputs)
        # self.step_activation_com(outputs, track="output")
        if not self.just_mean:
            self.output_captured.append(outputs.detach().cpu())

        norms = outputs.view(-1, seq_len).norm(dim=-1)
        # print(norms.shape)
        self.output_captured_norms.extend(norms.tolist())

        if self.debug:
            self.timings.append(time.perf_counter() - t0)
        if len(outputs) % 1000 == 0 and self.debug:
            print(
                f"recording {len(outputs)} outputs with avg. overhead of {sum(self.timings) / len(self.timings)}s"
            )

    def activation_center_of_mass(self):
        if self.just_mean is False:
            if len(self.output_captured) > 0:
                return torch.stack(
                    [act.view(-1, act.shape[-1]) for act in self.output_captured]
                ).mean(dim=0)
            else:
                return torch.tensor(0.0).cpu()
        return self.iterative_stat_tracker.running_mean

    def reset(self):
        self.output_captured = []
        self.output_captured_norms = []
        self.iterative_stat_tracker.reset()
