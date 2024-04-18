import numpy as np
import torch
from activation_recording import apply_hooks
from tqdm import tqdm
from transformers.trainer import DataLoader


def get_unconditional_activations(
    model,
    tokenizer,
    max_samples,
):
    # model = model.model
    # generate unconditional samples starting with bos_token
    desired_samples = max_samples
    data = []
    bar = tqdm(total=desired_samples)
    print(
        f"generating {desired_samples} samples with max_length {model.config.max_length}"
    )
    while len(data) < desired_samples:
        new_data = model.generate(
            torch.tensor([[tokenizer.bos_token_id] * 4], device=model.device),
            do_sample=True,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            # top_k=50,
            num_return_sequences=50,
        )
        data.extend(new_data)
        bar.update(len(new_data))
    print(len(data))
    inputs = [tokenizer.decode(i, skip_special_tokens=True) for i in data]
    print(len(inputs), len(inputs[0]))
    print(
        "avg length of unconditional samples:",
        np.mean([len(i) for i in inputs]),
    )

    uncond_hooks, uncond_handles = apply_hooks(
        model, "model.layers.{}", debug=False, just_mean=False
    )

    with torch.inference_mode():
        BATCH_SIZE = 1
        batch = []
        # trackers = [IterativeStatsTracker() for i in range(len(recorders))]
        for i, input_text in tqdm(enumerate(inputs)):
            # if len(batch) < BATCH_SIZE: # TODO: how to batch inputs w/ truncation / padding and ensure only correct activations get tracked?
            #     batch.append(input_text)
            #     continue
            # last_running_mean = hijacker.activation_center_of_mass()
            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                return_attention_mask=False,
                truncation=True,
                # max_length=model.config.max_length or 1024,
                max_length=1024,
            )["input_ids"].to(model.device)

            if input_ids.dtype != torch.long:
                print(f"Skipping {i} due to dtype {input_ids.dtype}")
                print(input_ids)
                continue
            # with torch.autograd.profiler.profile(use_cuda=True) as prof:
            # Your code segment here

            _ = model(input_ids, return_dict=True)

            # hiddenstates = out["hidden_states"]
            # # print(hiddenstates)
            # for l, layer_actvations in enumerate(hiddenstates[1:]):
            #     trackers[l].update(layer_actvations[:, -1:, :])
            #     # print(layer_actvations[:, -1:, :].shape)

            # print(trackers[-4].running_mean[0, :15])
            # print(recorders[-4]._running_output_com[0, :15])
    # print(hiddenstates.shape)
    # print(prof)
    for handle in uncond_handles:
        handle.remove()

    print(
        "MAX MEM AFTER UNCoND",
        torch.cuda.max_memory_allocated(model.device) / 1024 / 1024 / 1024,
    )
    return uncond_hooks


def get_conditional_activations(
    model,
    tokenizer,
    train_dataset,
    limit_reftinit_samples,
):
    hooks, handles = apply_hooks(model, "model.layers.{}", just_mean=False, debug=False)

    print(hooks)
    with torch.inference_mode():
        model.intervention_types = "TEST"
        print(model)
        print(model)
        print(model.device)
        # reft_model.eval()
        tran_dataloader = DataLoader(train_dataset, batch_size=1)
        limit = limit_reftinit_samples
        counter = 0
        for sample in tqdm(tran_dataloader):
            # print(sample)
            # move sample to modle device
            sample = {
                k: v.to(model.device)
                for k, v in sample.items()
                if k != "intervention_locations"
            }
            model(
                input_ids=sample["input_ids"],
                attention_mask=sample["attention_mask"],
                labels=sample["labels"],
            )

            counter += 1
            if counter > limit:
                break

    print(hooks)

    for handle in handles:
        handle.remove()
    return hooks


def project_out(data, direction):
    """Project out the component along the given direction from the data"""
    projection = torch.matmul(data, direction.unsqueeze(-1))
    return data - projection * direction


def project_data(data, direction):
    return torch.matmul(data, direction.unsqueeze(-1)).squeeze()


def calculate_criterion(group1_projected, group2_projected):
    mean1 = torch.mean(group1_projected)
    mean2 = torch.mean(group2_projected)
    return (mean2 - mean1) ** 2


def dca_with_criterion_change(group1, group2, num_k, change_threshold=0.01):
    directions = []
    current_group1 = group1.clone()
    current_group2 = group2.clone()

    previous_criterion_value = 0
    significant_change = True

    for qq in range(num_k):
        mean1 = torch.mean(current_group1, dim=0)
        mean2 = torch.mean(current_group2, dim=0)

        direction = mean2 - mean1
        direction = direction / torch.norm(direction)
        directions.append(direction)

        # Project data and calculate new criterion value
        proj1 = project_data(group1, direction)
        proj2 = project_data(group2, direction)
        criterion_value = calculate_criterion(proj1, proj2)

        # Check if the change in criterion value is significant
        if len(directions) > 1:  # Skip the first iteration
            change = criterion_value - previous_criterion_value
            print(
                "criterion",
                qq,
                change,
                criterion_value,
                previous_criterion_value,
                abs(change) < change_threshold,
            )
            if abs(change) < change_threshold:
                significant_change = False

        previous_criterion_value = criterion_value

        # Project out the current direction
        current_group1 = project_out(current_group1, direction)
        current_group2 = project_out(current_group2, direction)

    return directions
