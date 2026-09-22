"""Token-level clipped objective for a joint grounding-answering trajectory."""


def joint_clipped_loss(
    current_grounding, old_grounding, grounding_mask,
    current_answer, old_answer, answer_mask,
    parent_indices, advantages, *, epsilon=.2, beta=0.,
    ref_grounding=None, ref_answer=None,
):
    import torch
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError("Require 0 < clip epsilon < 1 and beta >= 0")
    parents = torch.as_tensor(parent_indices, device=current_grounding.device)
    ground = current_grounding[parents]
    old_ground = old_grounding[parents].detach()
    ground_mask = grounding_mask[parents]
    advantage = advantages.detach()[:, None]

    def surrogate(current, old, reference):
        ratio = torch.exp(current - old.detach())
        loss = -torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
        if beta:
            if reference is None:
                raise ValueError("Nonzero beta requires reference likelihoods")
            delta = reference.detach() - current
            loss = loss + beta * (torch.exp(delta) - delta - 1)
        return loss

    ground_loss = surrogate(ground, old_ground, ref_grounding[parents] if ref_grounding is not None else None)
    answer_loss = surrogate(current_answer, old_answer, ref_answer)
    counts = (ground_mask.sum(1) + answer_mask.sum(1)).clamp_min(1)
    return ((ground_loss * ground_mask).sum(1) / counts
            + (answer_loss * answer_mask).sum(1) / counts).mean()
