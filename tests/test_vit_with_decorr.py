import torch

from vit_pytorch.vit_with_decorr import DecorrelationLoss


def test_decorr_loss_samples_token_axis():
    tokens = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 4.0, 0.0]]
            ],
            [
                [[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 4.0], [5.0, 1.0, 2.0]]
            ],
        ]
    )
    seed = 42
    num_sampled = 2

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        sample_scores = torch.randn(tokens.shape[:-1])
        token_indices = sample_scores.argsort(dim=-1)[..., :num_sampled]
        selected_tokens = tokens.gather(
            -2, token_indices.unsqueeze(-1).expand(*token_indices.shape, tokens.shape[-1])
        )
        expected = DecorrelationLoss(sample_frac=1.0)(selected_tokens)

        torch.manual_seed(seed)
        actual = DecorrelationLoss(sample_frac=0.5)(tokens)

    torch.testing.assert_close(actual, expected)
