# Reward functions for toxic text generation.
#
# Given reward   : log p(toxic) under SkolkovoInstitute/roberta_toxicity_classifier.
# Heldout reward : fraction of samples classified toxic by the multilingual
#                  textdetox/xlmr-large-toxicity-classifier, which is never used for scaling.
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

MODELS = {}

HELDOUT_MODEL_NAME = "textdetox/xlmr-large-toxicity-classifier"


def logmeanexp(scores):
    if not isinstance(scores, torch.Tensor):
        tensor_scores = torch.tensor(scores)
    else:
        tensor_scores = scores

    # Exception handling for empty tensor
    if tensor_scores.shape[-1] == 0:
        return -100.0

    result = torch.logsumexp(tensor_scores, dim=-1) - np.log(tensor_scores.shape[-1])

    if not isinstance(scores, torch.Tensor):
        return result.tolist()
    else:
        return result


def _compute_roberta_score(
    model,
    tokenizer,
    texts,
    label_idx,
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
):
    """
    Compute log mean probability of the label for each text.
    """
    # get individual texts
    all_texts = []
    original_indices = []

    for i, text in enumerate(texts):
        # currently batches within single generation
        split_text = [t for t in text.split(delimiter) if t.strip()]
        if just_first:
            split_text = split_text[:1]

        all_texts.extend(split_text)
        original_indices.extend([i] * len(split_text))

    # batch the input
    batched_input = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        batched_input.append(batch)

    # get scores
    all_scores = []
    for batch in batched_input:
        tokenized = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        outputs = model(**tokenized)
        # use log softmax so that the reward lives on a log scale
        scores = torch.nn.functional.log_softmax(outputs.logits, dim=-1)[
            :, label_idx
        ].tolist()
        all_scores.extend(scores)

    # average the log scores
    unreduced_per_text_scores = [[] for _ in range(len(texts))]
    for i, score in zip(original_indices, all_scores):
        unreduced_per_text_scores[i].append(score)

    avg_scores = [logmeanexp(scores) for scores in unreduced_per_text_scores]
    return avg_scores, unreduced_per_text_scores


def toxicity_score(
    texts,
    label='positive',
    device='cuda',
    delimiter='<|endoftext|>',
    just_first=True,
    batch_size=8,
    max_length=512,
):
    '''Get toxicity score for a list of texts, each can have multiple documents separated by delimiter'''

    global MODELS

    key = 'toxicity'
    if key not in MODELS:
        MODELS[key] = {
            'tokenizer': RobertaTokenizer.from_pretrained(
                'SkolkovoInstitute/roberta_toxicity_classifier'
            ),
            'model': RobertaForSequenceClassification.from_pretrained(
                'SkolkovoInstitute/roberta_toxicity_classifier'
            ),
        }
        MODELS[key]['model'].eval()
        MODELS[key]['model'].to(device)

    tokenizer = MODELS[key]['tokenizer']
    model = MODELS[key]['model']
    label_to_id = {'positive': 1, 'negative': 0}

    label_idx = label_to_id[label]

    return _compute_roberta_score(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        label_idx=label_idx,
        device=device,
        delimiter=delimiter,
        just_first=just_first,
        batch_size=batch_size,
        max_length=max_length,
    )


def compute_reward_scores(texts, reward_name, reward_label, args):
    """Dispatcher for specific reward model calls."""
    if reward_name == 'toxicity':
        return toxicity_score(
            texts=texts,
            label=reward_label,
            max_length=args.seq_len,
            batch_size=args.reward_eval_batch_size,
        )[0]
    raise ValueError(f'Unknown reward function: {reward_name}')


def evaluate_generation(output_ids, tokenizer, args):
    """
    Decodes and computes rewards for generated sequences.
    Handles batching internally to prevent OOM in the reward model.
    """
    assert output_ids.shape[1] == args.gen_length, f"output_ids.shape[1] {output_ids.shape[1]} != args.gen_length {args.gen_length}"
    sequences = output_ids[:, :args.gen_length]

    decoded_texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    if args.reward_eval_batch_size is None:
        reward_eval_batch_size = args.batch_size
    else:
        reward_eval_batch_size = args.reward_eval_batch_size

    all_scores = []
    for i in range(0, len(decoded_texts), reward_eval_batch_size):
        batch_texts = decoded_texts[i : i + reward_eval_batch_size]
        all_scores.extend(
            compute_reward_scores(
                texts=batch_texts,
                reward_name=args.reward_name,
                reward_label=args.reward_label,
                args=args,
            )
        )

    return all_scores


@torch.no_grad()
def heldout_reward(prompt_and_texts, device='cuda', label_idx=1):
    """Heldout reward: toxic-classification accuracy under the multilingual classifier.

    One text at a time and no padding, matching the accuracy protocol the paper numbers were
    produced with.
    """
    global MODELS

    key = 'toxicity_heldout'
    if key not in MODELS:
        MODELS[key] = {
            'tokenizer': AutoTokenizer.from_pretrained(HELDOUT_MODEL_NAME),
            'model': AutoModelForSequenceClassification.from_pretrained(HELDOUT_MODEL_NAME),
        }
        MODELS[key]['model'].eval()
        MODELS[key]['model'].to(device)

    tokenizer = MODELS[key]['tokenizer']
    model = MODELS[key]['model']

    hits = []
    for text in prompt_and_texts:
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        pred = model(**encoded).logits.argmax(dim=-1).view(-1).tolist()
        hits.extend([1 if p == label_idx else 0 for p in pred])

    return sum(hits) / len(hits)
