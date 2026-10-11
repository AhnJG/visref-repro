"""InternVL3.5 loading, official image preprocessing and Thinking-mode prompt."""
import io

import torch

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---- official model-card preprocessing (dynamic tiling) ----------------------
def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(im) for im in images])


# ---- model -------------------------------------------------------------------
def load_model(model_id, revision, dtype=torch.float16, device='cuda'):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=dtype, trust_remote_code=True,
                                      low_cpu_mem_usage=True, use_flash_attn=False).eval().to(device)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True, use_fast=False)
    # InternVL remote code selects 'eager' when flash-attn is off; SDPA is numerically equivalent and
    # avoids materializing B x H x L x L attention for long thinking contexts.
    model.language_model.config._attn_implementation = 'sdpa'
    model.requires_grad_(False)
    return model, tok


def special_ids(tok, wait_text, probe_prompt='\n\n\\boxed'):
    ids = {name: tok.convert_tokens_to_ids(t) for name, t in [('think_end', '</think>'), ('eos', '<|im_end|>')]}
    ids['wait'] = tok.encode(wait_text, add_special_tokens=False)
    ids['probe_prompt'] = tok.encode(probe_prompt, add_special_tokens=False)
    texts = [tok.decode([i]) for i in range(len(tok))]
    ids.update(brace_ids(texts))
    return ids


def brace_ids(texts):
    """For the '\\boxed' probe. brace_text: id -> text of tokens containing '{' or '}' (brace-depth tracking).
    close_ids: tokens whose first non-space character is '}' -- at box depth 1 every one of them closes the box
    with nothing added, i.e. the same answer event, so they are merged into one outcome for the entropy."""
    return {'brace_text': {i: s for i, s in enumerate(texts) if '{' in s or '}' in s},
            'close_ids': frozenset(i for i, s in enumerate(texts) if s.lstrip().startswith('}'))}


def build_prompt(question, num_image_tokens):
    """internvl2_5 template (MPT style) with the R1 Thinking system prompt; same as model.chat()."""
    image = '<img>' + '<IMG_CONTEXT>' * num_image_tokens + '</img>'
    return (f'<|im_start|>system\n{R1_SYSTEM_PROMPT}<|im_end|>\n'
            f'<|im_start|>user\n{image}\n{question}<|im_end|>\n<|im_start|>assistant\n')


@torch.no_grad()
def encode_sample(model, tok, image, question, max_num=12):
    """Return (prompt embeddings [L, d], visual tokens V [N, d])."""
    lm = model.language_model
    pv = load_image(image, max_num=max_num).to(lm.device, dtype=lm.dtype)
    vis = model.extract_feature(pv).reshape(-1, lm.config.hidden_size)
    ids = tok(build_prompt(question, vis.shape[0]), return_tensors='pt').input_ids[0].to(lm.device)
    emb = lm.get_input_embeddings()(ids).clone()
    sel = ids == tok.convert_tokens_to_ids('<IMG_CONTEXT>')
    assert int(sel.sum()) == vis.shape[0]
    emb[sel] = vis.to(emb.dtype)
    return emb, vis


def load_mathvista(parquet_path):
    """MathVista testmini rows (HF AI4Math/MathVista parquet) with decoded PIL images."""
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(parquet_path)
    rows = {}
    for r in df.to_dict('records'):
        img = r['decoded_image']
        rows[str(r['pid'])] = {
            'pid': str(r['pid']), 'query': r['query'], 'answer': r['answer'],
            'question_type': r['question_type'], 'answer_type': r['answer_type'],
            'choices': None if r['choices'] is None else [str(c) for c in r['choices']],
            'precision': r['precision'], 'unit': r['unit'],
            'image': lambda b=img['bytes']: Image.open(io.BytesIO(b)).convert('RGB')}
    return rows
