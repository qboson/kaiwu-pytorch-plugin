import json
import torch
import os
import hashlib
import numpy as np


def text_to_embedding(text, dim=512):
    # deterministic pseudo-embedding from text using hash seed
    h = hashlib.sha256(text.encode('utf-8')).digest()
    # RandomState expects a 32-bit seed
    seed = int.from_bytes(h[:8], 'big') % (2 ** 32)
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=(dim,)).astype('float32')
    return torch.from_numpy(vec)


def prepare_from_json(json_path, out_pt_path):
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    examples = []
    for item in data:
        # best answer
        try:
            best = item.get('bon_best', {})
            best_ans = best.get('answer', None)
            if best_ans:
                emb = text_to_embedding(best_ans)
                examples.append({'embedding': emb, 'reward': 1.0})
        except Exception:
            pass
        # losers
        try:
            losers = item.get('loser_list', [])
            for l in losers:
                ans = l.get('answer', None)
                if ans:
                    emb = text_to_embedding(ans)
                    examples.append({'embedding': emb, 'reward': 0.0})
        except Exception:
            pass

    # save as list of dicts compatible with RewardEmbeddingDataset
    torch.save(examples, out_pt_path)
    print(f"Saved {len(examples)} examples to {out_pt_path}")


if __name__ == '__main__':
    import sys
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    # default source JSON path
    src = os.path.join(repo_root, 'RMB-Reward-Model-Benchmark', 'RMB_dataset', 'BoN_set', 'Harmlessness', 'S1.json')
    out = os.path.join(here, 'rmb_dataset.pt')
    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        out = sys.argv[2]
    prepare_from_json(src, out)
