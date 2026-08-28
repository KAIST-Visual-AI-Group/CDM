"""Download the released checkpoints straight into each application's checkpoints/ directory.

    python scripts/download_checkpoints.py                # everything
    python scripts/download_checkpoints.py --apps dna     # one application

The Hub lays the files out by application; this maps them onto the per-application
`checkpoints/` directories the configs read from.
"""

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "jh27kim/cdm-checkpoints"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# app -> [(path in the HF repo, path under cdm/<app>/checkpoints/)]
FILES = {
    "texts_mdm": [
        ("toxicity/mdlm.ckpt", "mdlm.ckpt"),
        ("toxicity/cdm/twist_best.pt", "cdm/twist_best.pt"),
    ],
    "dna": [
        ("dna/mpra.ckpt", "mpra.ckpt"),
        ("dna/reward_oracle_ft.ckpt", "reward_oracle_ft.ckpt"),
        ("dna/reward_oracle_eval.ckpt", "reward_oracle_eval.ckpt"),
        ("dna/human_state_dict.h5", "human_state_dict.h5"),
        ("dna/cdm/twist_best.pt", "cdm/twist_best.pt"),
    ],
    "proteins": [
        ("proteins/cdm/twist_best.pt", "cdm/twist_best.pt"),
    ],
    "texts": [
        ("dllm/cdm/twist_best.pt", "cdm/twist_best.pt"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", nargs="+", choices=sorted(FILES), default=sorted(FILES))
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    for app in args.apps:
        for repo_path, local_name in FILES[app]:
            dest = os.path.join(ROOT, "cdm", app, "checkpoints", local_name)
            if os.path.exists(dest) and not args.force:
                print(f"[skip] {dest} exists")
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"[get ] {repo_path} -> {os.path.relpath(dest, ROOT)}")
            cached = hf_hub_download(repo_id=REPO_ID, filename=repo_path)
            # copy rather than symlink so the tree survives a hub cache clear
            shutil.copyfile(cached, dest)

    print("done")


if __name__ == "__main__":
    main()
