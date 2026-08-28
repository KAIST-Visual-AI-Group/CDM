"""Rewards for protein designability.

Given reward   : negative self-consistency RMSD between the generated structure and the ESMFold
                 prediction of the generated sequence (higher is better).
Heldout reward : scTM, the same protocol with TM-score in place of RMSD.
"""

import os
import shutil
import sys
import time
import warnings

import esm
import pyrosetta
import torch
from biotite.structure import rmsd, superimpose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import save_results

# Suppress transformers warnings
warnings.filterwarnings(
    'ignore', message='Some weights of the model checkpoint.*were not used when initializing.*')


def pdb_to_crmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    Compute the minimum RMSD between two PDB structures after superposition.
    If backbone=True, only consider backbone atoms (N, CA, C).
    Returns negative RMSD (higher = better).
    """
    from biotite.structure.io.pdb import PDBFile

    ori_pdb = PDBFile.read(ori_pdb_file)
    gen_pdb = PDBFile.read(gen_pdb_file)
    ori_atoms = ori_pdb.get_structure(model=1)
    gen_atoms = gen_pdb.get_structure(model=1)

    if backbone:
        ori_atoms = ori_atoms[(ori_atoms.atom_name == "N") |
                              (ori_atoms.atom_name == "CA") |
                              (ori_atoms.atom_name == "C")]
        gen_atoms = gen_atoms[(gen_atoms.atom_name == "N") |
                              (gen_atoms.atom_name == "CA") |
                              (gen_atoms.atom_name == "C")]

    # Truncate to the shorter length to handle length mismatches
    min_len = min(len(ori_atoms), len(gen_atoms))
    ori_atoms = ori_atoms[:min_len]
    gen_atoms = gen_atoms[:min_len]

    # Superimpose and compute RMSD
    gen_superimposed, _ = superimpose(ori_atoms, gen_atoms)
    return -rmsd(ori_atoms, gen_superimposed)

def pdb_to_sctm(ori_pdb_file, gen_pdb_file):
    """TM-score between two PDB structures, averaged over both normalisations.

    Heldout reward for protein designability: identical self-consistency protocol as
    ``pdb_to_crmsd`` with TM-score substituted for RMSD, so it needs no extra folding pass
    beyond the one that already produced ``gen_pdb_file``.
    """
    import tmtools
    from tmtools.io import get_residue_data, get_structure

    try:
        chain_a = next(get_structure(str(ori_pdb_file)).get_chains())
        chain_b = next(get_structure(str(gen_pdb_file)).get_chains())
        coords_a, seq_a = get_residue_data(chain_a)
        coords_b, seq_b = get_residue_data(chain_b)
        result = tmtools.tm_align(coords_a, coords_b, seq_a, seq_b)
        return float((result.tm_norm_chain1 + result.tm_norm_chain2) / 2)
    except Exception as e:
        print(f"  [warn] TM-score failed on {ori_pdb_file} <-> {gen_pdb_file}: {e}")
        return float("nan")


class FoldReward():
    def __init__(self, tokenizer, struct_tokenizer, folding_model="esm", device="cuda"):
        self.tokenizer = tokenizer
        self.struct_tokenizer = struct_tokenizer
        self.device = device
        if folding_model == "esm":
            self.folding_model = esm.pretrained.esmfold_v1().to(self.device)
        else:
            raise ValueError(f"Unknown folding model: {folding_model}")

        self.folding_model.eval()

        pyrosetta.init(options="-mute all")

    def score_sequences(self, sequences, ori_pdb_path, output_dir, log_time=False, also_tm=False):
        if log_time:
            start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        sequences = [seq.replace("X", "A") for seq in sequences]
        
        esmf_outputs = self.folding_model.infer(sequences, num_recycles=4)
        pdb_outputs = self.folding_model.output_to_pdb(esmf_outputs)
        
        crmsds, tms = [], []
        for i, pdb_output in enumerate(pdb_outputs):
            esmf_sample_path = os.path.join(output_dir, "pdb", f"folded_{i}.pdb")
            with open(esmf_sample_path, "w") as f:
                f.write(pdb_output)

            # Handle either a single path (str) or a list of paths
            ref_path = ori_pdb_path[i] if isinstance(ori_pdb_path, list) else ori_pdb_path
            crmsd = pdb_to_crmsd(ref_path, esmf_sample_path, backbone=True)
            crmsds.append(crmsd)
            if also_tm:
                tms.append(pdb_to_sctm(ref_path, esmf_sample_path))

        if log_time:
            end_time = time.time()
            print(f"Folding time: {end_time - start_time}")

        if also_tm:
            return crmsds, tms
        return crmsds

    def __call__(self, input_seq):
        return self._score(input_seq, also_tm=False)

    def crmsd_and_sctm(self, input_seq):
        """Given reward (-scRMSD) and heldout reward (scTM) from one folding pass."""
        return self._score(input_seq, also_tm=True)

    def _score(self, input_seq, also_tm=False):
        # Decode sequences from tokens to amino acid sequences
        decoded_sequences = self.tokenizer.batch_decode(
            input_seq, skip_special_tokens=True
        )
        output_dir = f"./tmp/outputs_folded_{time.time()}"
        os.makedirs(output_dir, exist_ok=True)
        outputs = {"output_tokens": input_seq}
        save_results(
            outputs=outputs,
            save_dir=output_dir,
            task="co_generation",
            tokenizer=self.tokenizer,
            struct_tokenizer=self.struct_tokenizer,
            headers=None,
            save_pdb=True,
            continue_write=False,   
        )

        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned_sequences = [''.join([aa for aa in sequence if aa in valid_aas]) for sequence in decoded_sequences]

        # Get batched crmsds scores
        scored = self.score_sequences(
            cleaned_sequences,
            ori_pdb_path=[f"{output_dir}/pdb/sample_{i}.pdb" for i in range(len(cleaned_sequences))],
            output_dir=output_dir,
            also_tm=also_tm,
        )
        crmsds, tms = scored if also_tm else (scored, None)

        shutil.rmtree(output_dir)

        crmsds = torch.tensor(crmsds, dtype=torch.float32, device=self.device)
        if also_tm:
            return crmsds, torch.tensor(tms, dtype=torch.float32, device=self.device)
        return crmsds