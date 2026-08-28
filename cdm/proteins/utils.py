import os

from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.core.import_pose import pose_from_pdbstring
import pyrosetta.rosetta.core.pose as pose

def pose_read_pdb(pdb_file, filter_by_CA=True):
    """
    pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    assert isinstance(pdb_file, str)
    if pdb_file.endswith('.pdb'):
        pose_pdb = pose_from_pdb(pdb_file)
    else:
        pose_pdb = pose.Pose()
        pose_from_pdbstring(pose_pdb, pdb_file)

    if not filter_by_CA:
        return pose_pdb
    filtered_pose = pose.Pose()
    for i in range(1, pose_pdb.total_residue() + 1):
        if pose_pdb.residue(i).has("CA"):
            filtered_pose.append_residue_by_bond(pose_pdb.residue(i))

    return filtered_pose

def save_fasta(
    save_name,
    output_results,
    struct_tokens=False,
    headers=None,
    continue_write=False,
):
    fp_save = (
        open(save_name, "w") if not continue_write else open(save_name, "a")
    )
    for idx, seq in enumerate(output_results):
        if headers is not None:
            fp_save.write(f">{headers[idx]}\n")
        else:
            fp_save.write(f">SEQUENCE_{idx}\n")
        seq = seq.split(" ")
        if struct_tokens:
            fp_save.write(f"{','.join(seq)}\n")
        else:
            fp_save.write(f"{''.join(seq)}\n")
    fp_save.close()

def save_results(
    tokenizer,
    struct_tokenizer,
    save_dir,
    task,
    outputs,
    headers=None,
    save_pdb=True,
    continue_write=False,
):
    # save to fasta
    os.makedirs(save_dir, exist_ok=True)
    if headers is None:
        pdb_dir = os.path.join(save_dir, "pdb")
        if os.path.exists(pdb_dir):
            num_existing = len(os.listdir(pdb_dir))
        else:
            num_existing = len(os.listdir(save_dir)) if os.path.exists(save_dir) else 0
        headers = [f"sample_{num_existing + i}" for i in range(len(outputs["output_tokens"]))]

    if task in ["sequence_generation"]:
        aatype_tokens = outputs["output_tokens"]
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=aatype_fasta_path,
            output_results=aatype_strings,
            headers=headers,
            continue_write=continue_write,
        )

    elif task in [
        "backbone_generation",
        "co_generation",
        "folding",
        "inverse_folding",
    ]:
        output_tokens = outputs["output_tokens"]
        struct_tokens, aatype_tokens = output_tokens.chunk(2, dim=-1)
        struct_token_fasta_path = os.path.join(save_dir, "struct_token.fasta")
        aatype_fasta_path = os.path.join(save_dir, "aatype.fasta")
        struct_tokens_strings = list(
            map(
                lambda s: ",".join(s.split()),
                tokenizer.batch_decode(
                    struct_tokens, skip_special_tokens=True
                ),
            )
        )
        aatype_strings = list(
            map(
                lambda s: "".join(s.split()),
                tokenizer.batch_decode(
                    aatype_tokens, skip_special_tokens=True
                ),
            )
        )
        save_fasta(
            save_name=struct_token_fasta_path,
            output_results=struct_tokens_strings,
            headers=headers,
            continue_write=continue_write,
        )
        save_fasta(
            save_name=aatype_fasta_path,
            output_results=aatype_strings,
            headers=headers,
            continue_write=continue_write,
        )
        if save_pdb:
            pdb_save_dir = os.path.join(save_dir, "pdb")
            os.makedirs(pdb_save_dir, exist_ok=True)
            for idx, (header, aatype_str, struct_tokens_str) in enumerate(
                zip(headers, aatype_strings, struct_tokens_strings)
            ):
                (
                    aatype_tensor,
                    struct_tokens_tensor,
                ) = struct_tokenizer.string_to_tensor(
                    aatype_str, struct_tokens_str
                )
                if "final_struct_feature" in outputs:
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens=outputs["final_struct_feature"][idx][
                            None
                        ],
                        res_mask=outputs["res_mask"][idx][None],
                    )
                else:
                    decoder_out = struct_tokenizer.detokenize(
                        struct_tokens_tensor
                    )

                decoder_out["aatype"] = aatype_tensor
                decoder_out["header"] = [header]

                struct_tokenizer.output_to_pdb(
                    decoder_out, output_dir=pdb_save_dir
                )
    else:
        raise NotImplementedError

    return