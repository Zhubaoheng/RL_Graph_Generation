import os

from rdkit import Chem
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import time
import warnings

### packages for visualization
from analysis.rdkit_functions import compute_molecular_metrics
import torch
from torchmetrics import Metric, MetricCollection
from torch import Tensor
import swanlab
import torch.nn as nn
import numpy as np
import pandas as pd

# import fcd

from metrics.abstract_metrics import compute_ratios


class TrainMolecularMetrics(nn.Module):
    def __init__(self, remove_h):
        super().__init__()
        self.train_atom_metrics = AtomMetrics(remove_h)
        self.train_bond_metrics = BondMetrics()

    def forward(
        self,
        masked_pred_epsX,
        masked_pred_epsE,
        pred_y,
        true_epsX,
        true_epsE,
        true_y,
        log: bool,
    ):
        self.train_atom_metrics(masked_pred_epsX, true_epsX)
        self.train_bond_metrics(masked_pred_epsE, true_epsE)
        if log:
            to_log = {}
            for key, val in self.train_atom_metrics.compute().items():
                to_log["train/" + key] = val.item()
            for key, val in self.train_bond_metrics.compute().items():
                to_log["train/" + key] = val.item()
            try:
                if swanlab.run:
                    swanlab.log(to_log)
            except:
                pass

    def reset(self):
        for metric in [self.train_atom_metrics, self.train_bond_metrics]:
            metric.reset()

    def log_epoch_metrics(self):
        epoch_atom_metrics = self.train_atom_metrics.compute()
        epoch_bond_metrics = self.train_bond_metrics.compute()

        to_log = {}
        for key, val in epoch_atom_metrics.items():
            to_log["train_epoch/epoch" + key] = val.item()
        for key, val in epoch_bond_metrics.items():
            to_log["train_epoch/epoch" + key] = val.item()

        try:
            if swanlab.run:
                swanlab.log(to_log)
        except:
            pass

        for key, val in epoch_atom_metrics.items():
            epoch_atom_metrics[key] = f"{val.item() :.3f}"
        for key, val in epoch_bond_metrics.items():
            epoch_bond_metrics[key] = f"{val.item() :.3f}"

        return epoch_atom_metrics, epoch_bond_metrics


class SamplingMolecularMetrics(nn.Module):
    def __init__(self, dataset_infos, dataset_smiles, cfg, add_virtual_states=False):
        super().__init__()
        di = dataset_infos
        self.generated_n_dist = GeneratedNDistribution(di.max_n_nodes)
        num_node_types = (
            di.output_dims["X"] + 1 if add_virtual_states else di.output_dims["X"]
        )
        self.generated_node_dist = GeneratedNodesDistribution(num_node_types)
        num_edge_types = (
            di.output_dims["E"] + 1 if add_virtual_states else di.output_dims["E"]
        )
        self.generated_edge_dist = GeneratedEdgesDistribution(num_edge_types)
        self.generated_valency_dist = ValencyDistribution(di.max_n_nodes)
        self.cfg = cfg

        n_target_dist = di.n_nodes.type_as(self.generated_n_dist.n_dist)
        n_target_dist = n_target_dist / torch.sum(n_target_dist)
        self.register_buffer("n_target_dist", n_target_dist)

        node_target_dist = di.node_types.type_as(self.generated_node_dist.node_dist)
        node_target_dist = node_target_dist / torch.sum(node_target_dist)
        self.register_buffer("node_target_dist", node_target_dist)
        if add_virtual_states:
            node_target_dist = torch.cat([node_target_dist, torch.tensor([0.0])])

        edge_target_dist = di.edge_types.type_as(self.generated_edge_dist.edge_dist)
        edge_target_dist = edge_target_dist / torch.sum(edge_target_dist)
        self.register_buffer("edge_target_dist", edge_target_dist)
        if add_virtual_states:
            edge_target_dist = torch.cat([edge_target_dist, torch.tensor([0.0])])

        valency_target_dist = di.valency_distribution.type_as(
            self.generated_valency_dist.edgepernode_dist
        )
        valency_target_dist = valency_target_dist / torch.sum(valency_target_dist)
        self.register_buffer("valency_target_dist", valency_target_dist)
        self.n_dist_mae = HistogramsMAE(n_target_dist)
        self.node_dist_mae = HistogramsMAE(node_target_dist)
        self.edge_dist_mae = HistogramsMAE(edge_target_dist)
        self.valency_dist_mae = HistogramsMAE(valency_target_dist)

        self.train_smiles = dataset_smiles["train"]
        self.val_smiles = dataset_smiles["val"]
        self.test_smiles = dataset_smiles["test"]
        self.dataset_info = di

    def forward(
        self,
        molecules: list,
        ref_metrics,
        name,
        current_epoch,
        val_counter,
        local_rank,
        test=False,
        labels=None,
    ):
        stability, rdkit_metrics, all_smiles, to_log = compute_molecular_metrics(
            molecules, self.train_smiles, self.dataset_info, labels, self.cfg, test
        )

        if test and local_rank == 0:
            with open(r"final_smiles.txt", "w") as fp:
                for smiles in all_smiles:
                    # write each item on a new line
                    fp.write("%s\n" % smiles)
                print("All smiles saved")

            print(all_smiles)
            df = pd.DataFrame(all_smiles, columns=["SMILES"])
            df.to_csv("final_smiles.csv", index=False)
            print("All SMILES saved to CSV")

        # self.dataset_info.compute_fcd = True
        self.dataset_info.compute_fcd = False
        if self.dataset_info.compute_fcd:
            to_log["fcd"] = compute_fcd(
                val_smiles=self.test_smiles if test else self.val_smiles,
                generated_smiles=all_smiles,
                # val_smiles=self.test_smiles,
                # generated_smiles=self.train_smiles,
            )
        else:
            print("FCD computation is disabled. Skipping.")
            to_log["fcd"] = -1

        print("fcd", to_log["fcd"])

        print("Starting custom metrics")
        self.generated_n_dist(molecules)
        generated_n_dist = self.generated_n_dist.compute()
        self.n_dist_mae(generated_n_dist)

        self.generated_node_dist(molecules)
        generated_node_dist = self.generated_node_dist.compute()
        self.node_dist_mae(generated_node_dist)

        self.generated_edge_dist(molecules)
        generated_edge_dist = self.generated_edge_dist.compute()
        self.edge_dist_mae(generated_edge_dist)

        self.generated_valency_dist(molecules)
        generated_valency_dist = self.generated_valency_dist.compute()
        self.valency_dist_mae(generated_valency_dist)

        for i, atom_type in enumerate(self.dataset_info.atom_encoder.keys()):
            generated_probability = generated_node_dist[i]
            target_probability = self.node_target_dist[i]
            to_log[f"molecular_metrics/{atom_type}_dist"] = (
                generated_probability - target_probability
            ).item()

        for j, bond_type in enumerate(
            ["No bond", "Single", "Double", "Triple", "Aromatic"]
            # ["No bond", "Single", "Double", "Triple"]
        ):
            if j < len(generated_edge_dist):
                generated_probability = generated_edge_dist[j]
                target_probability = self.edge_target_dist[j]
                to_log[f"molecular_metrics/bond_{bond_type}_dist"] = (
                    generated_probability - target_probability
                ).item()

        for valency in range(6):
            generated_probability = generated_valency_dist[valency]
            target_probability = self.valency_target_dist[valency]
            to_log[f"molecular_metrics/valency_{valency}_dist"] = (
                generated_probability - target_probability
            ).item()

        n_mae = self.n_dist_mae.compute()
        node_mae = self.node_dist_mae.compute()
        edge_mae = self.edge_dist_mae.compute()
        valency_mae = self.valency_dist_mae.compute()

        ratios = compute_ratios(
            gen_metrics=to_log,
            ref_metrics=ref_metrics["test"] if test else ref_metrics["val"],
            metrics_keys=["fcd"] if self.dataset_info.compute_fcd else [],
        )
        to_log.update(ratios)

        try:
            if swanlab.run:
                swanlab.log(to_log)
                swanlab.log({
                    "summary/Gen n distribution": generated_n_dist.tolist(),
                    "summary/Gen node distribution": generated_node_dist.tolist(),
                    "summary/Gen edge distribution": generated_edge_dist.tolist(),
                    "summary/Gen valency distribution": generated_valency_dist.tolist()
                })

                swanlab.log(
                    {
                        "basic_metrics/n_mae": n_mae,
                        "basic_metrics/node_mae": node_mae,
                        "basic_metrics/edge_mae": edge_mae,
                        "basic_metrics/valency_mae": valency_mae,
                    }
                )
        except:
            pass

        if local_rank == 0:
            print("Custom metrics computed.")
        if local_rank == 0:
            valid_unique_molecules = rdkit_metrics[1]
            text_filename = f"graphs/{name}/valid_unique_molecules_e{current_epoch}_b{val_counter}.txt"
            if not os.path.exists(os.path.dirname(text_filename)):
                os.makedirs(os.path.dirname(text_filename))
            textfile = open(
                text_filename,
                "w",
            )
            textfile.writelines(valid_unique_molecules)
            textfile.close()
            print("Stability metrics:", stability, "--", rdkit_metrics[0])

        return to_log

    def reset(self):
        for metric in [
            self.n_dist_mae,
            self.node_dist_mae,
            self.edge_dist_mae,
            self.valency_dist_mae,
        ]:
            metric.reset()


def compute_fcd(val_smiles, generated_smiles):
    """smiles have must be a list of str"""

    print("Starting FCD computation")
    start = time.time()

    # not using fcd.canonical_smiles because both smiles are already in canonical form (result from the Chem.MolToSmiles)
    # filter out None values (not sanitizable molecules)
    generated_smiles = [smile for smile in generated_smiles if smile is not None]

    # supress warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            fcd_score = fcd.get_fcd(generated_smiles, val_smiles)
        except Exception as e:
            print(f"Error in FCD computation. Setting FCD to -1.")
            fcd_score = -1

    end = time.time()
    print("FCD computation time:", end - start, "FCD score is", fcd_score)

    return fcd_score


class GeneratedNDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "n_dist",
            default=torch.zeros(max_n + 1, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule
            n = atom_types.shape[0]
            self.n_dist[n] += 1

    def compute(self):
        return self.n_dist / torch.sum(self.n_dist)


class GeneratedNodesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_atom_types):
        super().__init__()
        self.add_state(
            "node_dist",
            default=torch.zeros(num_atom_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            atom_types, _ = molecule

            for atom_type in atom_types:
                assert (
                    int(atom_type) != -1
                ), "Mask error, the molecules should already be masked at the right shape"
                self.node_dist[int(atom_type)] += 1

    def compute(self):
        return self.node_dist / torch.sum(self.node_dist)


class GeneratedEdgesDistribution(Metric):
    full_state_update = False

    def __init__(self, num_edge_types):
        super().__init__()
        self.add_state(
            "edge_dist",
            default=torch.zeros(num_edge_types, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules):
        for molecule in molecules:
            _, edge_types = molecule
            mask = torch.ones_like(edge_types)
            mask = torch.triu(mask, diagonal=1).bool()
            edge_types = edge_types[mask]
            unique_edge_types, counts = torch.unique(edge_types, return_counts=True)
            for type, count in zip(unique_edge_types, counts):
                self.edge_dist[type] += count

    def compute(self):
        return self.edge_dist / torch.sum(self.edge_dist)


class MeanNumberEdge(Metric):
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("total_edge", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, molecules, weight=1.0) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            triu_edge_types = torch.triu(edge_types, diagonal=1)
            bonds = torch.nonzero(triu_edge_types)
            self.total_edge += len(bonds)
        self.total_samples += len(molecules)

    def compute(self):
        return self.total_edge / self.total_samples


class ValencyDistribution(Metric):
    full_state_update = False

    def __init__(self, max_n):
        super().__init__()
        self.add_state(
            "edgepernode_dist",
            default=torch.zeros(3 * max_n - 2, dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, molecules) -> None:
        for molecule in molecules:
            _, edge_types = molecule
            edge_types[edge_types == 4] = 1.5
            edge_types[edge_types == 5] = 0.0  # zero out virtual states
            valencies = torch.sum(edge_types, dim=0)
            unique, counts = torch.unique(valencies, return_counts=True)
            for valency, count in zip(unique, counts):
                self.edgepernode_dist[valency] += count

    def compute(self):
        return self.edgepernode_dist / torch.sum(self.edgepernode_dist)


class HistogramsMAE(MeanAbsoluteError):
    def __init__(self, target_histogram, **kwargs):
        """Compute the distance between histograms."""
        super().__init__(**kwargs)
        assert (target_histogram.sum() - 1).abs() < 1e-3
        self.target_histogram = target_histogram

    def update(self, pred):
        pred = pred / pred.sum()
        self.target_histogram = self.target_histogram.type_as(pred)
        super().update(pred, self.target_histogram)


class MSEPerClass(MeanSquaredError):
    full_state_update = False

    def __init__(self, class_id):
        super().__init__()
        self.class_id = class_id

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds[..., self.class_id]
        target = target[..., self.class_id]
        super().update(preds, target)


class HydroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class CarbonMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class NitroMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class OxyMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class FluorMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BoronMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class BrMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class ClMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class IodineMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class PhosphorusMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SulfurMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SeMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SiMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


# Bonds MSE


class NoBondMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class SingleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class DoubleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class TripleMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AromaticMSE(MSEPerClass):
    def __init__(self, i):
        super().__init__(i)


class AtomMetrics(MetricCollection):
    def __init__(self, dataset_infos):
        remove_h = dataset_infos.remove_h
        self.atom_decoder = dataset_infos.atom_decoder
        num_atom_types = len(self.atom_decoder)

        types = {
            "H": 0,
            "C": 1,
            "N": 2,
            "O": 3,
            "F": 4,
            "B": 5,
            "Br": 6,
            "Cl": 7,
            "I": 8,
            "P": 9,
            "S": 10,
            "Se": 11,
            "Si": 12,
        }

        class_dict = {
            "H": HydroMSE,
            "C": CarbonMSE,
            "N": NitroMSE,
            "O": OxyMSE,
            "F": FluorMSE,
            "B": BoronMSE,
            "Br": BrMSE,
            "Cl": ClMSE,
            "I": IodineMSE,
            "P": PhosphorusMSE,
            "S": SulfurMSE,
            "Se": SeMSE,
            "Si": SiMSE,
        }

        metrics_list = []
        for i, atom_type in enumerate(self.atom_decoder):
            metrics_list.append(class_dict[atom_type](i))

        super().__init__(metrics_list)


class BondMetrics(MetricCollection):
    def __init__(self):
        mse_no_bond = NoBondMSE(0)
        mse_SI = SingleMSE(1)
        mse_DO = DoubleMSE(2)
        mse_TR = TripleMSE(3)
        mse_AR = AromaticMSE(4)
        super().__init__([mse_no_bond, mse_SI, mse_DO, mse_TR, mse_AR])


if __name__ == "__main__":
    from torchmetrics.utilities import check_forward_full_state_property
