#!/usr/bin/env python
# coding: utf-8

# # LIT-PCBA 
# this dataset is a proposed "more difficult" test than DUD-E. I used a 75/25 (sklearn default) stratified split. The dataset is heavily imbalanced, like DUD-E.
# 
# - UPDATE: I'm using the AVE split now

# In[1]:


import pickle
from sklearn.metrics import roc_auc_score
import numpy as np
import seaborn as sns
import torch
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import sys
# sys.path.append('..')
sys.path.insert(0, "/g/g13/jones289/workspace/hd-cuda-master")
# print(sys.path)
import hdpy
import hdpy.ecfp
# from hdpy.analysis import load_pkl
from hdpy.metrics import compute_enrichment_factor
from pathlib import Path


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=17)
plt.rc('figure', titlesize=20)


# SEED=125
SEED=5
# SEED=6
# SEED=7
# SEED=8
# SEED=2

#TODO: rename to result_p 
data_p = Path(f"/p/vast1/jones289/hd_results/{SEED}")

lit_pcba_full_data_p = Path(f"/p/vast1/jones289/lit_pcba/lit_pcba_full_data")


# In[2]:


green_color_pal = sns.color_palette("Greens", 10)
blue_color_pal = sns.color_palette("Blues", 10)
rocket_color_pal = sns.color_palette("rocket", 10)


# In[3]:


# model_dict = {
#     "molehd-bpe": ("MoleHD-BPE", rocket_color_pal[4]),
#     "hdbind-rp-molformer": ("HDBind+MolFormer-10k", green_color_pal[6]),
#     "hdbind-rp-molformer-100k": ("HDBind+MolFormer-100k", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-100": ("HDBind+ECFP-100", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-1k": ("HDBind+ECFP-1k", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1": ("HDBind+ECFP-10k", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-100k": ("HDBind+ECFP-100k", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-1m": ("HDBind+ECFP-1m", green_color_pal[6]),    
#     "mlp-1024-1": ("MLP", blue_color_pal[7]),
#     "Vina": ("Vina", "salmon"),
# }
model_dict = {
    "molehd-bpe": ("MoleHD-BPE", rocket_color_pal[4]),
    "hdbind-rp-ecfp-1024-1": ("HDB-ECFP", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-100k": ("HDB-ECFP-100k", green_color_pal[6]),
    "hdbind-rp-molclr":  ("HDB-MolCLR", green_color_pal[6]),
#     "hdbind-rp-molclr-100k":  ("HDB-MolCLR-100k", green_color_pal[6]),
    "hdbind-rp-molformer": ("HDB-MolFormer", green_color_pal[6]),
#     "hdbind-rp-molformer-100k": ("HDB-MolFormer-100k", green_color_pal[6]),
    "hdbind-rp-molformer-ecfp-combo": ("HDB-MolFormer+ECFP", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-100": ("HDBind+ECFP-100", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-1k": ("HDBind+ECFP-1k", green_color_pal[6]),
#     "hdbind-rp-ecfp-1024-1-1m": ("HDBind+ECFP-1m", green_color_pal[6]),    
    "mlp-1024-1": ("MLP", blue_color_pal[7]),
    "Vina": ("Vina", "salmon"),
}


# # LIT-PCBA Results

# export CONDA_ROOT=/usr/workspace/jones289/anaconda3-power #installed using the LC installer script
# export PATH=$CONDA_ROOT/bin:$PATH
# export CONDA_EXE=$CONDA_ROOT/bin/conda
# ml load gcc/11
# export PYTHONPATH=$PWD:$PYTHONPATH
# source activate /usr/workspace/jones289/anaconda3-power/envs/opence-1.8.0

# In[4]:


from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score


def compute_metrics(y_pred, y_score, y_true, p):
#     import pdb
#     pdb.set_trace()


    if y_score.squeeze().ndim == 2:
        enrich = float(compute_enrichment_factor(scores=y_score[:, 1], labels=y_true, n_percent=p))
        roc = roc_auc_score(y_score=y_score[:, 1], y_true=y_true)
    else:
        enrich = float(compute_enrichment_factor(scores=y_score, labels=y_true, n_percent=p))
        roc = roc_auc_score(y_score=y_score, y_true=y_true)

    
    
    return {"precision": precision_score(y_pred=y_pred, y_true=y_true, zero_division=0),
            "recall": recall_score(y_pred=y_pred, y_true=y_true),
           "f1": f1_score(y_pred=y_pred, y_true=y_true, zero_division=0),
            "enrich": enrich,            
           "roc": roc
           }


def aggregate_results(dataset, split, target_list=None):
    assert target_list is not None
#     import pdb
#     pdb.set_trace() 
    
#     if split == "random":
#         raise NotImplementedError # rename the files with "-random" to be consistent
        
    
    model_metric_dict = {"model": [], "enrich": [], "p":[], "train_time":[], "test_time": [], "target": [],
                        "seed": [], "recall": [], "precision": [], "f1": [], 
#                          "smiles": []
                        }


    for model_name, model_tup in tqdm(model_dict.items(), total=len(model_dict), position=0):
                        

        data_path_list = list(data_p.glob(f"{model_name}.{dataset}-*-{split}*.pkl"))
#         print(f"{model_name}\t{dataset}\t{len(model_tup)}\t {len(data_path_list)}")

        
        for path in tqdm(data_path_list, total=len(data_path_list), position=1):
            target = path.name.split(".")[1].split("-")[-2]
            print(target, path, model_name)

            if target in target_list or target_list == "all":
                pass
            else:
                continue
            
            model_data_dict = torch.load(path)
            
            # apparently we have duplicates in the test set...
            

            for seed in range(len(model_data_dict['trials'])):

    
                
                trial_dict = model_data_dict['trials'][seed]
            
            

                for p in [.01, .1]:

                    try:

                        
                        y_pred = trial_dict["y_pred"]
                        y_score = trial_dict["eta"]
                        
                        if model_name == "mlp":
                            y_score = y_score[:, 1]
                        
                        
#                         y_true = model_data_dict["y_test"]
                        y_true = trial_dict["y_true"]


                        # TODO: there are some issues with the number of smiles not lining up with the number of predictions/etc...
#                         import pdb
#                         pdb.set_trace()
                        print(f"y_pred: {y_pred.shape}\ty_score: {y_score.shape}\ty_true: {y_true.shape}")

                        
                        
                        metrics = compute_metrics(y_pred=y_pred, 
                                                  y_score=y_score, 
                                                  y_true=y_true,
                                                 p=p)

                        
                        
                        
                        
                        
                        
                        model_metric_dict["target"].append(target)
                        model_metric_dict["test_time"].append(trial_dict["test_time"])
                        model_metric_dict["enrich"].append(metrics["enrich"])
                        model_metric_dict["p"].append(p)
                        model_metric_dict["seed"].append(seed)
                        model_metric_dict["precision"].append(metrics["precision"])
                        model_metric_dict["recall"].append(metrics["recall"])
                        model_metric_dict["f1"].append(metrics["f1"])
                        model_metric_dict["model"].append(model_name)
#                         model_metric_dict["smiles"].append(model_data_dict["smiles_test"])
                        
                    except Exception as e:
                        print(e)


    model_metric_df = pd.DataFrame({key: value for key, value in model_metric_dict.items() if key not in ["train_time", "test_time", "precision", "recall", "f1"]})

#     import pdb
#     pdb.set_trace()
    return model_metric_df
        


# # VINA result

# In[5]:


from pathlib import Path

def load_lit_pcba_vina(nrows=None, split=None, target_list=None):

    
#     assert target_list is not None
    assert split is not None



    root_p = Path("/p/lustre2/ahashare/zhang30/LIT-PCBA-Data/")
    lig_map_p = Path("/p/lustre2/ahashare/zhang30/LIT-PCBA-Data/lig_rec/")
#     /p/lustre2/ahashare/zhang30/LIT-PCBA-Data/lig_rec/ # this path stores the map between conveyor and the original files
    
    path_list = [path for path in root_p.glob("*-actives.csv")]
#     print(len(path_list))

    df_list = []

    for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
        
#         import pdb
#         pdb.set_trace()
        target = path.name.split(".")[0].split("-")[0]
#         print(target, target_list, (target in target_list))
        if target in target_list or target_list == "all":

        
            active_lig_map = pd.read_csv(lig_map_p / f"lig-{target}-actives.csv")
            inactive_lig_map = pd.read_csv(lig_map_p / f"lig-{target}-inactives.csv")

            lig_map = pd.concat([active_lig_map, inactive_lig_map])

            
#             import pdb
#             pdb.set_trace()
            active_smiles_df = pd.read_csv(f"{lit_pcba_full_data_p}/{target}/actives.smi", delim_whitespace=True, header=None)
            inactive_smiles_df = pd.read_csv(f"{lit_pcba_full_data_p}/{target}/inactives.smi", delim_whitespace=True, header=None)
            smiles_df = pd.concat([active_smiles_df, inactive_smiles_df])

            input_num_mols = smiles_df.shape[0]


            # Filter out the test set smiles strings

            test_smiles_path = f"/p/vast1/jones289/hd_results/{SEED}/hdbind-rp-molformer.lit-pcba-{target}-{split}.{SEED}.pkl"         
            print(test_smiles_path)
            test_smiles = torch.load(test_smiles_path)["smiles_test"]

#             import pdb
#             pdb.set_trace()

            # TODO: there's missing docking data?
            smiles_df = pd.merge(smiles_df, pd.DataFrame({0: test_smiles}), on=0)

            smiles_df = smiles_df.drop_duplicates(subset=[0])

            print(f"started with {input_num_mols} molecules for {target}, after merging with available docking data, have {smiles_df.shape[0]}/{len(test_smiles)} in test set")



    #         smiles_df = smiles_df[smiles_df[1].apply(lambda x: x in test_smiles.values)]





            target_df = pd.merge(lig_map, smiles_df, left_on=" name", right_on=1)

            # dock_score_cols = {idx: value for idx,value in enumerate(["file", "key", "Mesg", "Box/cx", "Box/cy", "Box/cz", "Box/dx", "Box/dy", "Box/dz", 
                        #    "ligName", 
                        #    "numPose", 
                        #    "scores/1", 
                        #    "scores/2", 
                        #    "scores/3", 
                        #    "scores/4", 
                        #    "scores/5", 
                        #    "scores/6", 
                        #    "scores/7", 
                        #    "scores/8", 
                        #    "scores/9", 
                        #    "scores/10", 
                        #    "Box/default"])}
            lig_id_col = 9
            active_dock_scores = pd.read_csv(root_p / Path(f"{target}-actives.csv.clean"), header=None)
            active_dock_scores["label"] = [1] * len(active_dock_scores)

            inactive_dock_scores = pd.read_csv(root_p / Path(f"{target}-inactives.csv.clean"), header=None)
            inactive_dock_scores["label"] = [0] * len(inactive_dock_scores)
            dock_scores = pd.concat([active_dock_scores, inactive_dock_scores])


            dock_scores['min_vina'] = dock_scores[dock_scores.columns[11:21]].min(axis=1)


            dock_scores = dock_scores.groupby(lig_id_col)[[lig_id_col, 'min_vina', 'label']].min().reset_index(drop=True)


            target_df = pd.merge(dock_scores, target_df, left_on=9, right_on=" name")

            target_df["target"] = [target] * len(target_df)

            # compute enrichment at 1% and 10%




    #         import pdb
    #         pdb.set_trace() 

            df_list.append(target_df)
        else:
            pass


    df = pd.concat(df_list)
    return df


# In[6]:


def make_box_plot(enrich_1_df, enrich_10_df, dataset:str, split:str):
    
    enrich_f, enrich_ax = plt.subplots(2,1, figsize=(12,10), sharex=True, sharey=False)
    enrich_ax = enrich_ax.flatten()
    enrich_1_ax, enrich_10_ax = enrich_ax[0], enrich_ax[1]
    
    
    enrich_f.suptitle(f"{dataset.upper()} ({split})")
    
    
    sns.swarmplot(data=enrich_1_df, x="model", y="enrich", 
                  order=list(model_dict.keys()),
                  palette={key: value[1] for key, value in model_dict.items()},
                  ax=enrich_1_ax)
    enrich_1_ax.set_title(f"Enrichment at 1\%", fontdict={"fontsize": 18})
    enrich_1_ax.set_xlabel("")
    enrich_1_ax.set_ylabel("")
    enrich_1_ax.tick_params(axis="x", labelrotation=22.5)

    enrich_1_ax.set_ylabel("EF")

    plt.tight_layout()

        
    sns.swarmplot(data=enrich_10_df, x="model", y="enrich",
                  order=list(model_dict.keys()),
                  palette={key: value[1] for key, value in model_dict.items()},
                  ax=enrich_10_ax)

    enrich_10_ax.set_title(f"Enrichment at 10\%", fontdict={"fontsize": 18})
    enrich_10_ax.set_xlabel("")
    enrich_10_ax.set_ylabel("")
    enrich_10_ax.tick_params(axis="x", labelrotation=22.5)
    labels = [model_dict[item.get_text()][0] for item in enrich_10_ax.get_xticklabels()]

    enrich_10_ax.set_xticklabels(labels)
    enrich_ax[0].set_ylabel("EF")
    enrich_ax[1].set_ylabel("EF")

    plt.tight_layout()

    enrich_f.savefig(f"{dataset}-{split}-enrich.png", dpi=600, bbox_inches="tight")





def compute_model_metric_df(dataset, split, target_list=None):
    
    assert target_list is not None
    
    docking_df = load_lit_pcba_vina(target_list=target_list, split=split)

    
    model_metric_df = aggregate_results(dataset=dataset,
                                        split=split, target_list=target_list)
    

    docking_dict = {"enrich": [], "p": [], "model": [], "target": []}

    vina_col='min_vina'
    label_col = "label"
    for target, target_df in docking_df.groupby("target"):

        for p in [.1, .01]:
            enrich = compute_enrichment_factor(scores=np.abs(target_df[vina_col]), 
                                      labels=target_df[label_col], 
                                      n_percent=p)

            docking_dict["enrich"].append(float(enrich))
            docking_dict["p"].append(p)
            docking_dict["model"].append("Vina")
            docking_dict["target"].append(target)
           
    model_metric_df = pd.concat([model_metric_df, pd.DataFrame(docking_dict)])
    
    
    # take the average over the random seeds dimension for each combo of MODEL X TARGET X P
    grp_df = (model_metric_df).groupby(["model", "target", "p"])["enrich"].mean().reset_index()
    grp_df = grp_df.sort_values(by="model")
    
    
    for name, group in grp_df.groupby(["model", "p"]):
        print(name)
        print(group.describe())


    make_box_plot(enrich_1_df=grp_df[grp_df["p"] == .01], 
          enrich_10_df=grp_df[grp_df["p"] == .1],
             dataset=dataset, split=split)

    return model_metric_df


# 

# In[ ]:


ave_df = compute_model_metric_df(dataset="lit-pcba", split="ave", target_list="all")


# In[ ]:


ave_df.to_csv("lit-pcba-ave.csv")



# In[ ]:


random_df = compute_model_metric_df(dataset="lit-pcba", split="random", target_list="all")


# In[ ]:

random_df.to_csv("lit-pcba-random.csv")
#random_df.describe()


# In[ ]:





# In[ ]:




