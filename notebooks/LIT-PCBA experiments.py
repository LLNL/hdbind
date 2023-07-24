#!/usr/bin/env python
# coding: utf-8

# # LIT-PCBA 
# this dataset is a proposed "more difficult" test than DUD-E. I used a 75/25 (sklearn default) stratified split. The dataset is heavily imbalanced, like DUD-E.

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
import torchmetrics
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import sys
# sys.path.append('..')
sys.path.insert(0, "/g/g13/jones289/workspace/hd-cuda-master/hdpy")
# print(sys.path)
import hdpy
import hdpy.ecfp_hd
from hdpy.analysis import load_pkl
from hdpy.metrics import compute_enrichment_factor


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=17)
plt.rc('figure', titlesize=20)


# In[2]:


data_p = Path("/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/results/125")
# data_p = Path("/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/results/124")
# data_p = Path("/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/results/4")
# data_p = Path("/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/results/0")
# data_p = Path("/usr/WS1/jones289/hd-cuda-master/hdpy/hdpy/before_rng_results/before_rng_results")


# In[3]:


green_color_pal = sns.color_palette("Greens", 10)
blue_color_pal = sns.color_palette("Blues", 10)
rocket_color_pal = sns.color_palette("rocket", 10)


# In[4]:


color_dict = {
#     "smiles-pe.atomwise.0": rocket_color_pal[2],
#     "smiles-pe.bpe.0": rocket_color_pal[4],   
# "selfies.atomwise": green_color_pal[4],
    "ecfp": green_color_pal[6],
#     "rp": green_color_pal[8],
    # "rf": blue_color_pal[4],
#     "mlp": blue_color_pal[7],
#     "HDC-MLP": green_color_pal[9],
    "HDC-RF": green_color_pal[9],
    "Vina": "salmon",
}



marker_dict = {
#     "smiles-pe": "+",
    "smiles-pe.atomwise.0": "+",
    "smiles-pe.bpe.0": "+",   
#     "smiles-pe.ngram.1": "+",
    "selfies.atomwise": "*",
#     "selfies.selfies-charwise": "*",
    "ecfp": "+",
    "rp": "+",
    "rf": "^",
#     "openhd": "*",
    "mlp": "+",
#     "Vina": "+",
}


model_order_list = [
    ("smiles-pe.atomwise.0", "MoleHD-Atomw."),
    ("smiles-pe.bpe.0", "MoleHD-BPE"),
#     ("smiles-pe.ngram.1", "SMILES uni-gram"),
    ("selfies.atomwise", "HDBind-SELFIES"),
#     ("selfies.selfies-charwise", "SELFIES uni-gram"),
    ("ecfp", "HDBind-ECFP"),
    ("rp", "HDBind-ECFP+RP"),
    ("rf", "RF"),
    ("mlp", "MLP"),
#     ("Vina", "Vina")
]


model_name_dict = {
    "smiles-pe.atomwise.0": "MoleHD-Atomw.", 
    "smiles-pe.bpe.0": "MoleHD-BPE",
#     "smiles-pe.ngram.1": "SMILES uni-gram",
    "selfies.atomwise": "HDBind-SELFIES",
#     "selfies.selfies-charwise": "SELFIES uni-gram",
    "ecfp": "HDBind-ECFP",
    "rp": "HDBind-RPFP",
    "rf": "RF",
    "mlp": "MLP",
    "Vina": "Vina",
    "HDC-MLP": "HDC-MLP",
    "HDC-RF": "HDC-RF"
}


linestyle_dict = {
    "smiles-pe.atomwise.0": "-", 
    "smiles-pe.bpe.0": ":",
    "selfies.None": "-",
    "ecfp": ":",
    "rp": "-.",
    "rf": "-",
    "mlp": ":",
    "Vina": "-"
}


# # LIT-PCBA Results

# In[6]:


def aggregate_results(dataset, range_limit=10, multistep_initial_p=None, 
                      multistep_p_list=None, 
                      multistep_sklearn_model=None):
    
    
    
    
    
    model_metric_dict = {"model": [], "enrich": [], "p":[], "train_time":[], "test_time": [], "target": [],
                        "seed": []}
    tokenizer="atomwise"
    ngram_order=0

    for model, color in color_dict.items():
                        
        metric_list = []
        encode_time_list = []
        train_time_list = []
        test_time_list = []
        train_size_list = []
        test_size_list = []
        target_size_list = []
        eta_list = []

        
        if model not in ["HDC-MLP", "HDC-RF"]:
            data_path_list = list(data_p.glob(f"{dataset.replace('-','_')}*.{model}*pkl"))
#             print(model)
        elif model in ["HDC-MLP", "HDC-RF"]:
#             print(f"{model}: multistep filter to be implemented")
            data_path_list = list(data_p.glob(f"{dataset.replace('-','_')}*.ecfp*pkl"))
        

        for path in tqdm(data_path_list, total=len(data_path_list)):
            
            print(path, model)
            
            with open(path, "rb") as handle:
                model_data_dict = pickle.load(handle)

            
            target = path.name.split(".")[1]
                
            hd_cache_dir = f"/p/lustre2/jones289/hd_cache/125/ecfp/{dataset}/random"

            
                
            for seed in range(range_limit):
                                
                y_true = model_data_dict["y_test"]
                eta = None 

                if model in ["rf", "mlp"]:

#                     eta = model_data_dict[seed]["model"].predict_proba(model_data_dict["x_test"])[:,1]
                    eta = model_data_dict[seed]["eta"][:, 1]

                    for p in [.01, .1]:

                        enrich = compute_enrichment_factor(sample_scores=eta, sample_labels=y_true, n_percent=p)

                        model_metric_dict["model"].append(model)
                        model_metric_dict["target"].append(target)
                        model_metric_dict["enrich"].append(enrich)
                        model_metric_dict["p"].append(p)
                        model_metric_dict["seed"].append(seed)

                        
                elif model.lower() in ["hdc-rf", "hdc-mlp"]:
                    import pdb 
                    pdb.set_trace()
                    sklearn_result_file = Path(f"{data_p}/{dataset.replace('-', '_')}.{target}.{multistep_sklearn_model}.None.{ngram_order}.pkl")

                    with open(sklearn_result_file, "rb") as handle:
                        sklearn_result_dict = pickle.load(handle)                        
                    
                    
                    if isinstance(model_data_dict[seed]["y_true"], np.ndarray):
                        y_true = model_data_dict[seed]["y_true"]
                    else:
                        y_true = np.concatenate(model_data_dict[seed]["y_true"])
                    
                    
                    target_test_hv_path = f"{hd_cache_dir}/{target}/test_dataset_hv.pth"
                
                
                    hv_test = torch.load(target_test_hv_path, map_location="cpu")
                    

                    hdc_model = model_data_dict[seed]["model"]
#                     hdc_model = hdc_model.to("cpu")
#                     hv_test = hv_test.cpu()
                    hdc_conf_scores = hdc_model.compute_confidence(hv_test)
#                     hdc_conf_scores = torch.from_numpy(model_data_dict[seed]["eta"])
                                            
#                     # filter the data
                    values , idxs = torch.sort(hdc_conf_scores.squeeze().cpu(), descending=True)

                    sample_n = int(np.ceil(multistep_initial_p * y_true.shape[0]))

                    hd_actives = sum(y_true[idxs[:sample_n]])

                    actives_database = sum(y_true)
#                     # rescore at 10% 
                            
#                     # rescore at 1%
                                            
                    for p in multistep_p_list:

#                         sklearn_model = sklearn_result_dict[seed]["model"]
#                         # get the indexes of the top initial-p% of compounds ranked by HDC
                        samp_idxs = (idxs[:sample_n]).numpy()

#                         # take result of filtering from HDC
                        x_test_samp = model_data_dict["x_test"][samp_idxs]
                        y_true_samp = y_true[samp_idxs]

#                         sklearn_scores = sklearn_model.predict_proba(x_test_samp)[:, 1]
                        sklearn_scores = sklearn_result_dict[seed]["eta"][:, 1]
            
            
                        enrich = compute_enrichment_factor(sample_scores=sklearn_scores, 
                                                sample_labels=y_true_samp,
                                                n_percent=p, 
                                                actives_database=sum(y_true), 
                                                database_size=y_true.shape[0])

                        model_metric_dict["model"].append(model)
                        model_metric_dict["target"].append(target)
                        model_metric_dict["enrich"].append(enrich)
                        model_metric_dict["p"].append(p)
                        model_metric_dict["seed"].append(seed)

                else:

                    import pdb
                    pdb.set_trace()
#                     print(f"{model} not implemented yet")
                    if isinstance(model_data_dict[seed]["y_true"], np.ndarray):
                        y_true = model_data_dict[seed]["y_true"]
                    else:
                        y_true = np.concatenate(model_data_dict[seed]["y_true"])
                    
                    target_test_hv_path = f"{hd_cache_dir}/{target}/test_dataset_hv.pth"
                
                
                    hv_test = torch.load(target_test_hv_path, map_location="cpu")
                    hdc_model = model_data_dict[seed]["model"]
#                     hdc_model = hdc_model.to("cpu")
#                     hv_test = hv_test.cpu()
                    hdc_conf_scores = hdc_model.compute_confidence(hv_test)
#                     hdc_conf_scores = model_data_dict[seed]["eta"]
                    cos_sims = torchmetrics.functional.pairwise_cosine_similarity(hv_test, torch.concat([hdc_model.am[key].reshape(1,-1) for key in sorted(hdc_model.am.keys())], dim=0))
                    
                    for p in [.01, .1]:

                        enrich = compute_enrichment_factor(sample_scores=hdc_conf_scores, 
                                                sample_labels=y_true,
                                                n_percent=p, 
                                                actives_database=sum(y_true), 
                                                database_size=y_true.shape[0])

                        model_metric_dict["model"].append(model)
                        model_metric_dict["target"].append(target)
                        model_metric_dict["enrich"].append(enrich)
                        model_metric_dict["p"].append(p)
                        model_metric_dict["seed"].append(seed)
                    

    return model_metric_dict
        


# In[ ]:


model_metric_dict = aggregate_results(dataset="lit-pcba",multistep_p_list=[.01, .1],  
                                      multistep_sklearn_model="rf", 
                                     multistep_initial_p=1.0)


# In[ ]:


# print([len(x) for x in model_metric_dict.values()])
# print([x for x in model_metric_dict.keys()])


# In[ ]:


model_metric_dict


# In[ ]:


# 'smiles-pe.ngram.2', 'smiles-pe.ngram.3', 'rf', 'mlp'


# In[ ]:


model_metric_df = pd.DataFrame({key: value for key, value in model_metric_dict.items() if key not in ["train_time", "test_time"]})

model_metric_df


# # VINA result

# In[ ]:


from pathlib import Path
def docking_main(nrows=None):
    root_p = Path("/p/lustre2/ahashare/zhang30/LIT-PCBA-Data/")

#     color_dict.update({"Vina": "salmon"})

    path_list = [path for path in root_p.glob("*-actives.csv")]


    df_list = []

    for idx, path in tqdm(enumerate(path_list), total=len(path_list)):
        print(idx, path)
        
#         '''
        target = path.name.split(".")[0].split("-")[0]
        print(target, idx+1, path)
        
        

        merged_df = None
        merged_df_path = Path(f"./lit_pcba_docking_analysis/{target}.csv")
        
        if not merged_df_path.exists():
            # can use the set of smiles in each result file
            result_pkl = Path(f"/g/g13/jones289/workspace/hd-cuda-master/hdpy/hdpy/results/124/lit_pcba.{target}.ecfp.atomwise.0.pkl")        




            target_train_smiles_list = []
            target_test_smiles_list = []

            with open(result_pkl, "rb") as handle:

                data = pickle.load(handle)

                target_train_smiles_list = data["smiles_train"]
                target_test_smiles_list = data["smiles_test"]

                print(f"total of {len(target_train_smiles_list)} in training set, total of {len(target_test_smiles_list)} in testing set.")

            df_cols = ['file', ' scores/1', ' ligName']
            active_df = pd.read_csv(root_p / Path(f"{target}-actives.csv"), sep=",", usecols=df_cols, nrows=nrows)
            active_df['y_true'] = [1] * len(active_df)



            inactive_df = pd.read_csv(root_p / Path(f"{target}-inactives.csv"), sep=",", usecols=df_cols, nrows=nrows)
            inactive_df['y_true'] = [0] * len(inactive_df)

            target_df = pd.concat([active_df, inactive_df])
            # this will search over all of the docking results for each target, across each of the multiple protein models




            active_smiles_df = pd.read_csv(f"/p/vast1/jones289/lit_pcba/{target}/actives.smi", delim_whitespace=True, header=None)
            inactive_smiles_df = pd.read_csv(f"/p/vast1/jones289/lit_pcba/{target}/inactives.smi", delim_whitespace=True, header=None)
            target_smiles_df = pd.concat([active_smiles_df, inactive_smiles_df])



            top_pose_target_df = target_df.groupby([' ligName'], as_index=False)[[' ligName', ' scores/1', 'y_true']].min()


        
            merged_df_path.parent.mkdir(exist_ok=True, parents=True)
            merged_df = pd.merge(top_pose_target_df, target_smiles_df, left_on=" ligName", right_on=1)
            merged_df = merged_df[merged_df.apply(lambda x: x[0] in target_test_smiles_list, axis=1)]
            merged_df['target'] = [target] * len(merged_df)
            merged_df.to_csv(merged_df_path, index=False)
        else:
            merged_df = pd.read_csv(merged_df_path)
            
            if 'target' not in merged_df.columns:
                merged_df['target'] = [target] * len(merged_df)
                merged_df.to_csv(merged_df_path, index=False)



        df_list.append(merged_df)

    df = pd.concat(df_list)
    return df
    
    
    
#######
vina_result = docking_main(nrows=None)
vina_result
###########

docking_dict = {"enrich": [], "p": [], "model": [], "target": []}

vina_enrich_list = []
target_list = []
vina_col=' scores/1'
for target, target_df in vina_result.groupby("target"):
    
    for p in [.1, .01]:
        enrich = compute_enrichment_factor(sample_scores=np.abs(target_df[vina_col]), 
                                  sample_labels=target_df["y_true"], 
                                  n_percent=p)
        
        docking_dict["enrich"].append(float(enrich))
        docking_dict["p"].append(p)
        docking_dict["model"].append("Vina")
        docking_dict["target"].append(target)
    

##################
model_metric_df = pd.concat([model_metric_df, pd.DataFrame(docking_dict)])

model_metric_df




# In[ ]:


model_metric_df["model"]


# In[ ]:


def make_plot(enrich_1_df, enrich_10_df):

    enrich_f, enrich_ax = plt.subplots(2,1, figsize=(12,10), sharex=True, sharey=False)
    enrich_ax = enrich_ax.flatten()
    enrich_1_ax, enrich_10_ax = enrich_ax[0], enrich_ax[1]
    
    sns.boxplot(data=enrich_1_df, x="model", y="enrich", ax=enrich_1_ax, palette=color_dict)
    enrich_1_ax.set_title("(a) LIT-PCBA Enrichment at 1\%", fontdict={"fontsize": 18})
    enrich_1_ax.set_xlabel("")
    enrich_1_ax.set_ylabel("")
    enrich_1_ax.tick_params(axis="x", labelrotation=22.5)

    enrich_1_ax.set_ylabel("EF")

    plt.tight_layout()

    
    sns.boxplot(data=enrich_10_df, x="model", y="enrich", ax=enrich_10_ax, palette=color_dict)
    enrich_10_ax.set_title("(b) LIT-PCBA Enrichment at 10\%", fontdict={"fontsize": 18})
    enrich_10_ax.set_xlabel("")
    enrich_10_ax.set_ylabel("")
    enrich_10_ax.tick_params(axis="x", labelrotation=22.5)
    labels = [item.get_text() for item in enrich_10_ax.get_xticklabels()]
    labels = [model_name_dict[x.get_text()] for x in enrich_10_ax.get_xticklabels()]
#     labels[-1] = combo_model_name
    enrich_10_ax.set_xticklabels(labels)
    enrich_ax[0].set_ylabel("EF")
    enrich_ax[0].set_ylabel("EF")

    plt.tight_layout()
    # enrich_10_f.savefig("enrich_10.png", dpi=600, bbox_inches="tight")
    # enrich_10_f
    
    enrich_f.savefig("lit-pcba-enrich.png", dpi=600, bbox_inches="tight")

#     return enrich_f


# In[ ]:


make_plot(enrich_1_df=model_metric_df, enrich_10_df=model_metric_df)


# # TIMINGS

# 

# In[ ]:


# model_metric_df.groupby("model").describe()[['enrich-1-mean', 'enrich-10-mean']]


# In[ ]:


# group_dict = {group_name: group_df for group_name, group_df in model_metric_df.groupby('model')}


# In[ ]:


# for group_name in group_dict.keys():
#     print(f"{group_name}-{group_dict[group_name]['train_time'].mean()}")


# In[ ]:


# for group_name in group_dict.keys():
#     print(f"{group_name}-{group_dict[group_name]['test_time'].mean()}")


# In[ ]:


# f, ax = plt.subplots(1,1, figsize=(14,8))

# g = sns.boxplot(data=model_metric_df[model_metric_df["model"] != "Vina"], x="model", y="train_time", ax=ax, palette=color_dict)
# ax.tick_params(axis="x", labelrotation=22.5)
# g.set_yscale("log")


# In[ ]:


# f, ax = plt.subplots(1,1, figsize=(14,8))
# 
# g = sns.boxplot(data=model_metric_df[model_metric_df["model"] != "Vina"], x="model", y="test_time", ax=ax, palette=color_dict)
# ax.tick_params(axis="x", labelrotation=22.5)
# g.set_yscale("log")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




