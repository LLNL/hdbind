import pandas as pd


def process_target(target_name="VDR", target_id="AID504847"):



    train_actives = pd.read_csv(f"/p/vast1/jones289/lit_pcba/AVE_unbiased/{target_name}/{target_id}_active_T.smi", header=None, delim_whitespace=True) 
    test_actives = pd.read_csv(f"/p/vast1/jones289/lit_pcba/AVE_unbiased/{target_name}/{target_id}_active_V.smi", header=None, delim_whitespace=True) 
    test_actives["label"] = [1] * len(test_actives)
    train_actives["label"] = [1] * len(train_actives)

    train_inactives = pd.read_csv(f"/p/vast1/jones289/lit_pcba/AVE_unbiased/{target_name}/{target_id}_inactive_T.smi", header=None, delim_whitespace=True) 
    test_inactives = pd.read_csv(f"/p/vast1/jones289/lit_pcba/AVE_unbiased/{target_name}/{target_id}_inactive_V.smi", header=None, delim_whitespace=True) 
    test_inactives["label"] = [0] * len(test_inactives)
    train_inactives["label"] = [0] * len(train_inactives)



    print("train_actives.shape", train_actives.shape, 
        "train_inactives.shape", train_inactives.shape,
        "test_actives.shape", test_actives.shape,
        "test_inactives.shape", test_inactives.shape)

    dock_actives = pd.read_csv(f"/p/vast1/jones289/LIT-PCBA-Data/{target_name}-actives.csv")
    dock_actives["id"] = dock_actives[" key"].apply(lambda x: int(x.split("/")[-1]))
    dock_inactives = pd.read_csv(f"/p/vast1/jones289/LIT-PCBA-Data/{target_name}-inactives.csv")
    dock_inactives["id"] = dock_inactives[" key"].apply(lambda x: int(x.split("/")[-1]))


    print("dock_actives.shape", dock_actives.shape, "dock_inactives.shape",dock_inactives.shape)



    train_df = pd.concat([train_actives, train_inactives])
    train_df["id"] = train_df[1]
    test_df = pd.concat([test_actives, test_inactives])
    test_df["id"] = test_df[1]

    print("train_df.shape", train_df.shape, "test_df.shape", test_df.shape)

    dock_actives["dock_id"] = dock_actives[' key'].apply(lambda x: int(x.split("/")[-1]))
    dock_inactives["dock_id"] = dock_inactives[' key'].apply(lambda x: int(x.split("/")[-1]))


    print("dock_actives.shape", dock_actives.shape, "dock_inactives.shape", dock_inactives.shape)


    active_lig_map_df = pd.read_csv(f"/p/vast1/jones289/LIT-PCBA-Data/lig_rec/lig-{target_name}-actives.csv")
    active_lig_map_df["id"] = active_lig_map_df[" key"].apply(lambda x: int(x.split("/")[-1]))


    inactive_lig_map_df = pd.read_csv(f"/p/vast1/jones289/LIT-PCBA-Data/lig_rec/lig-{target_name}-inactives.csv")
    inactive_lig_map_df["id"] = inactive_lig_map_df[" key"].apply(lambda x: int(x.split("/")[-1]))

    print("active_lig_map_df.shape", active_lig_map_df.shape, "inactive_lig_map_df.shape", inactive_lig_map_df.shape)

    active_df = pd.merge(dock_actives, active_lig_map_df, left_on="dock_id", right_on="id")
    inactive_df = pd.merge(dock_inactives, inactive_lig_map_df, left_on="dock_id", right_on="id")

    print("active_df.shape", active_df.shape, "inactive_df.shape", inactive_df.shape)

    train_active_df = pd.merge(active_df, train_df, left_on=" name", right_on="id")
    test_active_df = pd.merge(active_df, test_df, left_on=" name", right_on="id")
    train_inactive_df = pd.merge(inactive_df, train_df, left_on=" name", right_on="id")
    test_inactive_df = pd.merge(inactive_df, test_df, left_on=" name", right_on="id")

    print("train_active_df.shape", train_active_df.shape, "train_inactive_df.shape", train_inactive_df.shape)
    print("test_active_df.shape", test_active_df.shape, "test_inactive_df.shape", test_inactive_df.shape)


    train_df = pd.concat([train_active_df, train_inactive_df])
    test_df = pd.concat([test_active_df, test_inactive_df])

    print("train_data size ", train_df.shape, train_df["id"].unique().shape)
    print("test_data size", test_df.shape, test_df["id"].unique().shape)


    # print(train_df.head())
    # print(test_df.head())


    # import pdb 
    # pdb.set_trace()
    train_df["min_vina"] = train_df[train_df.columns[11:21]].min(axis=1)
    train_df["target"] = [target_name] * len(train_df)

    test_df["min_vina"] = test_df[test_df.columns[11:21]].min(axis=1)
    test_df["target"] = [target_name] * len(test_df)



    # if split == "random":

        # load the file for the random split


        # return test_df

    # elif split == "ave":

        # return test_df

def main():

    target_tup_list = [

        ("492947", "ADRB2"),
        ("1030", "ALDH1"),
        ("743075", "ESR1_ago"),
        ("743080", "ESR1_ant"),
        ("588795", "FEN1"),
        ("2101", "GBA"),
        ("602179", "IDH1"),
        ("504327", "KAT2A"),
        ("995", "MAPK1"),
        ("493208", "MTORC1"),
        ("1777", "OPRK1"),
        ("1631", "PKM2"),
        ("743094", "PPARG"),
        ("651631", "TP53"),
        ("504847", "VDR"),


    ]

    target_tup_list = [target_tup_list[0]]

    df_list = []
    for target_id, target_name in target_tup_list:
        print(target_id, target_name) 
        target_df = process_target(target_name, target_id=f"AID{target_id}")
        df_list.append(target_df)
        print()

    df = pd.concat(df_list)

    return df
if __name__ == "__main__":

    main()