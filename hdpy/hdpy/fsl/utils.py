import random
import numpy as np
import os

from hdpy.utils import convert_pdbbind_affinity_to_class_label


def map_features_by_class(data, class_list, col_index=0, dataset=None):
    '''
        Takes a data (iterable?), list of acceptable classes, column index, and task_type for 
        the label as input and outputs a dictionary with a list of data elements that indexed
        by the unique label values.
    '''

    assert dataset is not None
    class_dict = {}

    for element in data:

        # will make the assumption that the provided label is some binary format (bind/no-bind, strong-bind/weak-bind, etc.) otherwise 
        # it is necessary to make an explicit case that handles the conversion of that measure to a binary format 
        element_class = None
        if dataset == "pdbbind":
            element_target = int(element[col_index]) # expecting ints, make sure this is converted to int for use as dict key
            element_class = convert_pdbbind_affinity_to_class_label(element_target)

        else:
            element_class = int(element[col_index])

        if element_class in class_list:

            # if we haven't added the class to the class_dict yet then do so
            if element_class not in class_dict.keys():
                class_dict[element_class] = []

            # append the data to the element class data list
            class_dict[element_class].append(element)

    return class_dict


def load_features(support_path_list:list, query_path_list:list, dataset:str):

    support_list = []

    for support_path in support_path_list: 
    # randomly sample the support
        data = np.load(support_path)
    
        support_list.append(data)
    
    support = np.concatenate(support_list)

    # divide the data into a dictionary indexed by class label
    #support_class_dict = map_features_by_class(support_data,class_list=class_list, col_index=-1, dataset=dataset)
    #support = []

    # nb_shot=-1 is a case where we use the entire dataset for training, otherwise we stratify by class
    #for class_idx in support_class_dict.keys(): 
    
    #    if nb_shot == -1:
    #        support.append(np.asarray(support_class_dict[class_idx]))
    #    elif class_idx in class_list:
    #        support.append(random.sample(support_class_dict[class_idx], nb_shot))
    
    #support = np.concatenate(support)
 

    query_list = []

    for query_path in query_path_list: 
    # randomly sample the support
        data = np.load(query_path)
    
        query_list.append(data)
    
    query = np.concatenate(query_list)



    # divide the data into a dictionary indexed by class label
    # query_class_dict = map_features_by_class(query_data, class_list=class_list, col_index=-1, dataset=dataset)
    # query = []

    # kquery=-1 is a case where we use the entire dataset for training, otherwise we stratify by class
    # for class_idx in query_class_dict.keys():
        # if kquery == -1:
            # query.append(np.asarray(query_class_dict[class_idx]))
        # elif class_idx in class_list:
            # query.append(np.asarray(random.sample(query_class_dict[class_idx], kquery)))
    # query = np.concatenate(query)


    if dataset == "pdbbind":
        # map the experimental -logki/kd value to a discrete category
        features_support = support[:, :-1]
        labels_support = support[:, -1]
        labels_support = np.asarray([convert_pdbbind_affinity_to_class_label(x) for x in labels_support])

        features_query = query[:, :-1]
        labels_query = query[:, -1]
        labels_query = np.asarray([convert_pdbbind_affinity_to_class_label(x) for x in labels_query])

    else:

        features_support = support[:, :-1]
        labels_support = support[:, -1]

        features_query = query[:, :-1]
        labels_query = query[:, -1]
 
    return features_support, labels_support, features_query, labels_query
