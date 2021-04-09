import numpy as np
import random
import math

# subproblem_list: list of lists, one for each subproblem, containing assigned entities
# input_set_dims: dictionary mapping entity to its dimensions
# calculate mean value of every dimension of each subproblem and return
def check_dims(subproblem_list, input_set_dims):
    
    num_subproblems = len(subproblem_list)
    num_dimensions = len(list(input_set_dims.values())[0])
    print("checking split of " + str(num_dimensions) + " dimensions over " + 
          str(num_subproblems) + " subproblems")
    
    subproblem_dim_sums = [np.zeros(num_dimensions) for _ in range(num_subproblems)]
    
    for k in range(num_subproblems):
        subproblem_entities = subproblem_list[k]
        
        for entity in subproblem_entities:
            entity_dims = input_set_dims[entity]
            subproblem_dim_sums[k] = np.add(np.asarray(entity_dims), subproblem_dim_sums[k])
        subproblem_dim_sums[k] = subproblem_dim_sums[k]/len(subproblem_entities)
        print("subproblem " + str(k) + ": " + str(subproblem_dim_sums[k]))
    return subproblem_dim_sums


def calc_cov_online(current_cov, current_means, num_entity, new_entity):
    num_dims = len(current_means)
    new_means = (current_means * num_entity + np.asarray(new_entity))/(num_entity + 1)
            
    resid_row_array = np.asarray([[d-new_means[i]]*num_dims for i, d in enumerate(new_entity)])
    resid_col_array = np.transpose(resid_row_array)
    
    new_cov = current_cov*(num_entity-1) + \
                (num_entity/(num_entity-1)) * (np.multiply(resid_row_array,resid_col_array))
    new_cov = new_cov/num_entity
    
    return new_cov
    
# calculate the change in MSE between subproblem inputs and all inputs covariance
def calc_dist_cov_change(input_set_dims, new_entity, origin_dist_covs, 
                         num_entity_mean, current_covs):
    num_dims = len(new_entity)
    #print("old covs: " + str(current_covs))
    num_entity_in_sp = num_entity_mean[0]
    new_covs = np.zeros((num_dims,num_dims))
    if num_entity_in_sp > 0:

        # calculate it from scratch for the first 50 elements
        if (num_entity_in_sp < 2):
            input_set_dims_new = copy.deepcopy(input_set_dims)
            for d in range(num_dims):
                input_set_dims_new[d] += [new_entity[d]]
            new_covs = np.cov(input_set_dims_new)
        # use online update estimate
        else:
            new_covs = calc_cov_online(current_covs, num_entity_mean[1], num_entity_mean[0], new_entity)
    
    old_mse = ((current_covs - origin_dist_covs)**2).mean(axis=None)
    new_mse = ((new_covs - origin_dist_covs)**2).mean(axis=None)
    
    dist_diff = old_mse - new_mse
    return dist_diff, new_covs

# Compute the change in distance (2-norm of dimensional means)
# from the original problem inputs when adding new entity
# TODO: better distance metric that considers covariance?
def calc_dist_mean_change(input_set_dims, new_entity, origin_dist, num_entity_mean):
    num_dims = len(new_entity)
    num_entity = num_entity_mean[0]
    current_means = num_entity_mean[1]
    
    #new_means = (current_means*num_entity + np.asarray(new_entity))/(num_entity+1)
    new_means = current_means + np.asarray(new_entity)
    
    sq_sum_distance_new = np.sum(np.square((new_means - origin_dist)/origin_dist))
    sq_sum_distance = np.sum(np.square((current_means - origin_dist)/origin_dist))

    return math.sqrt(sq_sum_distance) - math.sqrt(sq_sum_distance_new)

# input_dict: keys are entities and values are (ordered) list of entity dimensions, 
# k: number of subproblems
def split_generic(input_dict, k, verbose=False, method='means'):
    
    num_inputs = len(input_dict)
    num_dimensions = len(list(input_dict.values())[0])
    
    # original_dist_dict: keys are dimension indices (0..d), values are frequency
    original_dist_means_array = np.zeros(num_dimensions)
    original_dist_inputs_by_dim = []
    for d in range(num_dimensions):
        inputs = [val[d] for val in input_dict.values()]
        original_dist_inputs_by_dim.append(inputs)
        sum_d = sum(inputs)
        original_dist_means_array[d] = sum_d#/(num_inputs*1.0)
    
    original_dist_cov = np.cov(original_dist_inputs_by_dim)
    
    # subproblem_dim_lists has k lists, one for each subproblem. Within each list are d lists,
    # each containing the value of a dimension for each entity assigned to that subproblem
    subproblem_dim_lists =  [[[] for _ in range(num_dimensions)] for _ in range(k)]
    
    # subproblem_entity_assignments is a list of lists
    subproblem_entity_assignments = [[] for _ in range(k)]
    
    # Assign each entity to the sub-problem that would have their distance from the
    # original distribution shrink the most.
    num_assigned = 0
    subproblem_num_entity_means = [[0, np.zeros(num_dimensions)] for _ in range(k)]
    subproblem_covs = [np.zeros((num_dimensions,num_dimensions)) for _ in range(k)]
    for entity, dims in input_dict.items():
        if num_assigned % 5000 == 0:
            print("Assigned " + str(num_assigned) + " entities")
        max_dist_change = -np.inf
        max_dist_sp = 0
        updated_cov = None
        random_sp1 = random.randint(0,k-1)
        while len(subproblem_entity_assignments[random_sp1]) > (num_inputs+1)/(k*1.0):
            random_sp1 = random.randint(0,k-1)
        random_sp2 = random.randint(0,k-1)
        while random_sp1 == random_sp2 or len(subproblem_entity_assignments[random_sp2]) > (num_inputs+1)/(k*1.0):
            random_sp2 = random.randint(0,k-1)
        
        for sp_index in [random_sp1, random_sp2]:
            
            # skip those that have more than equal share of currently assigned entities
            if len(subproblem_entity_assignments[sp_index]) > (num_inputs+1)/(k*1.0):
                continue
            dist_change = 0
            if method == 'means':
                dist_change = calc_dist_mean_change(subproblem_dim_lists[sp_index], dims, 
                                           original_dist_means_array, subproblem_num_entity_means[sp_index])
            elif method == 'covs':
                dist_change, new_cov = calc_dist_cov_change(subproblem_dim_lists[sp_index], dims, 
                                           original_dist_cov, subproblem_num_entity_means[sp_index],
                                                       subproblem_covs[sp_index])
            #print("subproblem " + str(i) + ", dist change: " + str(dist_change))
            if dist_change >= max_dist_change:
                max_dist_change = dist_change
                max_dist_sp = sp_index
                if method == 'covs':
                    updated_cov = new_cov
                
        subproblem_entity_assignments[max_dist_sp].append(entity)
        if method == 'cov':
            subproblem_covs[max_dist_sp] = new_cov
        
        # update means to reflect entity assignment
        for d in range(num_dimensions):
            subproblem_dim_lists[max_dist_sp][d].append(dims[d])
            
        num_entity = subproblem_num_entity_means[max_dist_sp][0]
        dim_means = subproblem_num_entity_means[max_dist_sp][1]
        subproblem_num_entity_means[max_dist_sp][0] += 1
        #subproblem_num_entity_means[max_dist_sp][1] = (dim_means*num_entity + np.asarray(dims))/(num_entity+1)
        subproblem_num_entity_means[max_dist_sp][1] = dim_means + np.asarray(dims)

        num_assigned += 1
        
        if verbose:
            print(subproblem_dim_lists)
            print(subproblem_entity_assignments)
            print('\n')
    return subproblem_entity_assignments