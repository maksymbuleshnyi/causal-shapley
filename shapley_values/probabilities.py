import numpy as np

def get_probability(unique_count, x_hat, indices_baseline, n):
    if len(indices_baseline) == 0:
        return 1

    count = 0
    x_hat_array = np.asarray(x_hat)
    
    for key, occurrences in unique_count.items():
        if all(key[j] == x_hat_array[j] for j in indices_baseline):
            count += occurrences

    return count / n

def conditional_prob(unique_count, x_hat, indices, indices_baseline, n):
    if len(indices) == 0:
        return get_probability(unique_count, x_hat, indices_baseline, n)

    numerator_indices = indices + indices_baseline
    numerator = get_probability(unique_count, x_hat, numerator_indices, n)
    denominator = get_probability(unique_count, x_hat, indices, n)
    return  numerator / (denominator + 1e-9)


def causal_prob(unique_count, x_hat, indices, indices_baseline, lenX, causal_struct: list[list[int]], confounding):
    
    in_coalition = set(indices)
    out_coalition = set(indices_baseline)
    # out_coalition | in_coalition
    # in_coalition | T
    prob = 1

    for index, component in enumerate(causal_struct):
        
        current_component = set(component)
        to_sample = list(current_component & out_coalition)
        if len(to_sample) > 0:
            parents = set([element for component1 in causal_struct[:index] for element in component1])
            if not confounding[index]:
                to_be_conditioned = (parents & out_coalition) | (parents & in_coalition) | (current_component & in_coalition)
            else:
                to_be_conditioned = (parents & out_coalition) | (parents & in_coalition)

            if len(to_be_conditioned) == 0:
                prob *= get_probability(unique_count, x_hat, list(to_sample), lenX)
            else:
                prob *= conditional_prob(unique_count, x_hat, list(to_be_conditioned), list(to_sample), lenX)

    return prob
