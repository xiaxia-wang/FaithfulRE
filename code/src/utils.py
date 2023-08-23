import configparser
import os
import random
import statistics

import math

from src.data import Data

"""This script takes a dataset representing an incomplete KG, and a dataset of positive examples,
and extends the latter with an equal number of negative examples; furthermore, it appends value '1' 
to each positive example and value 0 to each negative example."""

# Given a constant `root` and a directed graph (as a dictionary)
# of with constants for vertices,
# return all constants that are at distance at most 3 from `root`
def get_3hop_neighbours(root, graph):
    if root in graph:
        # Step 1
        onehop = graph[root]
        # Step 2
        twohop = set()
        for constant in onehop:
            if constant in graph:
                twohop = twohop.union(graph[constant])
        # Step 3
        threehop = set()
        for constant in twohop:
            if constant in graph:
                threehop = threehop.union(graph[constant])
        return onehop.union(twohop).union(threehop)
    else:
        return set([])


def generate_negative_examples(incomplete_graph_file, positive_examples_file, output):

    constants = set()
    relations = set()
    classes = set()

    # Process file of positive facts
    positive_examples = set()
    for line in open(positive_examples_file, "r").readlines():
        ent1, ent2, ent3 = line.split()
        if ent3.endswith('\n'):
            ent3 = ent3[:-1]
        read_triple = (ent1, ent2, ent3)
        if read_triple not in positive_examples:
            positive_examples.add(read_triple)
        if ent1 not in constants:
            constants.add(ent1)
        if ent2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            classes.add(ent3)
        else:
            constants.add(ent3)
            relations.add(ent2)
    print("Total positive examples read: {}".format(len(positive_examples)))

    # Total number of negative examples needed
    n_examples = len(positive_examples)
    print("Trying to generate {} negative examples...".format(n_examples))

    #  Process incomplete graph file
    # 'True known facts' is the union of the incomplete graph and the positive examples
    true_known_facts = positive_examples.copy()
    graph_dict_norelation = {}
    for line in open(incomplete_graph_file, "r").readlines():
        ent1, ent2, ent3 = line.split()
        if ent3.endswith('\n'):
            ent3 = ent3[:-1]
        read_triple = (ent1, ent2, ent3)
        if read_triple not in positive_examples:
            true_known_facts.add(read_triple)
        if ent1 not in constants:
            constants.add(ent1)
        if ent2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            classes.add(ent3)
        else:
            constants.add(ent3)
            relations.add(ent2)
        if ent1 in graph_dict_norelation:
            graph_dict_norelation[ent1].add(ent3)
        else:
            graph_dict_norelation[ent1] = set([ent3])

    # Convert sets to list in order to sample
    constants = list(constants)
    relations = list(relations)

    negative_examples = set()
    visible_counter = 0
    checkpoint = math.floor(n_examples/10)
    print(incomplete_graph_file, flush=True)

    # Create a list of all candidate pairs
    pairs = []
    for head in constants:
        for tail in get_3hop_neighbours(head,graph_dict_norelation):
            for relation in relations:
                pairs.append((head,relation,tail))

    # Choose the negative examples at random
    while len(negative_examples) < n_examples:
        fact = random.choice(pairs)
        if fact not in set.union(true_known_facts,negative_examples):
            negative_examples.add(fact)
            visible_counter += 1
            if visible_counter % checkpoint == 0:
                print("Found {} negative examples so far.".format(len(negative_examples)))

    print("All negative examples found.")

    #  Print to output file
    output_file = open(output, "w")
    pe = iter(positive_examples)
    ne = iter(negative_examples)
    # Interleave positive and negative facts
    for fact in pe:
        (ent1, ent2, ent3) = fact
        output_file.write(ent1 + '\t' + ent2 + '\t' + ent3 + '\t' + "1" + '\n')
        (ent1, ent2, ent3) = next(ne)
        output_file.write(ent1 + '\t' + ent2 + '\t' + ent3 + '\t' + "0" + '\n')
    output_file.close()


def sample_negative_examples_for_valid():
    config = configparser.ConfigParser()
    config.read("../config.ini")

    datasets_dir = config["DEFAULT"]["datasets_dir"]
    # for dir1 in ["fb237", "nell", "wn18rr"]:
    #     for dir2 in ["v1", "v2", "v3", "v4"]:
    #         base_dir = datasets_dir + "/" + dir1 + "_" + dir2
    #         original_graph = os.path.join(base_dir, "train_original.txt")
    #         positive_valid = os.path.join(base_dir, "valid.txt")
    #         output_file = os.path.join(base_dir, "valid_positive_negative_score.txt")
    #         generate_negative_examples(original_graph, positive_valid, output_file)
    # original_graph = datasets_dir + "/family/facts.txt"
    # positive_valid = datasets_dir + "/family/valid.txt"
    # output_file = datasets_dir + "/family/valid_positive_negative_score.txt"
    # generate_negative_examples(original_graph, positive_valid, output_file)

    original_graph = datasets_dir + "/family/facts.txt"
    positive_valid = datasets_dir + "/family/test_positive.txt"
    output_file = datasets_dir + "/family/test_positive_negative_score.txt"
    generate_negative_examples(original_graph, positive_valid, output_file)

def count_dataset_degrees():
    config = configparser.ConfigParser()
    config.read("../config.ini")

    datasets_dir = config["DEFAULT"]["datasets_dir"]
    for dir1 in ["fb237", "nell", "wn18rr"]:
        for dir2 in ["v1", "v2", "v3", "v4"]:
            base_dir = datasets_dir + "/" + dir1 + "_" + dir2
            print(base_dir, ": ")
            option_default_seed = 33
            option_no_extra_fasts = False
            data = Data(base_dir, option_default_seed, option_no_extra_fasts)
            triples_all = set(data.train_facts) | set(data.train) | set(data.valid)
            in_degree, out_degree = {}, {}
            for t in triples_all:
                '''each triple as: (relation, head, tail)'''
                out_count = out_degree.get(t[1], 0)
                out_degree[t[1]] = out_count + 1
                in_count = in_degree.get(t[2], 0)
                in_degree[t[2]] = in_count + 1
            in_degree_all, out_degree_all = list(in_degree.values()), list(out_degree.values())
            print(min(in_degree_all),
                  round(statistics.fmean(in_degree_all), 1),
                  statistics.median(in_degree_all),
                  max(in_degree_all))
            print(min(out_degree_all),
                  round(statistics.fmean(out_degree_all), 1),
                  statistics.median(out_degree_all),
                  max(out_degree_all))
            # print(statistics.median(in_degree_all), statistics.median(out_degree_all))
            print("=" * 32, flush=True)

def show_all_related_triples(data_file, entity):
    for line in open(data_file, "r").readlines():
        spo = line.strip().split()
        if spo[0] == entity or spo[2] == entity:
            print(line.strip())

if __name__ == "__main__":
    '''analyzing experimental results'''
    config = configparser.ConfigParser()
    config.read("../config.ini")
    datasets_dir = config["DEFAULT"]["datasets_dir"]
    # sample_negative_examples_for_valid()
    # count_dataset_degrees()

