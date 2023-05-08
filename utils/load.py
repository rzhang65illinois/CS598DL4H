import numpy as np
import random
from sklearn.preprocessing import StandardScaler
def load_data(file_path, solution_path, normalize_flag):
    inputdata = np.load(file_path)
    datasize, d = inputdata.shape
    if normalize_flag:
        inputdata = StandardScaler().fit_transform(inputdata)
    if solution_path is None:
        gtrue = np.zeros(d)
    else:
        gtrue = np.load(solution_path)
    true_graph = np.int32(np.abs(gtrue) > 1e-3)
    return inputdata, true_graph

def gen_instance_graph(inputdata, dimension):
    datasize, d = inputdata.shape
    seq = np.random.randint(datasize, size=dimension)
    input_ = inputdata[seq]
    return input_.T

def train_batch(inputdata, batch_size, dimension):
    input_batch = []
    for _ in range(batch_size):
        input_= gen_instance_graph(inputdata, dimension)
        input_batch.append(input_)
    return input_batch

def prior_knowledge_graph(true_graph, num_1, num_0):
    a = np.ones(true_graph.shape)*2
    # has relation
    x, y = np.where(true_graph==1)
    label_1 = random.sample(range(0, len(x)), int(num_1))
    a[(x[label_1], y[label_1])] = 1
    # has no relation
    x, y = np.where(true_graph==0)
    label_0 = random.sample(range(0, len(x)), int(num_0))
    a[(x[label_0], y[label_0])] = 0
    return a
