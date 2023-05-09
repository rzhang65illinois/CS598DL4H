import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import Actor, Critic
from reward import get_Reward
from utils.load import load_data, train_batch, prior_knowledge_graph
from utils.eval import graph_prunned_by_coef, count_accuracy
from utils.bic import BIC_lambdas

output_dir = 'datasets/LUCAS' # please update the directory!
save_model_path = '{}/model'.format(output_dir)
plot_dir = '{}/plot'.format(output_dir)
graph_dir = '{}/graph'.format(output_dir)

file_path = '{}/data.npy'.format(output_dir)
solution_path = '{}/true_graph.npy'.format(output_dir)
inputdata, true_graph = load_data(file_path, solution_path, True)

# Parameters
num_prior_edge = 3
reg_type = 'LR'
nb_epoch = 10000                 
input_dimension = 64
lambda_iter_num = 1000
batch_size = 32
sl, su, strue = BIC_lambdas(inputdata, None, None, None, reg_type)
max_length = true_graph.shape[0]
lambda1 = 0
lambda1_upper = 5
lambda1_update_add = 1
lambda2 = 1/(10**(np.round(max_length/3)))
lambda2_upper = 0.01
lambda2_update_mul = 10
lambda3 = 0
lambda3_upper = 1
lambda3_update_add = 0.1
max_reward_score_cyc = (lambda1_upper+1, 0, 0)
alpha = 0.99
avg_baseline = -1.0
lr1_start = 0.001
lr1_decay_rate = 0.96
lr1_decay_step = 5000
a = prior_knowledge_graph(true_graph, num_prior_edge, 0)
rewards_batches = []

# Learning
actor = Actor(max_length)
critic = Critic(max_length)  
opt_actor = torch.optim.Adam(actor.parameters(), lr=lr1_start, betas=(0.9, 0.99), eps=1e-08)
opt_critic = torch.optim.Adam(critic.parameters(), lr=lr1_start, betas=(0.9, 0.99), eps=1e-07)
lambdaf = lambda epoch: lr1_decay_rate ** (epoch/lr1_decay_step)
scd_actor = torch.optim.lr_scheduler.LambdaLR(opt_actor, lr_lambda=lambdaf)
scd_critic = torch.optim.lr_scheduler.LambdaLR(opt_critic, lr_lambda=lambdaf)
loss_fn = nn.MSELoss()
for i in (range(1, nb_epoch + 1)):
    opt_actor.zero_grad()
    opt_critic.zero_grad()
    input_batch = train_batch(inputdata, batch_size, input_dimension)
    graphs, scores, entropy, logprob = actor(torch.tensor(np.array(input_batch)).float())
    action = torch.reshape(scores.detach(), (batch_size, max_length**2))
    state_value = critic(action)
    callreward = get_Reward(batch_size, max_length, input_dimension, inputdata, sl, su, lambda1_upper, 'BIC', reg_type, 0.0)
    reward = callreward.cal_rewards(graphs, a, lambda1, lambda2, lambda3)[:,0]
    rewards_batches.append(np.mean(reward))
    avg_baseline = alpha*avg_baseline + (1.0-alpha)*np.mean(reward)
    discount_reward = torch.tensor(reward) - avg_baseline
    advantage = torch.tensor(discount_reward.detach()) - torch.squeeze(state_value)
    critic_loss = torch.mean(advantage**2)
    actor_loss = torch.mean(discount_reward.detach()*torch.mean(torch.stack(logprob),(0,2)))
    actor_loss.backward()
    critic_loss.backward()
    opt_actor.step()
    opt_critic.step()
    scd_actor.step()
    scd_critic.step()
    if (i+1) % lambda_iter_num == 0:
        ls_kv = callreward.update_all_scores(lambda1, lambda2, lambda3)
        graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]
        if cyc_min < 1e-5:
            lambda1_upper = score_min
        lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
        lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
        lambda3 = min(lambda3+lambda3_update_add, lambda3_upper)
        graph_batch = np.array([list(map(int, ((len(graph_int) - len(np.base_repr(curr_int))) * '0' + np.base_repr(curr_int)))) for curr_int in graph_int], dtype=int)
        graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, inputdata))
        # estimate accuracy
        acc_est = count_accuracy(true_graph, graph_batch_pruned.T)
        print(acc_est)
