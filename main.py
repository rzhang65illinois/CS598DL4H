import numpy as np
import torch
import torch.nn as nn

from data_loader import DataGenerator_read_data
from model import Actor, Critic
from reward import get_Reward
from utils.load import load_data, train_batch, prior_knowledge_graph
from utils.eval import graph_prunned_by_coef, count_accuracy
from utils.bic import BIC_lambdas

# Input parameters
#max_length = 8     # Total number of nodes in the graph
#data_size = 1000   # Sample size
reg_type = 'LR'
nb_epoch = 10000                 
input_dimension = 64
lambda_iter_num = 1000
batch_size = 32

output_dir = '.../CS598DL4H/datasets/SACHS' # please update the directory!
save_model_path = '{}/model'.format(output_dir)
plot_dir = '{}/plot'.format(output_dir)
graph_dir = '{}/graph'.format(output_dir)

file_path = '{}/data.npy'.format(output_dir)
#solution_path = '{}/DAG.npy'.format(config.data_path)
solution_path = None
inputdata, true_graph = load_data(file_path, solution_path, True)

#sl, su, strue = BIC_lambdas(inputdata, None, None, true_graph.T, reg_type)
import matplotlib.pyplot as plt
sl, su, strue = BIC_lambdas(inputdata, None, None, None, reg_type)
lambda1 = 0
lambda1_upper = 5
lambda1_update_add = 1
lambda2 = 1/(10**(np.round(max_length/3)))
lambda2_upper = 0.01
lambda2_update_mul = 10
lambda3 = 0
lambda3_upper = 1
lambda3_update_add = 0.1#

max_reward_score_cyc = (lambda1_upper+1, 0, 0)
rewards_avg_baseline, rewards_batches, reward_max_per_batch = [], [], []
lambda1s, lambda2s, lambda3s = [], [], []
graphss = []
probsss = []
max_rewards = []
max_reward = float('-inf')
image_count = 0
image_count2= 0
accuracy_res, accuracy_res_pruned = [], []
max_reward_score_cyc = (lambda1_upper+1, 0, 0)
alpha = 0.99
avg_baseline = -1.0
lr1_start = 0.001
lr1_decay_rate = 0.96
lr1_decay_step = 5000
a = prior_knowledge_graph(true_graph, 4, 26)
max_length = 11
#3, 36
#4, 121*0.25 - 4 = 26
'''
a = np.ones((max_length, max_length))*2
a= np.int32(a)
a[0][2]=1
a[10][9]=1
a[8][11]=1
a[5][0]=0
a[2][1]=0
a[6][2]=0
a[9][3]=0
a[1][4]=0
a[3][5]=0
a[1][6]=0
a[8][7]=0
a[4][8]=0
a[11][9]=0
a[4][10]=0
a[9][11]=0
'''
'''
#Smoking
true_graph[1][0]=1
true_graph[11][0]=1
#Anxiety
true_graph[0][2]=1
#Peer presure
true_graph[0][3]=1
#Genetics
true_graph[5][4]=1
true_graph[11][4]=1
#Attention disorder
true_graph[7][5]=1
#Fatigue
true_graph[7][8]=1
#Allergy
true_graph[10][9]=1
#Coughing
true_graph[8][10]=1
#Lung cancer
true_graph[8][11]=1
true_graph[10][11]=1
'''
'''
a = np.ones((8, 8))*2
a= np.int32(a)
a[6][2]=1
a[4][3]=1
a[1][5]=0
a[6][7]=0
true_graph = np.zeros((8, 8))
true_graph[1][0]=1
true_graph[3][2]=1
true_graph[6][2]=1
true_graph[4][3]=1
true_graph[7][6]=1
true_graph[7][4]=1
true_graph[5][4]=1
true_graph[4][1]=1
'''
rewards_batches, max_rewards = [], []
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
    #print(graphs.shape, scores.shape, entropy[0].shape, logprob[0].shape)
    #print(scores.shape)
    action = torch.reshape(scores.detach(), (batch_size, max_length**2))
    state_value = critic(action)
    callreward = get_Reward(batch_size, max_length, input_dimension, inputdata, sl, su, lambda1_upper, 'BIC', reg_type, 0.0)
    reward_tuple = callreward.cal_rewards(graphs, a, lambda1, lambda2, lambda3)
    reward = reward_tuple[:,0]
    rewards_batches.append(np.mean(reward))
    avg_baseline = alpha*avg_baseline + (1.0-alpha)*np.mean(reward)
    discount_reward = torch.tensor(reward) - avg_baseline
    advantage = torch.tensor(reward) - torch.squeeze(state_value)
    critic_loss = torch.mean(advantage.pow(2))
    #actor_loss = torch.sum(-torch.mean(torch.stack(logprob),(0,2)) * advantage.detach())
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
        #print(graph_batch.T, graph_batch_pruned.T)
        acc_est = count_accuracy(true_graph, graph_batch)
        acc_est2 = count_accuracy(true_graph, graph_batch_pruned)
        print(acc_est, acc_est2)
        plt.figure(1)
        plt.plot(rewards_batches, label='reward per batch')
        plt.legend()
        plt.show()
        plt.close()
