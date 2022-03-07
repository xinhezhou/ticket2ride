import sys
sys.path.append("../")
from cnn_network import CNNComplex, CNNSimple
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_utils import push_memory, load_net
from exprience_replay import ReplayMemory, Transition
import torch.nn as nn

def optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    next_state_batch = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch, reward_batch)
    action_probs = policy_net(state_batch)
    state_action_values = action_probs.gather(1, action_batch)
    

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print(state_action_values, expected_state_action_values)
    entropy = 0
    # print(action_probs)
    for i in range(action_probs.shape[0]):
        probs_list = action_probs[i][action_probs[i] != 0]
        entropy -= torch.sum(probs_list * torch.log(probs_list))
        if torch.isnan(entropy):
            print("what")
            print(probs_list, torch.log(probs_list))
            return 
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) + 0.0001 * entropy
    # print(entropy)
    # entropy = np.sum(np.mean(action_probs) * np.log(action_probs))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train_cnn(policy_net, target_net, memory, checkpoint_file, loss_file, BATCH_SIZE=128, GAMMA=0.9, TARGET_UPDATE=10, round=1000):

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    losses = []
    optimizer = optim.RMSprop(policy_net.parameters())

    losses = []

    for _ in range(round):
        for i in range(TARGET_UPDATE):
            optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
        target_net.load_state_dict(policy_net.state_dict())
    torch.save({
                'state_dict': target_net.state_dict(),
            }, checkpoint_file)
    plt.plot(range(len(losses)), losses)
    plt.savefig(loss_file)


datasets = [
    ["../dataset/small_memory.json", "small_m.pth.tar", "small_f.pth.tar", "small_loss_m.pdf", "small_loss_f.pdf"],
    ["../dataset/medium_memory.json", "medium_m.pth.tar", "medium_f.pth.tar", "medium_loss_m.pdf", "medium_loss_f.pdf"],
    ["../dataset/large_memory.json", "large_m.pth.tar", "large_f.pth.tar", "large_loss_m.pdf", "large_loss_f.pdf"],
]
for memory_file, checkpoint_m, checkpoint_f, loss_m, loss_f in datasets:
    network = CNNSimple
    policy_net = load_net(None, network, False)
    target_net = load_net(None, network, True)
    memory = ReplayMemory(1000000)
    push_memory(memory_file, memory)
    train_cnn(policy_net, target_net, memory, checkpoint_m, loss_f)
    train_cnn(policy_net, target_net, memory, checkpoint_m, loss_f)