import sys
sys.path.append("../")
import torch
from RLplayers.cnn_player import CNNPlayer
from RLplayers.cnn_network import CNNSimple
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from RLplayers.rl_utils import push_memory, load_net
from RLplayers.exprience_replay import ReplayMemory, Transition


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
    # print(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch, reward_batch)
    action_values = policy_net(state_batch)
    state_action_values = action_values.gather(1, action_batch)
    

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
    # print(action_probs)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss)
    # print(entropy)
    # entropy = np.sum(np.mean(action_probs) * np.log(action_probs))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    # print(loss)
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train_cnn(policy_net, target_net, memory, checkpoint_file, loss_file, losses=[], BATCH_SIZE=100, GAMMA=0.99, TARGET_UPDATE=10, round=1000):

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(),  lr=0.0000001)

    for _ in range(round):
        for i in range(TARGET_UPDATE):
            optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
        target_net.load_state_dict(policy_net.state_dict())
    torch.save({
                'state_dict': target_net.state_dict(),
            }, checkpoint_file)
    plt.clf()
    plt.plot(range(len(losses)), losses)
    plt.savefig(loss_file)
    return losses

if __name__ == '__main__':
    memory_file = "dqn_memory.json"
    checkpoint_file = "dqn_sl_checkpoint.pth.tar"
    loss_file = "dqn_sl_loss,pdf"
    network = CNNSimple
    policy_net = load_net(None, 65, network, False)
    target_net = load_net(None, 65, network, True)
    memory = ReplayMemory(1000000)
    push_memory(memory_file, memory)
    train_cnn(policy_net, target_net, memory, checkpoint_file, loss_file, round=4000)