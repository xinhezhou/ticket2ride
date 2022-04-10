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
        # return
        transitions = memory.get_all()
    else:
        transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))


    next_state_batch = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # print(batch.reward)


    action_values = policy_net(state_batch)
    state_action_values = action_values.gather(2, action_batch.unsqueeze(-1))


    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values = target_net(next_state_batch).max(2)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.unsqueeze(-1)



    criterion = nn.SmoothL1Loss()
    
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = criterion(state_action_values,  expected_state_action_values.unsqueeze(1))
    

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def train_cnn(policy_net, target_net, memory, checkpoint_file, loss_file, losses=[], BATCH_SIZE=100, GAMMA=0.999, TARGET_UPDATE=10, round=1000):

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(),  lr=0.00001)

    for _ in range(round):
        optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
        
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