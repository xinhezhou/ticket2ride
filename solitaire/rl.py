

import json
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import sys
from solitaire import play_game
from sl import optimize_model
from convert_memory import convert_records
sys.path.append("..")
from RLplayers.cnn_network import CNNSimple
from RLplayers.exprience_replay import ReplayMemory
from RLplayers.cnn_player import CNNPlayer
from RLplayers.rl_utils import push_memory, load_net
from RLplayers.linear_player import DQNPlayer, QValueNetwork, compute_input_matrix
from game import Game




if __name__ == '__main__':
    """"
    TEST 2: without prior memory and checkpoint 
    """
    # memory = ReplayMemory(100000)
    # initial_checkpoint = "dqn_sl_checkpoint.pth.tar"
    # selfplay_checkpoint ="dqn_selfplay_sl.pth.tar"
    # record_file = "dqn_selfplay_sl_record.json"
    # memory_file = "dqn_selfplay_sl_memory.json"
    # loss_file = "dqn_selfplay_sl_loss.pdf"
    # policy_net = load_net(initial_checkpoint, 65, CNNSimple)
    # target_net = load_net(initial_checkpoint, 65, CNNSimple, eval=True)
        
    # losses = []
    # for _ in range(100):
    #     rewards, records = play_game(100, Game, CNNPlayer, target_net)
    #     with open(record_file, "w") as outfile:
    #         json.dump(records, outfile)
    #     print(sum(rewards))
    #     convert_records(record_file, memory_file)
    #     push_memory(memory_file, memory)
    #     losses = train_cnn(policy_net, target_net, memory, selfplay_checkpoint, loss_file, losses=losses, round = 30)
    #     policy_net = load_net(selfplay_checkpoint, 65, CNNSimple)
    #     target_net = load_net(selfplay_checkpoint, 65, CNNSimple, eval=True)
        
    # """
    # TEST 2: without prior memory and checkpoint 
    # """
    memory = ReplayMemory(100000)
    initial_checkpoint = None
    selfplay_checkpoint ="dqn_selfplay.pth.tar"
    record_file = "dqn_selfplay_record.json"
    memory_file = "dqn_selfplay_memory.json"
    loss_file = "dqn_selfplay_loss.pdf"
    policy_net = load_net(initial_checkpoint, 437, QValueNetwork)
    target_net = load_net(initial_checkpoint, 437, QValueNetwork, eval=True)
    optimizer = optim.RMSprop(policy_net.parameters(),  lr=0.0001)
        
    losses = []
    eps = 0.5
    for i in range(10000):
        if eps > 0.1:
            eps *= 0.999
        rewards, records = play_game(1, Game, DQNPlayer, model=policy_net, eps=eps)
        with open(record_file, "w") as outfile:
            json.dump(records, outfile)
        convert_records(record_file, memory_file, input_f=compute_input_matrix)
        push_memory(memory_file, memory)
        optimize_model(policy_net, target_net, optimizer, memory, losses, 100, 0.999)
        
        if i % 1000 == 0:
            rewards, records = play_game(1, Game, DQNPlayer, model=policy_net, eps=0)
            print(eps, sum(rewards)/len(rewards))
            target_net.load_state_dict(policy_net.state_dict())
            torch.save({
                'state_dict': target_net.state_dict(),
            }, selfplay_checkpoint)
            plt.clf()
            plt.plot(range(len(losses)), losses)
            plt.savefig(loss_file)
                
