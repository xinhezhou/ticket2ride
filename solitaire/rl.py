

import json
import sys
from solitaire import play_game
from sl import train_cnn
from convert_memory import convert_records
sys.path.append("..")
from RLplayers.cnn_network import CNNSimple
from RLplayers.exprience_replay import ReplayMemory
from RLplayers.cnn_player import CNNPlayer
from RLplayers.rl_utils import push_memory, load_net
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
    policy_net = load_net(initial_checkpoint, 65, CNNSimple)
    target_net = load_net(initial_checkpoint, 65, CNNSimple, eval=True)
        
    losses = []
    for _ in range(100):
        rewards, records = play_game(100, Game, CNNPlayer, target_net, eps=0.2)
        with open(record_file, "w") as outfile:
            json.dump(records, outfile)
        print(sum(rewards))
        convert_records(record_file, memory_file)
        push_memory(memory_file, memory)
        losses = train_cnn(policy_net, target_net, memory, selfplay_checkpoint, loss_file, losses=losses, round = 30)
        policy_net = load_net(selfplay_checkpoint, 65, CNNSimple)
        target_net = load_net(selfplay_checkpoint, 65, CNNSimple, eval=True)
        
