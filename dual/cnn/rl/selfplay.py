

import sys
sys.path.append("../")
from cnn_utils import load_net
from dataset.convert_memory import convert_records
from cnn_player import CNNPlayer
from nonRLplayers.random_player import RandomPlayer
from sl.cnn import train_cnn, push_memory
from evaluation.evaluation import evaluate_net
from exprience_replay import ReplayMemory

def train_selfplay(initial_checkpoint, selfplay_checkpoint, record_file, loss_file, memory, player_classes, destination_cards, opponent_net=None, first=True, round=10):
    policy_net = load_net(initial_checkpoint)
    target_net = load_net(initial_checkpoint, True)
    for _ in range(round):
        train_cnn(policy_net, target_net, memory, selfplay_checkpoint, loss_file)

    target_nets = [target_net, load_net(opponent_net)]
    if not first:
        target_nets = target_nets[:]
    evaluate_net(target_nets, player_classes, destination_cards,record_file, counter=100, display=False)



"""
TEST 1: with prior memory and checkpoint 
"""
memory = ReplayMemory(100000)
initial_memory = "../dataset/medium_memory.json"
push_memory(initial_memory, memory)
initial_checkpoint = "../sl/medium_m.pth.tar"
selfplay_checkpoint ="with_prior_random.pth.tar"
record_file = "../dataset/with_prior_random.json"
memory_file = "../dataset/with_prior_random.json"
player_classes = [
    CNNPlayer, RandomPlayer
]
destination_cards = [
    [[1, 3], [1, 6]],
    [[0, 6], [3, 6]],
]
train_selfplay(initial_checkpoint, selfplay_checkpoint, record_file, memory_file, player_classes, destination_cards)
for i in range(100):
    train_selfplay(selfplay_checkpoint, selfplay_checkpoint, record_file, memory_file, player_classes, destination_cards)
    convert_records(record_file, memory_file)

