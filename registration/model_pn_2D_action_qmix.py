import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import numpy as np

#from pointnet import PointNetfeat
from config import *
from mv import MVModel



class Agent(nn.Module):

    def __init__(self):
        super().__init__()

        self.Agent2D = Agent2D()
        self.Agent3D = Agent3D()
        self.Qmix = QMix()
        #self.bn0 = nn.BatchNorm1d(2048, momentum=0.1)
        #self.bn1 = nn.BatchNorm1d(2048, momentum=0.1)

    def forward(self, src, tgt):
        # O(src, tgt) -> S
        state_2d, action_2d, value_2d, emb_tgt_2d = self.Agent2D(src, tgt)
        state_3d, action_3d, value_3d, emb_tgt_3d = self.Agent3D(src, tgt)
        #state_3d = self.bn0(state_3d)
        #state_2d = self.bn1(state_2d)
        state = torch.cat((state_2d, state_3d), dim=1)
        #print("22222:", torch.sum(state_2d[0]), torch.mean(state_2d[0]),torch.max(state_2d[0]),torch.min(state_2d[0]))
        #print("33333:", torch.sum(state_3d[0]), torch.mean(state_3d[0]),torch.max(state_3d[0]),torch.min(state_3d[0]))
        value = torch.cat((value_2d.view(-1, 1, 1), value_3d.view(-1, 1, 1)), dim=2)
        #print(value_2d[0], value_3d[0])
        action_t = torch.cat((action_2d[0].view(action_2d[0].shape[0], -1, 1), action_3d[0].view(action_3d[0].shape[0], -1, 1)), dim = 2)
        action_r = torch.cat((action_2d[1].view(action_2d[1].shape[0], -1, 1), action_3d[1].view(action_3d[1].shape[0], -1, 1)), dim = 2)
        value_new, action_t_new, action_r_new = self.Qmix(value, state, action_t, action_r)
        state = [state_3d, state_2d]
        action = [action_t_new, action_r_new]
        value = value_new
        emb_tgt = [emb_tgt_3d, emb_tgt_2d]
        return state, action, value, emb_tgt




class QMix(nn.Module):
    def __init__(self):
        super(QMix, self).__init__()

        self.n_agents = 2
        self.state_dim = 4096

        self.embed_dim = 1024
        hypernet_embed = 1024
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, action_t, action_r):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1))
        hidden_t = F.elu(torch.bmm(action_t, w1))
        hidden_r = F.elu(torch.bmm(action_r, w1))
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        #print("V:", v[0])
        y = torch.bmm(hidden, w_final)
        #print("hidden_t.shape", hidden_t.shape)
        #print("w_final.shape", w_final.shape)
        #hidden_t.shape torch.Size([32, 33, 1024])
        #w_final.shape torch.Size([32, 1024, 1])
        #y_t.shape torch.Size([32, 33, 1])
        y_t = torch.bmm(hidden_t, w_final)
        y_r = torch.bmm(hidden_r, w_final)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        y_t = y_t.view(bs, 3, -1)
        y_r = y_r.view(bs, 3, -1)
        return q_tot, y_t, y_r



class Agent2D(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_emb_2d = StateEmbed2D()
        self.actor_critic2d = ActorCriticHead()

    def forward(self, src, tgt):
        # O(src, tgt) -> S
        state_2d, emb_tgt_2d = self.state_emb_2d(src, tgt)
        # S -> a, v
        action_2d, value_2d = self.actor_critic2d(state_2d)

        # reshape a to B x axis x [step, sign]
        action_t = action_2d[0]
        action_r = action_2d[1]
        action = [action_t, action_r]
        value = value_2d
        action = (action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
                  action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1))
        value = value.view(-1, 1, 1)
        state = state_2d
        emb_tgt = emb_tgt_2d
        return state, action, value, emb_tgt


class Agent3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.state_emb_3d = StateEmbed3D()
        self.actor_critic3d = ActorCriticHead()

    def forward(self, src, tgt):
        # O(src, tgt) -> S
        state_3d, emb_tgt_3d = self.state_emb_3d(src, tgt)
        # S -> a, v
        action_3d, value_3d = self.actor_critic3d(state_3d)
        # reshape a to B x axis x [step, sign]
        action_t = action_3d[0]
        action_r = action_3d[1]
        action = [action_t, action_r]
        value = value_3d
        action = (action[0].view(-1, 3, 2 * NUM_STEPSIZES + 1),
                  action[1].view(-1, 3, 2 * NUM_STEPSIZES + 1))
        value = value.view(-1, 1, 1)
        state = state_3d
        emb_tgt = emb_tgt_3d
        return state, action, value, emb_tgt



class StateEmbed2D(nn.Module):

    def __init__(self):
        super().__init__()
        #self.model = PointNetfeat(global_feat=True)
        self.mv_model = MVModel()
        #self.conv0 = nn.Conv1d(2048, 1024, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape
        emb_src =  self.mv_model(src)
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            emb_tgt = self.mv_model(tgt)
            #emb_tgt = self.conv0(emb_tgt.view(emb_tgt.shape[0], emb_tgt.shape[1], -1)).view(emb_tgt.shape[0], 1024)
        #print("emb_tgt.shape:", emb_tgt.shape)
        state = torch.cat((emb_src, emb_tgt), dim=-1)
        state = state.view(B, -1)
        return state, emb_tgt

class StateEmbed3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.convp1 = nn.Conv1d(IN_CHANNELS, 64, 1)
        self.convp2 = nn.Conv1d(64, 128, 1)
        self.convp3 = nn.Conv1d(128, 1024, 1)

    def forward(self, src, tgt):
        B, N, D = src.shape
        # O=(src,tgt) -> S=[Phi(src), Phi(tgt)]
        emb_src = self.embed(src.transpose(2, 1))
        if BENCHMARK and len(tgt.shape) != 3:
            emb_tgt = tgt  # re-use target embedding from first step
        else:
            emb_tgt = self.embed(tgt.transpose(2, 1))
        state = torch.cat((emb_src, emb_tgt), dim=-1)
        state = state.view(B, -1)
        return state, emb_tgt

    def embed(self, x):
        B, D, N = x.shape

        # embedding: BxDxN -> BxFxN
        x1 = F.relu(self.convp1(x))
        x2 = F.relu(self.convp2(x1))
        x3 = self.convp3(x2)

        # pooling: BxFxN -> BxFx1
        x_pooled = torch.max(x3, 2, keepdim=True)[0]
        return x_pooled.view(B, -1)


class ActorCriticHead(nn.Module):

    def __init__(self):
        super().__init__()
        STATE_DIM = FEAT_DIM
        self.activation = nn.ReLU()

        self.emb_r = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_r = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_t = nn.Sequential(
            nn.Linear(STATE_DIM, HEAD_DIM*2),
            self.activation,
            nn.Linear(HEAD_DIM*2, HEAD_DIM),
            self.activation
        )
        self.action_t = nn.Linear(HEAD_DIM, NUM_ACTIONS * NUM_STEPSIZES + NUM_NOPS)

        self.emb_v = nn.Sequential(
            nn.Linear(HEAD_DIM * 2, HEAD_DIM),
            self.activation
        )
        self.value = nn.Linear(HEAD_DIM, 1)

    def forward(self, state):
        # S -> S'
        emb_t = self.emb_t(state)
        emb_r = self.emb_r(state)
        # S' -> pi
        action_logits_t = self.action_t(emb_t)
        action_logits_r = self.action_r(emb_r)

        # S' -> v
        state_action = torch.cat([emb_t, emb_r], dim=1)
        emb_v = self.emb_v(state_action)
        value = self.value(emb_v)

        return [action_logits_t, action_logits_r], value


# -- action helpers
def action_from_logits(logits, deterministic=True):
    distributions = _get_distributions(*logits)
    actions = _get_actions(*(distributions + (deterministic,)))

    return torch.stack(actions).transpose(1, 0)


def action_stats(logits, action):
    distributions = _get_distributions(*logits)
    logprobs, entropies = _get_logprob_entropy(*(distributions + (action[:, 0], action[:, 1])))

    return torch.stack(logprobs).transpose(1, 0), torch.stack(entropies).transpose(1, 0)


def _get_distributions(action_logits_t, action_logits_r):
    distribution_t = Categorical(logits=action_logits_t)
    distribution_r = Categorical(logits=action_logits_r)

    return distribution_t, distribution_r


def _get_actions(distribution_t, distribution_r, deterministic=True):
    if deterministic:
        action_t = torch.argmax(distribution_t.probs, dim=-1)
        action_r = torch.argmax(distribution_r.probs, dim=-1)
    else:
        action_t = distribution_t.sample()
        action_r = distribution_r.sample()
    return action_t, action_r


def _get_logprob_entropy(distribution_t, distribution_r, action_t, action_r):
    logprob_t = distribution_t.log_prob(action_t)
    logprob_r = distribution_r.log_prob(action_r)

    entropy_t = distribution_t.entropy()
    entropy_r = distribution_r.entropy()

    return [logprob_t, logprob_r], [entropy_t, entropy_r]


# --- model helpers
def load(model, path):
    infos = torch.load(path)
    model.load_state_dict(infos['model_state_dict'])
    return infos


def save(model, path, infos={}):
    infos['model_state_dict'] = model.state_dict()
    torch.save(infos, path)


def plot_grad_flow(model):
    """
    via https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"no grad for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, -1, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=-1, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=torch.max(torch.stack(max_grads)).cpu())
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
