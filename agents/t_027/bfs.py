from template import Agent
from Reversi.reversi_model import ReversiGameRule
from collections import deque
import random, time

LIMIT_TIME = 0.93
PRIOR_CEIL = [(0,0),(0,7),(7,0),(7,7)]
SUB_CEIL = [(1,1),(6,1),(6,6),(1,6)]
class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = ReversiGameRule(2)
    def GetSuccessorActions(self,game_state,agent_id):
        return self.game_rule.getLegalActions(game_state, agent_id)
    def ExecuteResult(self,game_state,action,agent_id):
        result_state = self.game_rule.generateSuccessor(game_state, action, agent_id)
        result_score = self.game_rule.calScore(result_state, agent_id)
        return (result_state, result_score)
    def SelectAction(self,actions,game_state):
        if self.game_rule.agent_colors is None:
            self.game_rule.agent_colors = game_state.agent_colors
        start_time = time.time()
        d = deque([(game_state, [])])
        max_score = -1
        rival_max_score = -1
        solution = random.choice(actions)
        while time.time() - start_time < LIMIT_TIME and len(d):
            start, path = d.popleft()
            new_actions = [x for x in actions if x not in SUB_CEIL]
            prior_actions = [x for x in new_actions if x in PRIOR_CEIL]
            if len(prior_actions) > 0:
                new_actions = prior_actions
                if len(prior_actions) == 1:
                    return prior_actions[0]
            if len(new_actions) < 1:
                new_actions = actions
            for action in new_actions:
                new_path = path + [action]
                new_state, new_score = self.ExecuteResult(game_state, action, self.id)
                if time.time() - start_time > LIMIT_TIME:
                    break
                if self.game_rule.gameEnds:
                    if max_score < new_score:
                        max_score = new_score
                        solution = new_path[0]
                        continue
                rival_new_actions = self.GetSuccessorActions(new_state, 1-self.id)
                for action in rival_new_actions:
                    rival_next_state, rival_next_score = self.ExecuteResult(new_state, action, 1-self.id)
                    rival_best_state = rival_next_state
                    if rival_max_score < rival_next_score:
                        rival_max_score = rival_next_score
                        rival_best_state = rival_next_state
                d.append((rival_best_state, new_path))

        return solution