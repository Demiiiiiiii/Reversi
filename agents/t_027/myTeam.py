from template import Agent
import random


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self,actions,game_state):
        return random.choice(actions)
from template import Agent
from template import GameState, GameRule, Agent
import random
import numpy as np
import copy
import operator
import time
from Reversi.reversi_utils import Cell,filpColor,boardToString,countScore, GRID_SIZE

        
class MCTS():
    def __init__(self, state, mycolor, parent=None, parent_action=None):
        self.state = state
        self.agentcolor = mycolor
        self.parent = parent
        self.parent_action = parent_action
        self.validPos = self._validPos()
        self.children = []
        self.number_of_visits = 0
        self.results =  {-1: 0, 0: 0, 1: 0}
        self.UntriedActions = None
        self.UntriedActions = self.untried_actions()
        return
        
    def _validPos(self):
        pos_list = []
        for x in range(8):
            for y in range(8):
                pos_list.append((x,y))
        return pos_list
            
    def untried_actions(self):
        self.UntriedActions = self.getLegalActions(self.state, self.agentcolor)
        return self.UntriedActions
        
    def q(self):
        wins = self.results[1]
        loses = self.results[-1]
        if self.agentcolor == 1:
            return loses - wins
        else:
            return wins - loses
        
    def node_visit_num(self):
        return self.number_of_visits
        
    def reverse_color(self,color):
        if color == 0:
            return 1
        else:
            return 0
        
    def expand(self):
	    
        action = self.UntriedActions.pop()
        color = int(self.agentcolor)
        #MOVE
        next_state = self.generateSuccessor(self.state, action,color)
        color = self.reverse_color(color)
        child_node = MCTS(next_state,color, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
        
        
    def is_terminal_node(self):
        #over
        return self.GameOver(self.state)
        
        
    def rollout(self):
        current_rollout_state = self.state
        current_color = self.agentcolor
        while not self.GameOver(current_rollout_state):
            
            possible_moves = self.getLegalActions(current_rollout_state, current_color)
            
            action = self.rollout_policy(possible_moves)
            current_rollout_state = self.generateSuccessor(current_rollout_state, action,current_color)
            current_color = self.reverse_color(current_color)

        return self.countScore(current_rollout_state) #game over
    
    def backpropagate(self, result):
        self.number_of_visits += 1.
        self.results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)
            
    def is_fully_expanded(self):
        return len(self.UntriedActions) == 0            
            
    def best_child(self, c_param=0.1):
        
        choices = [(c.q() / c.node_visit_num()) + c_param * np.sqrt((2 * np.log(self.node_visit_num()) / c.node_visit_num())) for c in self.children]
        return self.children[np.argmax(choices)]    
            
    def rollout_policy(self, possible_moves):
        
        return possible_moves[np.random.randint(len(possible_moves))]
        
        
    def TreePolicy(self):

        current_node = self
        while not current_node.is_terminal_node():
            
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
                
                
    def BestAction(self,time_start):
        ttt = 0
        time_now = time.time() - time_start
        while (time_now < 0.9) :
            #print(time_now)
            v = self.TreePolicy()
            reward = v.rollout()
            v.backpropagate(reward)
            ttt+=1
            time_now = time.time() - time_start
        #print(ttt)
        return self.best_child(c_param=0.1).parent_action   
            
    def getLegalActions(self, game_state, agent_color):
        actions = []
        # print(f"Current game state: \n{boardToString(game_state.board,GRID_SIZE)}")
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if game_state[x][y] == -1:
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        if temp_pos in self.validPos and game_state[temp_pos[0]][temp_pos[1]] != -1 and game_state[temp_pos[0]][temp_pos[1]] != agent_color:
                            while temp_pos in self.validPos:
                                if game_state[temp_pos[0]][temp_pos[1]] == -1:
                                    break
                                if game_state[temp_pos[0]][temp_pos[1]] == agent_color:
                                    actions.append(pos)
                                    break
                                temp_pos = tuple(map(operator.add,temp_pos,direction))
        if len(actions) == 0:
            actions.append("Pass")
        return actions
            
    def GameOver(self,state):
        if self.getLegalActions(state,0) == ["Pass"] and self.getLegalActions(state,1) == ["Pass"]:
             return True
        else: return False
            
    def countScore(self,board):
        player_color = 1
        score = 0
        oppo = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == player_color:
                    score += 1
                elif board[i][j] == oppo:
                    score -= 1
        if score < 0:
            score = -1
        elif score > 0:
            score = 1
        else:
            score = 0
        return score
        
    def generateSuccessor(self, state, action, agent_color):
        if action == "Pass":
            return state
        else:
            next_state = copy.deepcopy(state)
            update_color = agent_color
            next_state[action[0]][action[1]] = update_color
            # iterate over all 8 directions and check pieces that require updates
            for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                cur_pos = (action[0] + direction[0], action[1] + direction[1])
                update_list = list()
                flag = False
                # Only searching for updates if the next piece in the direction is from the agent's opponent
                while cur_pos in self.validPos and next_state[cur_pos[0]][cur_pos[1]] != -1:
                    if next_state[cur_pos[0]][cur_pos[1]] == update_color:
                        flag = True
                        break
                    update_list.append(cur_pos)
                    cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
                if flag and len(update_list) != 0:
                    for i,j in update_list:
                        next_state[i][j] = update_color
            return next_state

            
class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.validPos = self._validPos()
        self.weight = [     [ 4, -3, 2, 2, 2, 2 ,-3, 4 ],
                            [-3, -4, -1, -1, -1, -1, -4, -3],
                            [2, -1, 1, 0, 0, 1, -1, 2 ],
                            [2, -1, 0, 1, 1, 0, -1, 2 ],
                            [2, -1, 0, 1, 1, 0, -1, 2 ],
                            [2, -1, 1, 0, 0, 1, -1, 2 ],
                            [-3, -4, -1, -1, -1, -1, -4, -3 ],
                            [4, -3, 2, 2, 2, 2, -3, 4]]
                            
        self.t = 0
        self.turn = 0
        

    def countWeight(self,board,grid_size,player_color):
        score = 0
        oppo = 0
        if player_color == 0:
            oppo = 1
        for i in range(grid_size):
            for j in range(grid_size):
                if board[i][j] == player_color:
                    score += self.weight[i][j]
                elif board[i][j] == oppo:
                    score -= self.weight[i][j]
                    
        return score

        
    def _validPos(self):
        pos_list = []
        for x in range(8):
            for y in range(8):
                pos_list.append((x,y))
        return pos_list
        
    def generateSuccessor(self, state, action, agent_color):
        if action == "Pass":
            return state
        else:
            next_state = copy.deepcopy(state)
            update_color = agent_color
            next_state[action[0]][action[1]] = update_color
            # iterate over all 8 directions and check pieces that require updates
            for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                cur_pos = (action[0] + direction[0], action[1] + direction[1])
                update_list = list()
                flag = False
                # Only searching for updates if the next piece in the direction is from the agent's opponent
                while cur_pos in self.validPos and next_state[cur_pos[0]][cur_pos[1]] != -1:
                    if next_state[cur_pos[0]][cur_pos[1]] == update_color:
                        flag = True
                        break
                    update_list.append(cur_pos)
                    cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
                if flag and len(update_list) != 0:
                    for i,j in update_list:
                        next_state[i][j] = update_color
            return next_state


    def getLegalActions(self, game_state, agent_color):
        actions = []
        # print(f"Current game state: \n{boardToString(game_state.board,GRID_SIZE)}")
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if game_state[x][y] == -1:
                    pos = (x,y)
                    for direction in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                        temp_pos = tuple(map(operator.add,pos,direction))
                        if temp_pos in self.validPos and game_state[temp_pos[0]][temp_pos[1]] != -1 and game_state[temp_pos[0]][temp_pos[1]] != agent_color:
                            while temp_pos in self.validPos:
                                if game_state[temp_pos[0]][temp_pos[1]] == -1:
                                    break
                                if game_state[temp_pos[0]][temp_pos[1]] == agent_color:
                                    actions.append(pos)
                                    break
                                temp_pos = tuple(map(operator.add,temp_pos,direction))
        if len(actions) == 0:
            actions.append("Pass")
        return actions
    
    def abpruning(self,previous_action,actions,game_state,depth,myid,current_action,alpha,beta):
        mycolor = myid
        nextd = depth / len(actions)
        if depth < 1 or nextd < 0.2:
            score = self.countWeight(game_state,8,mycolor)
            if (self.weight[0][0]!=1):
                score += len(actions) - len(previous_action)
            if myid == 1:
                score = -score
            return (score,current_action)
        elif actions[0] == "Pass" and current_action == "Pass":
            score = self.countWeight(game_state,8,mycolor)
            if myid == 1:
                score = -score
            return (score,current_action)
            
            
        reverse_id = 0
        current_v = 9999
        best_action = current_action
        if myid == 0:
            reverse_id = 1
            current_v = -9999

        for act in actions:
            self.t += 1
            one_move = self.generateSuccessor(game_state, act, mycolor)
            r_actions = self.getLegalActions(one_move, reverse_id)
            v = self.abpruning(actions,r_actions,one_move,nextd,reverse_id,act,alpha,beta)
            if (myid == 0):
                if v[0]>current_v:
                    current_v = v[0]
                    best_action = act
                if current_v > beta:
                    break
                if current_v>alpha:
                    alpha = current_v
            else:
                if v[0]<current_v:
                    current_v = v[0]
                    best_action = act
                if current_v<alpha:
                    break
                if current_v<beta:
                    beta = current_v
             
            if (alpha>=beta):
                break
        return (current_v,best_action)
    
    def SelectAction(self,actions,game_state):
        current_t = time.time()
        simple_board = [[j.value for j in i] for i in game_state.board]
        color = self.id
        color = game_state.agent_colors[color].value
        self.turn+=1
        self.t = 0
        if self.turn > 27 or self.turn < 3:
            choice = self.abpruning(actions,actions,simple_board,800,color,"NULL",-9999,9999)
            print(time.time() - current_t,self.t)
            return choice[1]

        
        root = MCTS(state = simple_board,mycolor=color)
        act = root.BestAction(current_t)
        print(time.time() - current_t)
        return act