"""
Taken from: https://github.com/SamRagusa/Checkers-Reinforcement-Learning

MIT License

Copyright (c) 2017 Sam Ragusa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
    Contributor: Batuhan Erden
    Contribution: @see report.pdf
"""


from Board import Board
import matplotlib.pyplot as plt


class Player:
    """
    A class to be inherited by any class representing a checkers player.
    This is used so that other functions can be written for more general use,
    without worry of crashing (e.g. play_n_games).
    
    NOTES:
    1) Create set playerID method
    """
    
    def set_board(self, the_board):
        """
        Sets the Board object which is known by the AI.
        """
        self.board = the_board
    
    def game_completed(self):
        """
        Should be overridden if AI implementing this class should be notified 
        of when a game ends, before the board is wiped.
        """
        pass
    
    def get_next_move(self):
        """
        Gets the desired next move from the AI.
        """
        pass


class DP_Value_Iteration(Player):

    def __init__(self, the_player_id, the_board=None):
        self.player_id = the_player_id
        self.board = the_board
        self.value_func = {}
        self.number_of_new_states = 0

    def game_completed(self):
        print("\nGame completed!")
        print("Board status: %s" % str(self.board.spots))
        print("Players' status: %s" % str(get_number_of_pieces_and_kings(self.board.spots)))
        
        if self.number_of_new_states != 0:
            print("Saving %d new states.." % self.number_of_new_states)
            self.board.save(self.value_func, "value_func.pkl")
            print("Saved!")
        
        self.number_of_new_states = 0

    def get_next_move(self):
        self.value_func = self.value_iteration() if len(self.value_func) == 0 else self.value_func
        
        action = self.policy(self.board.spots)
        next_state = self.board.get_potential_spots_from_moves([action])[0]

        self.add_new_incoming_states(self.board.spots, next_state)
        return action

    def add_new_incoming_states(self, *states):
        for state in states:
            if self.board.make_tuple(state) not in self.value_func:
                self.number_of_new_states += 1
                self.value_func[self.board.make_tuple(state)] = 0

    def value_iteration(self, epsilon=0.0001):
        value_func = self.board.load("value_func.pkl") if len(self.value_func) == 0 else self.value_func
        print("Value iteration started with %d possible states, its aim is to converge.." % len(value_func))

        while True:
            delta = 0

            for state in value_func:
                v = value_func[state]
                value_func[state] = self.bellman(value_func, self.board.make_list(state))
                delta = max(delta, abs(v - value_func[state]))

            print("delta: %.8f" % delta)

            if delta <= epsilon:
                print("Convergence achieved, value iteration finished!")
                self.board.save(value_func, "value_func.pkl")

                return value_func

    def policy(self, state):
        board = Board(state, self.board.player_turn)

        actions = board.get_possible_next_moves()
        next_states = board.get_potential_spots_from_moves(actions)

        max_gain = 0
        best_possible_action = actions[0]

        for i in range(len(actions)):
            current_gain = self.value_func.get(self.board.make_tuple(next_states[i]), 0)

            if current_gain >= max_gain:
                max_gain = current_gain
                best_possible_action = actions[i]

        return best_possible_action

    def bellman(self, value_func, state, gamma=0.1):
        prob, max_gain = self.get_prob_and_max_gain_from_next_states(value_func, state)
        return self.get_reward(state) + prob * gamma * max_gain

    def get_prob_and_max_gain_from_next_states(self, value_func, state):
        prob = 1
        max_gain = 0
        next_states = self.get_possible_next_states(state, self.board.player_turn)

        for next_state in next_states:
            opp_next_states = self.get_possible_next_states(next_state, not self.board.player_turn)
            sigma = sum(value_func.get(self.board.make_tuple(opp_next_state), 0) for opp_next_state in opp_next_states)

            if sigma >= max_gain:
                prob = 1 if len(opp_next_states) == 0 else 1 / len(opp_next_states)
                max_gain = sigma

        return prob, max_gain

    def get_reward(self, state):
        """
        State Info: [P1_pieces, P2_pieces, P1_kings, P2_kings]
        """
        state_info = self.get_state_info(state)
        next_states = self.get_possible_next_states(state, self.board.player_turn)
        terminal_state = True if len(next_states) == 0 else False

        if terminal_state:
            if state_info[1] == 0 and state_info[3] == 0:  # Won
                return 100
            if state_info[0] == 0 and state_info[2] == 0:  # Lost
                return -100

            return 0  # Draw

        return 1.5 * (state_info[0] + 3.33 * state_info[2] - state_info[1] - 3.33 * state_info[3])

    def get_state_info(self, state):
        board = Board(state, self.board.player_turn)
        state_info = get_number_of_pieces_and_kings(board.spots)

        return state_info

    @staticmethod
    def get_possible_next_states(state, player_turn):
        board = Board(state, player_turn)
        actions = board.get_possible_next_moves()

        if len(actions) == 0:
            next_states = []
        else:
            next_states = board.get_potential_spots_from_moves(actions)

        return next_states


def get_number_of_pieces_and_kings(spots, player_id=None):
    """
    Gets the number of pieces and the number of kings that each player has on the current 
    board configuration represented in the given spots. The format of the function with defaults is:
    [P1_pieces, P2_pieces, P1_kings, P2_kings]
    and if given a player_id:
    [player_pieces, player_kings]
    """
    piece_counter = [0,0,0,0]  
    for row in spots:
        for element in row:
            if element != 0:
                piece_counter[element-1] = piece_counter[element-1] + 1
    
    if player_id == True:
        return [piece_counter[0], piece_counter[2]]
    elif player_id == False:
        return [piece_counter[1], piece_counter[3]]
    else:
        return piece_counter


class Alpha_beta(Player):
    """
    A class representing a checkers playing AI using Alpha-Beta pruning.   
    
    TO DO:
    1) Be able to take in any reward function (for when not win/loss) 
    so that you can make a more robust set of training AI
    """
    
    def __init__(self, the_player_id, the_depth, the_board=None):
        """
        Initialize the instance variables to be stored by the AI. 
        """
        self.board = the_board
        self.depth = the_depth
        self.player_id = the_player_id

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """
        A method implementing alpha-beta pruning to decide what move to make given 
        the current board configuration. 
        """
        if board.is_game_over():
            if get_number_of_pieces_and_kings(board.spots, board.player_turn) == [0,0]:
                if maximizing_player:
                    #Using integers instead of float("inf") so it's less than float("inf") not equal to
                    return -10000000, None
                else:
                    return 10000000, None
            elif get_number_of_pieces_and_kings(board.spots, not board.player_turn) == [0,0]:
                if maximizing_player:
                    return 1000000, None
                else:
                    return -1000000, None
            else:
                return 0, None

        if depth == 0:
            players_info = get_number_of_pieces_and_kings(board.spots)
            if board.player_turn != maximizing_player:
                return  players_info[1] + 2 * players_info[3] - (players_info[0] + 2 * players_info[2]), None
            return  players_info[0] + 2 * players_info[2] - (players_info[1] + 2 * players_info[3]), None
        possible_moves = board.get_possible_next_moves()

        potential_spots = board.get_potential_spots_from_moves(possible_moves)
        desired_move_index = None
        if maximizing_player:
            v = float('-inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, False)
                if v < alpha_beta_results[0]: 
                    v = alpha_beta_results[0]
                    alpha = max(alpha, v)
                    desired_move_index = j
                if beta <= alpha: 
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
        else:
            v = float('inf')
            for j in range(len(potential_spots)):
                cur_board = Board(potential_spots[j], not board.player_turn)
                alpha_beta_results = self.alpha_beta(cur_board, depth - 1, alpha, beta, True)
                if v > alpha_beta_results[0]:  
                    v = alpha_beta_results[0]
                    desired_move_index = j
                    beta = min(beta, v)
                if beta <= alpha:
                    break
            if desired_move_index is None:
                return v, None
            return v, possible_moves[desired_move_index]
    
    def get_next_move(self):
        return self.alpha_beta(self.board, self.depth, float('-inf'), float('inf'), self.player_id)[1]


def play_n_games(player1, player2, num_games, move_limit):
    """
    Plays a specified amount of games of checkers between player1, who goes first,
    and player2, who goes second.  The games will be stopped after the given limit on moves.
    This function outputs an array of arrays formatted as followed (only showing game 1's info):
    [[game1_outcome, num_moves, num_own_pieces, num_opp_pieces, num_own_kings, num_opp_kings]...]
    gameN_outcome is 0 if player1 won, 1 if lost, 2 if tied, and 3 if hit move limit.
    
    PRECONDITIONS:
    1)Both player1 and player2 inherit the Player class
    2)Both player1 and player2 play legal moves only
    """
    game_board = Board()
    player1.set_board(game_board)
    player2.set_board(game_board)
     
    players_move = player1
    outcome_counter = [[-1,-1,-1,-1,-1,-1] for j in range(num_games)] 
    for j in range(num_games):
        #print(j)
        move_counter = 0
        while not game_board.is_game_over() and move_counter < move_limit:
            game_board.make_move(players_move.get_next_move())
             
            move_counter = move_counter + 1
            if players_move is player1:
                players_move = player2
            else:
                players_move = player1
        else:
            piece_counter = get_number_of_pieces_and_kings(game_board.spots)
            if piece_counter[0] != 0 or piece_counter[2] != 0:
                if piece_counter[1] != 0 or piece_counter[3] != 0:
                    if move_counter == move_limit:
                        outcome_counter[j][0] = 3
                    else:
                        outcome_counter[j][0] = 2
#                     if (j+1)%100==0:
#                         print("Tie game for game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
                else:
                    outcome_counter[j][0] = 0
#                     if (j+1)%100==0:
#                         print("Player 1 won game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
            else:
                outcome_counter[j][0] = 1
#                 if (j+1)%100==0:
#                     print("Player 2 won game #" + str(j + 1) + " in " + str(move_counter) + " turns!")
                
            outcome_counter[j][1] = move_counter
            outcome_counter[j][2] = piece_counter[0]
            outcome_counter[j][3] = piece_counter[1]
            outcome_counter[j][4] = piece_counter[2]
            outcome_counter[j][5] = piece_counter[3]
             
            player1.game_completed()
            player2.game_completed()
            #game_board.print_board()
            game_board.reset_board()
     
    return outcome_counter


def pretty_outcome_display(outcomes):
    """
    Prints the outcome of play_n_games in a easy to understand format.
    
    TO DO:
    1) Add functionality for pieces in each game
    2) Add ability to take other strings for AI info and display it
    """
    game_wins = [0,0,0,0]
    total_moves = 0
    max_moves_made = float("-inf")
    min_moves_made = float("inf")
    for outcome in outcomes:
        total_moves = total_moves + outcome[1]
        if outcome[1] < min_moves_made:
            min_moves_made = outcome[1]
        if outcome[1] > max_moves_made:
            max_moves_made = outcome[1]
        
        game_wins[outcome[0]] = game_wins[outcome[0]] + 1
    
    print("Games Played: ".ljust(35), len(outcomes))
    print("Player 1 wins: ".ljust(35), game_wins[0])
    print("Player 2 wins: ".ljust(35), game_wins[1])
    print("Games exceeded move limit: ".ljust(35), game_wins[3])
    print("Games tied: ".ljust(35), game_wins[2])
    print("Total moves made: ".ljust(35), total_moves)  
    print("Average moves made: ".ljust(35), total_moves/len(outcomes))
    print("Max moves made: ".ljust(35), max_moves_made)
    print("Min moves made: ".ljust(35), min_moves_made)


def plot_end_game_information(outcome, interval, title="End of Game Results"):
    """
    """
    player1_wins = [0 for _ in range(int(len(outcome)/interval))]
    player2_wins = [0 for _ in range(int(len(outcome)/interval))]
    ties = [0 for _ in range(int(len(outcome)/interval))]
    move_limit = [0 for _ in range(int(len(outcome)/interval))]
    
    for j in range(int(len(outcome)/interval)):
        for i in range(interval):
            if outcome[j*interval + i][0] == 0:
                player1_wins[j] = player1_wins[j] + 1
            elif outcome[j*interval + i][0] == 1:
                player2_wins[j] = player2_wins[j] + 1
            elif outcome[j*interval + i][0] == 2:
                ties[j] = ties[j] + 1
            else:
                move_limit[j] = move_limit[j] + 1
                
    plt.figure(title)
    
    p1_win_graph, = plt.plot(player1_wins, label = "Player 1 wins")
    p2_win_graph, = plt.plot(player2_wins, label = "Player 2 wins")
    tie_graph, = plt.plot(ties, label = "Ties")
    move_limit_graph, = plt.plot(move_limit, label = "Move limit reached")
    
    plt.ylabel("Occurance per " +str(interval) + " games")
    plt.xlabel("Interval")
    
    plt.legend(handles=[p1_win_graph, p2_win_graph, tie_graph, move_limit_graph])


NUM_TRAINING_ROUNDS = 25
NUM_GAMES_TO_TRAIN = 25
TRAINING_MOVE_LIMIT = 500

ALPHA_BETA_DEPTH = 2

PLAYER0 = DP_Value_Iteration(True)
PLAYER1 = Alpha_beta(False, ALPHA_BETA_DEPTH)
 
training_info = []
validation_info = []

for j in range(NUM_TRAINING_ROUNDS):
    training_info.extend(play_n_games(PLAYER0, PLAYER1, NUM_GAMES_TO_TRAIN, TRAINING_MOVE_LIMIT))
    print("Round " + str(j+1) + " completed!")
    print("")

plt.show()
 
pretty_outcome_display(training_info)
print("")
pretty_outcome_display(validation_info)
