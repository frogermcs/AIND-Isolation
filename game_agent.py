"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")

    # return number_of_moves_and_blank_spaces(game, player)
    # return number_of_moves_and_location(game, player)
    return number_of_moves_improved(game, player)


# 80-85%
def number_of_moves_and_blank_spaces(game, player):
    """This is modification of our number_of_moves_improved function. Besides moves count difference, there is
    additional parameter: number of empty fields around current player position minus number of empty fields around
    opponent. Empty fields are checked in 5x5 square around player (reachable area for L-shaped move). Assumption was that
    the more empty fields, the better (comparing to opponent).

    Major factor in this evaluation function was still the difference between player moves and opponent moves.
    The difference between empty fields around player and opponent is just an adjustment to original
    evaluation: number_of_moves_improved + (num_of_blanks / 100).

    Final result is better for about 5% than number_of_moves_improved function.

    Factors for own_blanks and opp_blanks were found in ranges [0, 10]. (2,1) is the best performing pair.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    opponent = game.get_opponent(player)

    own_moves = len(game.get_legal_moves(player)) * 5
    opp_moves = len(game.get_legal_moves(opponent)) * 2

    game_blank_spaces = game.get_blank_spaces()
    own_blanks = len(get_blank_spaces_around(game, game_blank_spaces, game.get_player_location(player))) * 2
    opp_blanks = len(get_blank_spaces_around(game, game_blank_spaces, game.get_player_location(opponent))) * 1

    return float(own_moves - opp_moves) + float(own_blanks - opp_blanks) / 100


def get_blank_spaces_around(game, game_blank_spaces, position):
    """This is helper function to find blank spaces around current position. The biggest possible square around box
     is 5x5. L shape moves (1,2 or 2,1 in each direction) generates bounds for square in this size.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    game_blank_spaces : hashable
        Array of empty spaces in entire board.

    position : (x, y)
        Position on board which is the center to find blank spaces around

    Returns
    ----------
    Array
        The array of blank spaces around given position
    """
    return [(x, y)
            for x in range(max(0, position[0] - 2), min(game.width, position[0] + 2) + 1)
            for y in range(max(0, position[1] - 2), min(game.width, position[1] + 2) + 1)
            if (x, y) in game_blank_spaces]


# (5,2,1)
# ~75-80%
def number_of_moves_and_location(game, player):
    """This is modification of our number_of_moves_improved function. Besides moves count difference, there is
    additional parameter: distance between current player and current opponent position. Assumption that it's better
    to keep opponent closer to ourselves improved our original evaluation efficiency by ~5%.

    Factors: 5 (own moves), 2 (opponent moves) and 1 (1/distance) were find as the best in ranges [1, 10] each.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    own_moves = len(game.get_legal_moves(player)) * 5
    opp_moves = len(game.get_legal_moves(game.get_opponent(player))) * 2
    own_loc = game.get_player_location(player)
    opp_loc = game.get_player_location(game.get_opponent(player))
    distance = math.sqrt((own_loc[0] - opp_loc[0]) ** 2 + (own_loc[1] - opp_loc[1]) ** 2)

    return float(own_moves - opp_moves + 1 / distance)


# 70-75%
def number_of_moves_improved(game, player):
    """This is modified function of The "Improved" evaluation discussed in lecture. Originally both: player and opponent
    moves are equally important in valuation. Here We used different const factors: 5 for own moves and 2 for opp moves.
    Factors were found in very simple "Machine Learning" way: in a loops we checked all values (i, j)
    in range [(1, 1)] ... [(10, 10)], where "i" is own_moves factor and "j" is opp_moves factor.

    The best performing and most stable values were: (9,1) and (5,2) - the last pair is used in this function.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """

    own_moves = len(game.get_legal_moves(player)) * 5
    opp_moves = len(game.get_legal_moves(game.get_opponent(player))) * 2

    return float(own_moves - opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='alphabeta', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        if not legal_moves:
            return (-1, -1)

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        def keep_spinning():
            return time_left() > 0 and (self.search_depth == -1 or depth <= self.search_depth)

        move = (-1, -1)
        try:
            depth = 1
            if self.iterative:
                while keep_spinning():
                    move = self.find_best_move(depth, game, legal_moves)
                    depth += 1
            else:
                move = self.find_best_move(depth, game, legal_moves)

        except Timeout:
            pass

        finally:
            return move

    def find_best_move(self, depth, game, legal_moves):
        if self.method == "minimax":
            _, move = self.minimax(game, depth)
        elif self.method == "alphabeta":
            _, move = self.alphabeta(game, depth)
        else:
            _, move = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        def min_value(game, d):
            if d == depth:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return -1

            v = float("inf")
            for move in legal_moves:
                v = min(v, max_value(game.forecast_move(move), d + 1))
            return v

        def max_value(game, d):
            if d == depth:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return -1

            v = float("-inf")
            for move in legal_moves:
                v = max(v, min_value(game.forecast_move(move), d + 1))
            return v

        best_move = float("-inf"), (-1, -1)
        for move in legal_moves:
            best_move = max(best_move, (min_value(game.forecast_move(move), 1), move))

        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        def min_value(game, d, alpha, beta):
            if d == depth:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return -1

            v = float("inf")
            for move in legal_moves:
                v = min(v, max_value(game.forecast_move(move), d + 1, alpha, beta))
                if v <= alpha:
                    return v

                beta = min(v, beta)
            return v

        def max_value(game, d, alpha, beta):
            if d == depth:
                return self.score(game, self)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                return -1

            v = float("-inf")
            for move in legal_moves:
                v = max(v, min_value(game.forecast_move(move), d + 1, alpha, beta))
                if v >= beta:
                    return v
                alpha = max(v, alpha)
            return v

        best_move = float("-inf"), (-1, -1)
        for move in legal_moves:
            best_move = max(best_move, (min_value(game.forecast_move(move), 1, alpha, beta), move))
            alpha = best_move[0]

        return best_move
