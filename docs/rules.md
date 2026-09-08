# Intransitive rules

The rules the engine implements. The site is the authority; anything here that
disagrees with the site is a bug in this document.

## Board and pieces

- 9x9 board. Squares are named `a1` to `i9` (file letter, rank digit).
- Two players, Blue and Red. Blue moves first.
- Each side has ten pieces: 3 rock, 4 paper, 3 scissors.
- Starting position, Blue's pieces: rock b4 c3 d2, paper b5 c4 d3 e2, scissors
  c5 d4 e3. Red's pieces are Blue's reflected across the anti-diagonal
  (a1 maps to i9).
- Blue's home is a1 and Blue's goal is i9. Red's home is i9 and Red's goal is a1.

## Moves

- A piece moves one square in any of the eight king directions.
- The target must be empty or hold an enemy piece that the moving piece
  beats. Rock beats scissors, scissors beats paper, paper beats rock.
- Moving onto an enemy piece captures it. A piece never moves onto a friendly
  piece or onto an enemy piece it does not beat.

## Game end

The game ends immediately, in this order of precedence:

1. A player moves a piece onto their goal square: that player wins.
2. A player has no pieces left: the other player wins.
3. The player to move has no legal move: the player to move loses.
4. 200 plies have passed since the last capture: draw.

There is no repetition rule.

## Canonical frame

The engine stores every position from the mover's point of view: the mover's
home is a1 and the mover's goal is i9, the mover's pieces are "own" and the
opponent's are "enemy". After a move the board is reflected across the
anti-diagonal and the sides are swapped, so the next mover sees the same frame.
Actions are `direction * 81 + from_square` with the eight directions in the
order (-1,-1) (-1,0) (-1,1) (0,-1) (0,1) (1,-1) (1,0) (1,1) as (rank, file)
deltas. A position is `(board, plies since capture, ply)`.
