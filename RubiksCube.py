import random
import sys
import multiprocessing
import time
from queue import PriorityQueue

# given possible moves for the cube at any given state

MOVES = {
    "U": [2,  0,  3,  1, 20, 21,  6,  7,  4,  5, 10, 11, 12, 13, 14, 15,  8,  9, 18, 19, 16, 17, 22, 23],
    "U'": [1,  3,  0,  2,  8,  9,  6,  7, 16, 17, 10, 11, 12, 13, 14, 15, 20, 21, 18, 19,  4,  5, 22, 23],
    "R": [0,  9,  2, 11,  6,  4,  7,  5,  8, 13, 10, 15, 12, 22, 14, 20, 16, 17, 18, 19,  3, 21,  1, 23],
    "R'": [0, 22,  2, 20,  5,  7,  4,  6,  8,  1, 10,  3, 12, 9, 14, 11, 16, 17, 18, 19, 15, 21, 13, 23],
    "F": [0,  1, 19, 17,  2,  5,  3,  7, 10,  8, 11,  9, 6,  4, 14, 15, 16, 12, 18, 13, 20, 21, 22, 23],
    "F'": [0,  1,  4,  6, 13,  5, 12,  7,  9, 11,  8, 10, 17, 19, 14, 15, 16,  3, 18,  2, 20, 21, 22, 23],
    "D": [0,  1,  2,  3,  4,  5, 10, 11,  8,  9, 18, 19, 14, 12, 15, 13, 16, 17, 22, 23, 20, 21,  6,  7],
    "D'": [0,  1,  2,  3,  4,  5, 22, 23,  8,  9,  6,  7, 13, 15, 12, 14, 16, 17, 10, 11, 20, 21, 18, 19],
    "L": [23,  1, 21,  3,  4,  5,  6,  7,  0,  9,  2, 11, 8, 13, 10, 15, 18, 16, 19, 17, 20, 14, 22, 12],
    "L'": [8,  1, 10,  3,  4,  5,  6,  7, 12,  9, 14, 11, 23, 13, 21, 15, 17, 19, 16, 18, 20,  2, 22,  0],
    "B": [5,  7,  2,  3,  4, 15,  6, 14,  8,  9, 10, 11, 12, 13, 16, 18,  1, 17,  0, 19, 22, 20, 23, 21],
    "B'": [18, 16,  2,  3,  4,  0,  6,  1,  8,  9, 10, 11, 12, 13,  7,  5, 14, 17, 15, 19, 21, 23, 20, 22],
}

OPP_MOVES = {
  "U": "U'",
  "U'": "U",
  "R": "R'",
  "R'": "R",
  "F": "F'",
  "F'": "F",
  "D": "D'",
  "D'": "D",
  "L": "L'",
  "L'": "L",
  "B": "B'",
  "B'": "B",
}

INV_MOVES = {
  "U": "D'",
  "D'": "U",
  "U'": "D",
  "D": "U'",
  "L'": "R",
  "R": "L'",
  "L": "R'",
  "R'": "L",
  "F": "B'",
  "B'": "F",
  "F'": "B",
  "B": "F'",
}

# the following 3 dictionaries are responsible for generating the heuristic
# ACTUAL_POS holds the x,y,z coordinates of the 8 possible corners (sorted) of the cube
# CORNERS holds the current position of the corners in order (we iterate over them sequentially)
# POSITIONS help traverse the cubeList (internal state of the cube) in order to parse the string to index ACTUAL_POS

ACTUAL_POS = {
    "GRY": (0,1,1),
    "GOW": (0,0,0),
    "GRW": (0,0,1),
    "GOY": (0,1,0),
    "BOW": (1,0,0),
    "BRW": (1,0,1),
    "BOY": (1,1,0),
    "BRY": (1,1,1)
}

CORNERS = [(0,0,0), (0,1,0), (0,1,1), (0,0,1), (1,0,1), (1,1,1), (1,0,0), (1,1,0)]
POSITIONS = [[2,8,17], [19,10,12], [6,11,13], [9,4,3], [5,20,1], [7,22,15], [16,0,21], [18,14,23]]


'''
sticker indices:

      0  1
      2  3
16 17  8  9   4  5  20 21
18 19  10 11  6  7  22 23
      12 13
      14 15

face colors:

    0
  4 2 1 5
    3

moves:
[ U , U', R , R', F , F', D , D', L , L', B , B']
'''

allowedColors={'W','Y','R','G','B','O'}
oppositeColors={
  "W":"Y",
  "Y":"W",
  "R":"O",
  "O":"R",
  "B":"G",
  "G":"B",
}

dls_iterations = 0

class cube:

  def __init__(self, string="WWWW RRRR GGGG YYYY OOOO BBBB", path = None, g_value = 0, h_value = 0, f_value = 0):
    # normalize stickers relative to a fixed corner
    stringList = string.split()
    if len(stringList) != 6:
        print("Invalid cube string")
    self.cubeList = [char for word in stringList for char in word]

    self.g_value = g_value
    self.h_value = h_value

    if isinstance(path, list):
      self.path = path.copy()
    elif path is not None:
      self.path = [path]
    else:
      self.path = []
    self.norm()
  
  def __lt__(self, other):
    return self

  def norm(self):
    newMap = {}
    color_map = {
        self.cubeList[10]: 'G',
        self.cubeList[12]: 'Y',
        self.cubeList[19]: 'O',
    }
    for k, v in color_map.items():
        newMap[k] = v
        newMap[oppositeColors[k]] = oppositeColors[v]
    for i, cube in enumerate(self.cubeList):
        self.cubeList[i] = newMap[cube]


  def isSolved(self):
    return all(self.cubeList[i] == self.cubeList[i+1] == self.cubeList[i+2] == self.cubeList[i+3] for i in range(0, 16, 4))
  
  def equals(self, cube):
    self.norm()
    cube.norm()
    return self.cubeList == cube.cubeList
  
  def clone(self):
    clonedCube = cube()
    clonedCube.cubeList = self.cubeList.copy()
    clonedCube.path = self.path.copy()
    return clonedCube
  
  def applyMove(self, move):
    if MOVES[move] is None:
        print("Invalid Move")
        sys.exit(0)
    self.cubeList = [self.cubeList[i] for i in MOVES[move]]
  
  def applyMovesStr(self, movesToApply):
    currentCube = self.clone()
    if movesToApply == "":
      return currentCube
    moves = movesToApply.split(" ")
    for i in moves:
      currentCube.applyMove(i)
    return currentCube
  
  def shuffle(self, n):
    listOfMoves = list(MOVES.keys())
    randomizedMoves = random.sample(listOfMoves, n)
    generatedMoves = " ".join(randomizedMoves)
    newCube = self.applyMovesStr(generatedMoves)
    self.cubeList = newCube.cubeList
    return generatedMoves
    
  # print state of the cube
  def print(self):
    print("\n")
    print("     "+ self.cubeList[0]+ " "+ self.cubeList[1])
    print("     " + self.cubeList[2] + " " + self.cubeList[3])
    print(self.cubeList[16] + " " + self.cubeList[17] + "  " + self.cubeList[8] + " " + self.cubeList[9] + "  " + self.cubeList[4] + " " + self.cubeList[5] + "  " + self.cubeList[20] + " " + self.cubeList[21])
    print(self.cubeList[18] + " " + self.cubeList[19] + "  " + self.cubeList[10] + " " + self.cubeList[11] + "  " + self.cubeList[6] + " " + self.cubeList[7] + "  " + self.cubeList[22] + " " + self.cubeList[23])
    print("     " + self.cubeList[12] + " " + self.cubeList[13])
    print("     " + self.cubeList[14] + " " + self.cubeList[15])
    print("\n")

  def formatPrint(self, moves):
    states = []
    states.append(self.cubeList)
    
    for move in moves:
      self.applyMove(move)
      states.append(self.cubeList)
    
    numCubes = len(states)
    numRows = (numCubes + 2)//3

    for i in range(numRows):
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print("     "+ currentState[0]+ " "+ currentState[1], end="                         ")
      print("")
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print("     "+ currentState[2]+ " "+ currentState[3], end="                         ")
      print("")
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print(currentState[16] + " " + currentState[17] + "  " + currentState[8] + " " + currentState[9] + "  " + currentState[4] + " " + currentState[5] + "  " + currentState[20] + " " + currentState[21], end="               ")
      print("")
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print(currentState[18] + " " + currentState[19] + "  " + currentState[10] + " " + currentState[11] + "  " + currentState[6] + " " + currentState[7] + "  " + currentState[22] + " " + currentState[23], end="               ")
      print("")
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print("     " + currentState[12] + " " + currentState[13], end="                         ")
      print("")
      for j in range(i*3, min(i*3 +3, numCubes)):
        currentState = states[j]
        print("     " + currentState[14] + " " + currentState[15], end="                         ")
      print("")
      print("")



def printCube(string = "WWWW RRRR GGGG YYYY OOOO BBBB"):
    toPrintCube = cube(string)
    toPrintCube.print()

def goal(string):
  goalStateCube = cube(string)
  print(goalStateCube.isSolved())

def norm(string):
  normalizedCube = cube(string)
  normalizedCube.norm()
  normalizedCube.print()

def applyMoves(movesToBeApplied, givenState="WWWW RRRR GGGG YYYY OOOO BBBB"):
  currentCube = cube(givenState)
  newCube = currentCube.applyMovesStr(movesToBeApplied)
  newCube.print()

def shuffleCube(n):
  currentCube = cube()
  listOfmoves = currentCube.shuffle(n)
  stringOfMoves = "".join(listOfmoves)
  print(stringOfMoves)
  currentCube.print()

def randomWalk(moves, N):
  start = time.time()
  
  defaultCube = cube()
  initialCube = defaultCube.applyMovesStr(moves)
  solved = False
  
  if initialCube.isSolved():
    print('Permuted cube in the initial state after applying moves was solved.')
    print(moves)
    initialCube.print()
    print(0) #0 iterations required
    print(time.time() - start)
    solved = True
  
  iterations = 0
  while solved is False:
    iterations = iterations + 1
    permutedCube = initialCube.clone()
    listOfMoves = list(MOVES.keys())
    randomizedMoves = random.sample(listOfMoves, N)
    finalMovesList = []
    for move in randomizedMoves:
      permutedCube.applyMove(move)
      finalMovesList.append(move)
      if permutedCube.isSolved():
        print(' '.join(finalMovesList))
        solved = True
        initialCube.print()
        for move in finalMovesList:
          initialCube.applyMove(move)
          initialCube.print()
        print(iterations)
        print(time.time() - start)
  return



def bfs(moves):
  start = time.time()

  defaultCube = cube()
  node = defaultCube.applyMovesStr(moves)
  initialCube = node.clone()
  open = [node]
  closed = set()
  iterations = 0
  
  while open:
    N = open.pop(0)
    if N.isSolved():
      end = time.time()
      movesToSolve = N.path
      print(' '.join(movesToSolve))
      initialCube.formatPrint(movesToSolve)
      print(iterations)
      print(end - start)
      print()
      return
    else:
      iterations = iterations + 1
    closed.add(tuple(N.cubeList))
    for move in MOVES.keys():
      permutedCube = N.clone()
      permutedCube.applyMove(move)
      if (tuple(permutedCube.cubeList) not in closed):
        permutedCube.path.append(move)
        open.append(permutedCube)

def heuristic(node):
  heuristic_value = 0
  for i in range(len(CORNERS)):
    corner_coord = CORNERS[i]
    corner_1, corner_2, corner_3 = node.cubeList[POSITIONS[i][0]], node.cubeList[POSITIONS[i][1]], node.cubeList[POSITIONS[i][2]]
    corner_sorted = ''.join(sorted(corner_1 + corner_2 + corner_3))
    corner_goal_coord = ACTUAL_POS[corner_sorted]
    manhattan_dist = sum(abs(val1-val2) for val1, val2 in zip(corner_coord, corner_goal_coord))
    heuristic_value += manhattan_dist
  return (heuristic_value/4)

def astar(moves):
  start = time.time()

  defaultCube = cube()
  node = defaultCube.applyMovesStr(moves)
  initialCube = node.clone()
  node.g_value = 0
  node.h_value = heuristic(node)
  f_value = node.g_value + node.h_value

  open = set()
  closed = set()
 
  node_queue = PriorityQueue()
  node_queue.put((f_value, node))
  open.add(tuple(node.cubeList))
  iterations = 0

  while node_queue:
    N = node_queue.get()[1]
    if N.isSolved():
      end = time.time()
      movesToSolve = N.path
      print(' '.join(movesToSolve))
      initialCube.formatPrint(movesToSolve)
      print(iterations)
      print(end - start)
      print()
      return
    else:
      iterations += 1
    closed.add(tuple(N.cubeList))
    for move in MOVES.keys():
      permutedCube = N.clone()
      permutedCube.applyMove(move)
      if ((tuple(permutedCube.cubeList) not in closed) and (tuple(permutedCube.cubeList) not in open)):
        permutedCube.path.append(move)
        permutedCube.g_value = N.g_value + 1
        permutedCube.h_value = heuristic(permutedCube)
        f_value = permutedCube.g_value + permutedCube.h_value
        node_queue.put((f_value, permutedCube))
        open.add(tuple(permutedCube.cubeList))

def modifiedAstar(moves):
  start = time.time()

  defaultCube = cube()
  node = defaultCube.applyMovesStr(moves)
  initialCube = node.clone()
  node.g_value = 0
  node.h_value = heuristic(node)
  f_value = node.g_value + node.h_value

  open = set()
  closed = set()
 
  node_queue = PriorityQueue()
  node_queue.put((f_value, node))
  open.add(tuple(node.cubeList))
  iterations = 0

  while node_queue:
    N = node_queue.get()[1]
    if N.isSolved():
      end = time.time()
      movesToSolve = N.path
      print(' '.join(movesToSolve))
      initialCube.formatPrint(movesToSolve)
      print(iterations)
      print(end - start)
      print()
      return
    else:
      iterations += 1
    closed.add(tuple(N.cubeList))
    possibleMoves = MOVES.keys()
    try:
      if OPP_MOVES[N.path[-1]] is not None:
        possibleMoves.remove(OPP_MOVES[N.path[-1]])
      if INV_MOVES[N.path[-1]] is not None:
        possibleMoves.remove(INV_MOVES[N.path[-1]])
    except:
      pass
    for move in possibleMoves:
      permutedCube = N.clone()
      permutedCube.applyMove(move)
      if ((tuple(permutedCube.cubeList) not in closed) and (tuple(permutedCube.cubeList) not in open)):
        permutedCube.path.append(move)
        permutedCube.g_value = N.g_value + 1
        permutedCube.h_value = heuristic(permutedCube)
        f_value = permutedCube.g_value + permutedCube.h_value
        node_queue.put((f_value, permutedCube))
        open.add(tuple(permutedCube.cubeList))


def recursive_dls(node, limit):
  global dls_iterations
  dls_iterations += 1
  if node.isSolved():
    return node
  elif limit == 0:
    return 0
  else:
    cutoffOccurred = False
    for move in MOVES.keys():
      permutedCube = node.clone()
      permutedCube.applyMove(move)
      permutedCube.path.append(move)
      result = recursive_dls(permutedCube, limit - 1)
      if result == 0:
        cutoffOccurred = True
      elif result != 1:
        return result
    if cutoffOccurred:
      return 0
    else:
      return 1

def dls(moves, limit):
  start = time.time()
  global dls_iterations

  defaultCube = cube()
  node = defaultCube.applyMovesStr(moves)
  initialCube = node.clone()

  result = recursive_dls(node, limit)

  if isinstance(result, cube) and result.isSolved():
    end = time.time()
    movesToSolve = result.path
    print(' '.join(movesToSolve))
    initialCube.formatPrint(movesToSolve)
    print(dls_iterations)
    print(end - start)
    print()
  else:
    print("Unfortunately, no solution could be found through depth limited search")
  dls_iterations = 0
  return

def ids_helper(moves, limit):
  
  defaultCube = cube()
  node = defaultCube.applyMovesStr(moves)
  result = recursive_dls(node, limit)
  return result

def ids(moves, limit):
  defaultCube = cube()
  initialCube = defaultCube.applyMovesStr(moves)
  start = time.time()
  total_iterations = 0
  isSolved = False

  for i in range(limit):
    global dls_iterations
    result = ids_helper(moves, i)
    total_iterations += dls_iterations
    if isinstance(result, cube) and result.isSolved():
      end = time.time()
      print(f"Depth: {i} d: {dls_iterations}")
      print(f"IDS solution found at depth {i}")
      movesToSolve = result.path
      print(' '.join(movesToSolve))
      initialCube.formatPrint(movesToSolve)
      print(f"Total iterations: {total_iterations}")
      print(end - start)
      print()
      isSolved = True
      break
    else:
      print(f"Depth: {i} d: {dls_iterations}")
    dls_iterations = 0
  if isSolved is False:
    print("Unfortunately, no solution could be found atthe given depth through iterative deepening search.")

def main():
  nargs=len(sys.argv)
  if sys.argv[1] == "print":
    if nargs==3:
      printCube(sys.argv[2])
    else:
      printCube()
  elif sys.argv[1] == "goal":
    if nargs==3:
      goal(sys.argv[2])
    else:
      print("goal command missing argument")
  elif sys.argv[1] == "norm":
    if nargs==3:
      norm(sys.argv[2])
    else:
      print("norm command missing argument")
  elif sys.argv[1] == "applyMovesStr":
    if nargs == 4:
      applyMoves(sys.argv[2],sys.argv[3])
    elif nargs == 3:
      applyMoves(sys.argv[2])
    else:
      print("applyMoves command missing an argument")
  elif sys.argv[1] == "shuffle":
    if nargs == 3:
      shuffleCube(int(sys.argv[2]))
    else:
      print("shuffle command missing an argument")
  elif sys.argv[1] == "random":
    if nargs == 5:
      p = multiprocessing.Process(target=randomWalk, name="random", args=(sys.argv[2], int(sys.argv[3]),))
      p.start()
      p.join(int(sys.argv[4]))
      if p.is_alive():
        print("No solution in the time limit!")
        p.terminate()
        p.join()
    else:
      print("random command missing an argument")
  elif sys.argv[1] == "bfs":
    if nargs==3:
      bfs(sys.argv[2])
  elif sys.argv[1] == "dls":
    if nargs==4:
      dls(sys.argv[2], int(sys.argv[3]))
  elif sys.argv[1] == "ids":
    if nargs==4:
      ids(sys.argv[2], int(sys.argv[3]))
  elif sys.argv[1] == "astar":
    if nargs==3:
      astar(sys.argv[2])
  elif sys.argv[1] == "competition":
    if nargs==3:
      modifiedAstar(sys.argv[2])
  else:
    print("Command parameters may be invalid")


if __name__=="__main__":
  main()
