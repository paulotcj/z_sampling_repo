#problem: https://replit.com/@ZhangMYihua/Matrix-traversal-DFS#index.js
#-------------------------------------------------------------------------
class create_data_structures:
    #-------------------------------------------------------------------------
    def create_matrix():
        matrix = [
            [ 1,  2,  3,  4,  5  ],
            [ 6,  7,  8,  9,  10 ],
            [ 11, 12, 13, 14, 15 ],
            [ 16, 17, 18, 19, 20 ]
        ]
        return matrix
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class MatrixDFS:
    #-------------------------------------------------------------------------
    def __init__(self) -> None:
        self.directions = self.get_directions()
        self.path_explored = []
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_seen_matrix(self, matrix):
        seen = [ [False for col in row] for row in matrix ]
        return seen
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_directions(self):
        directions = [
            [ -1,  0 ],#up
            [  0,  1 ],#right
            [  1,  0 ],#down
            [  0, -1 ] #left
        ]
        return directions
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def traversal_dfs(self, matrix, row=0, col=0):
        self.seen = self.create_seen_matrix(matrix)  # this is dependent on the matrix
        queue = [(row, col)]

        while queue:
            row, col = queue.pop(0)
            if row < 0 or row >= len(matrix): continue
            if col < 0 or col >= len(matrix[0]) : continue 
            if self.seen[row][col] : continue

            self.seen[row][col] = True
            self.path_explored.append(matrix[row][col])
        
            queue = [ [row + x[0], col + x[1]] for x in self.directions ]

        return self.path_explored    
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#expected result: (20) [1, 2, 3, 4, 5, 10, 15, 20, 19, 14, 9, 8, 13, 18, 17, 12, 7, 6, 11, 16]
x = MatrixDFS()
matrix = create_data_structures.create_matrix()
result = x.traversal_dfs(matrix)
expected = [1, 2, 3, 4, 5, 10, 15, 20, 19, 14, 9, 8, 13, 18, 17, 12, 7, 6, 11, 16]
print(f'result: {result}')
print(f'expected: {expected}')
print(f'Is the result what was expected?: {result == expected}')
