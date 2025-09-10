#problem: https://leetcode.com/problems/number-of-islands
from typing import List, Dict

#-------------------------------------------------------------------------
class Solution:
    #-------------------------------------------------------------------------
    def numIslands(self, grid: List[List[str]]) -> int:
        

        island_cnt : int = 0
        #----------------
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                #----
                if grid[row][col] == '1':
                    grid[row][col] = '*'
                    island_cnt += 1
                    self.bfs(grid, row, col)
                    # print(f'grid: {grid}')
                #----
        #----------------
        
        return island_cnt
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def bfs(self,grid : List[List[int]], row:int,col:int) -> None:
        row_len : int = len(grid)
        col_len : int = len(grid[0])

        queue : List[List[int]] = [[row,col]]

        # print(f'start of BFS - queue: {queue}')

        new_row : int = 0
        new_col : int = 0

        #----------------
        while queue:
            r, c = queue.pop(0)
            # print(f'popped: (r,c) {r,c}')

            #note: you only need to check the variable that you are changing, so if you do row - 1, check if
            # (row - 1) >= 0, if you do row + 1, check if (row + 1) < row_len

            #searching pattern is: up, down, left, right
            new_row , new_col = r - 1 , c
            if new_row >= 0 and grid[new_row][new_col] == '1': #up (-1,0)
                grid[new_row][new_col] = '*'
                queue.append([new_row,new_col])

            new_row , new_col = r  , c + 1
            if new_col < col_len and grid[new_row][new_col] == '1': #right (0,+1)
                grid[new_row][new_col] = '*'
                queue.append([new_row,new_col])

            new_row , new_col = r + 1  , c 
            if new_row < row_len and grid[new_row][new_col] == '1': #down (+1,0)
                grid[new_row][new_col] = '*'
                queue.append([new_row,new_col])

            new_row , new_col = r  , c - 1
            if new_col >= 0 and grid[new_row][new_col] == '1': #left (0,-1)
                grid[new_row][new_col] = '*'
                queue.append([new_row,new_col])

            # print(f'queue: {queue}')
        #---------------- end of while
        # print(f'end of BFS')
    #-------------------------------------------------------------------------    
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class aux:
    #-------------------------------------------------------------------------
    def create_grid_1():
        grid = [
            ["1","1","1","1","0"],
            ["1","1","0","1","0"],
            ["1","1","0","0","0"],
            ["0","0","0","0","0"]
            ]
        return grid, 1
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_grid_2():
        grid = [
            ["1","1","0","0","0"],
            ["1","1","0","0","0"],
            ["0","0","1","0","0"],
            ["0","0","0","1","1"]
        ]
        return grid, 3
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def create_grid_3():
        grid = [
                ["1","1","1"],
                ["0","1","0"],
                ["1","1","1"]
            ]
        return grid, 1
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def test():

        x = Solution()
        grid, expected = aux.create_grid_1()
        result = x.numIslands(grid)
        print(f'expected: {expected} - result: {result} - pass: {expected == result}')
        print('-------')

        x = Solution()
        grid, expected = aux.create_grid_2()
        result = x.numIslands(grid)
        print(f'expected: {expected} - result: {result} - pass: {expected == result}')
        print('-------')

        x = Solution()
        grid, expected = aux.create_grid_3()
        result = x.numIslands(grid)
        print(f'expected: {expected} - result: {result} - pass: {expected == result}')
        print('-------')
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
aux.test()
