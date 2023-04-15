#!/usr/bin/env python
"""
file  :  board_gui.py
brief :  GUI Class for End of the Track Board Game
author:  Sadanand Modak, 2023
"""
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


class EOTgui:
    def __init__(self, rows=8, cols=7, ini=[1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52], wbc="orange", bbc="brown", mc="blue", tsec=5):
        self.N_ROWS = rows
        self.N_COLS = cols
        self.board_state = np.array(ini)
        self.N_BLOCKS_PER = len(self.board_state)//2 - 1
        self.WHITE_BCOLOR = wbc
        self.BLACK_BCOLOR = bbc
        self.pieces_patches = []
        self.t_transition = tsec  # seconds
        self.MOVE_COLOR = mc
        self.ax = self.make_board()

    def _convert_to_gui_convention_xy(self, x, y):
        return (x, self.N_ROWS - y - 1)

    def decode_single_pos(self, n: int):
        return self._convert_to_gui_convention_xy(n % self.N_COLS, n//self.N_COLS)

    def get_ball(self, pidx, enc_pos):
        if pidx == 0:
            # white
            return matplotlib.patches.Circle(self.decode_single_pos(enc_pos), radius=0.12, color="white", linewidth=1)
        elif pidx == 1:
            # black
            return matplotlib.patches.Circle(self.decode_single_pos(enc_pos), radius=0.12, color="black", linewidth=1)
        else:
            raise ValueError("INVALID pidx in get_ball_patch")

    def get_block(self, pidx, enc_pos):
        l = 0.4
        (x, y) = self.decode_single_pos(enc_pos)
        anchor = (x-l/2, y-l/2)
        if pidx == 0:
            # white
            return matplotlib.patches.Rectangle(anchor, l, l, color=self.WHITE_BCOLOR, linewidth=1)
        elif pidx == 1:
            # black
            return matplotlib.patches.Rectangle(anchor, l, l, color=self.BLACK_BCOLOR, linewidth=1)
        else:
            raise ValueError("INVALID pidx in get_block_patch")

    def make_board(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        res = np.add.outer(range(self.N_ROWS), range(self.N_COLS)) % 2
        plt.imshow(res, cmap="binary_r")
        plt.xticks([])
        plt.yticks([])
        return ax

    def set_board_state(self, state):
        self.board_state = np.array(state)

    def get_board_state(self):
        return self.board_state

    def initialize_pieces(self):
        for i in range(len(self.board_state)):
            if i <= self.N_BLOCKS_PER:
                # white
                if i == self.N_BLOCKS_PER:
                    o = self.get_ball(0, self.board_state[i])
                else:
                    o = self.get_block(0, self.board_state[i])
            else:
                # black
                if i == len(self.board_state)-1:
                    o = self.get_ball(1, self.board_state[i])
                else:
                    o = self.get_block(1, self.board_state[i])
            p = self.ax.add_patch(o)
            self.pieces_patches.append(p)
        plt.pause(self.t_transition)

    def highlight_move(self, i):
        self.pieces_patches[i].set(color=self.MOVE_COLOR)
        plt.pause(self.t_transition)

    def update_state(self):
        for i in range(len(self.board_state)):
            self.pieces_patches[i].remove()
            if i <= self.N_BLOCKS_PER:
                # white
                if i == self.N_BLOCKS_PER:
                    o = self.get_ball(0, self.board_state[i])
                else:
                    o = self.get_block(0, self.board_state[i])
            else:
                # black
                if i == len(self.board_state)-1:
                    o = self.get_ball(1, self.board_state[i])
                else:
                    o = self.get_block(1, self.board_state[i])
            p = self.ax.add_patch(o)
            self.pieces_patches[i] = p
        plt.pause(self.t_transition)


if __name__ == "__main__":
    e = EOTgui()
    e.initialize_pieces()
    e.highlight_move(1)
    e.set_board_state([1, 15, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52])
    e.update_state()

    e.highlight_move(5)
    e.set_board_state([1, 15, 3, 4, 5, 15, 50, 51, 52, 53, 54, 52])
    e.update_state()
    plt.show()
