import enum
import tkinter as tk
from typing import Callable, Any, Sequence, List, Tuple, Optional
from guibbon.interactive_overlays import Point
from guibbon.widgets.widget import WidgetInterface
import numpy as np
import numpy.typing as npt

CallbackMatrixEditor = Callable[[npt.NDArray], None]


class Cell:
    def __init__(self, canvas, x1, y1, size):
        self.canvas = canvas
        hsize = size // 2
        self.tile_id = self.canvas.create_rectangle(
            x1 - hsize,
            y1 - hsize,
            x1 + hsize,
            y1 + hsize)
        self.text_id = self.canvas.create_text(x1, y1, text="0", fill="black", font=('Helvetica', hsize))

    def set_value(self, value):
        self.canvas.itemconfig(self.text_id, text=str(value))

    def set_color(self, u8):
        print(u8)
        u8 = min(max(0, int(round(u8))), 255)
        self.canvas.itemconfig(self.tile_id, fill="#%02x%02x%02x" % (u8, u8, u8))
        self.canvas.itemconfig(self.text_id, fill="white" if u8 < 127 else "black")

    def __del__(self):
        # delete canvas shapes when destroying the cell
        self.canvas.delete(self.tile_id)
        self.canvas.delete(self.text_id)


class MatrixEditorWidget(WidgetInterface):
    colors = {
        "grey": "#%02x%02x%02x" % (191, 191, 191),
    }

    def __init__(self, tk_frame: tk.Frame, matrix_editor_name: str, height: int, width: int, initial_matrix: npt.NDArray, on_change: Optional[CallbackMatrixEditor] = None,
                 widget_color=None):
        self.name = tk.StringVar(value=matrix_editor_name)
        self.height = height
        self.width = width
        assert initial_matrix.shape == (height, width)
        self.on_change: Optional[CallbackMatrixEditor] = on_change

        self.matrix = initial_matrix.copy().astype(int)

        tk.Label(tk_frame, textvariable=self.name, bg=widget_color).pack(padx=2, side=tk.LEFT)

        self.cell_size = 20

        self.canvas = tk.Canvas(master=tk_frame, height=1, width=1, borderwidth=0, bg=widget_color)
        self.canvas.pack(side=tk.TOP)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", self.on_click)

        self.cells_matrix = []
        self.reset_canvas()

        self.update_canvas()

    def reset_canvas(self):
        self.canvas.config(width=self.width * self.cell_size, height=self.height * self.cell_size)

        self.cells_matrix = []
        shift = self.cell_size // 2 + 1
        for i in range(self.height):
            row = []
            y1 = i * self.cell_size + shift
            for j in range(self.width):
                x1 = j * self.cell_size + shift
                cell = Cell(self.canvas, x1, y1, self.cell_size)
                row.append(cell)
            self.cells_matrix.append(row)

    def update_canvas(self):
        max_val = max(1, np.array(self.matrix).max())

        print("max", max_val)
        for i in range(self.height):
            for j in range(self.width):
                cell = self.cells_matrix[i][j]
                value = self.matrix[i, j]
                cell.set_value(value)
                cell.set_color((value * 255) // max_val)
                self.canvas.pack(side=tk.TOP)

    def on_click(self, event):
        match event.num:
            case 1:
                delta = 1
            case 3:
                delta = -1
            case _:
                return

        N = self.height * self.width
        x, y = event.x, event.y
        i = min(y // self.cell_size, self.height-1)
        j = min(x // self.cell_size, self.width-1)
        self.matrix[i, j] = min(max(0, self.matrix[i, j] + delta), N)
        self.update_canvas()

        if self.on_change is not None:
            self.on_change(self.matrix)

    def set_matrix(self, matrix):
        if matrix.shape != (self.height, self.width):
            assert len(matrix.shape) == 2
            self.height, self.width = matrix.shape
            self.reset_canvas()

        self.matrix = matrix.copy().astype(int)
        self.update_canvas()
