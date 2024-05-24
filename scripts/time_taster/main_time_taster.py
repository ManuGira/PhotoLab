import dataclasses

import guibbon as gbn
import numpy as np
from guibbon.typedef import Image_t


def generate_time_tasting_image(time: float):
    res = np.zeros((200, 200), dtype=np.uint8) + 127 + 10*int(time)
    return res

class TimeTaster:
    def __init__(self):
        self.winname = "Time Taster"
        win: gbn.Guibbon = gbn.create_window(self.winname)
        rng = [f"{v:.1f}" for v in np.linspace(1, 9, 81)[:-1]]
        win.create_slider("time", rng, self.onchange_time_slider)

        self.time: float = 0
        self.result: Image_t
        self.is_update_needed: bool = True

    def onchange_time_slider(self, index, value):
        self.time = float(value)
        self.is_update_needed = True

    def update(self):
        self.result = generate_time_tasting_image(self.time)
        self.is_update_needed = False

    def show(self):
        while True:
            if self.is_update_needed:
                self.update()
                gbn.imshow(self.winname, self.result, mode="fit")
                self.is_update_needed = False
            gbn.waitKeyEx(100)


if __name__ == "__main__":
    mc = TimeTaster()
    mc.show()
