import cv2
import imageio
import guibbon as gbn
import numpy as np
from guibbon.typedef import Image_t
import time


def generate_time_tasting_image(beat: float, bpm: float):
    H = 544
    W = int(H * 4 / 3)
    W = int(round(W-W%16))

    cx, cy = W // 2, H // 2

    res = np.zeros((H, W, 3), dtype=np.uint8)

    scale = 1

    radius = int(scale * H * 0.5)
    theta = (beat % 1) * 360

    theta_s = (((beat + 0.5) % 1) - 0.5) * 360 * 60 / bpm

    ## OUTER CIRCLE
    cv2.circle(res, center=(cx, cy), radius=radius, color=(63, 63, 63), thickness=-1)
    cv2.ellipse(res, center=(cx, cy), axes=(radius, radius), angle=-90, startAngle=0, endAngle=theta_s, color=(115, 115, 255), thickness=-1)
    # ticks
    for k in range(10):
        t = k / 10
        theta_tick = t * 2 * np.pi - np.pi / 2
        px = int(round(radius * np.cos(theta_tick) + cx))
        py = int(round(radius * np.sin(theta_tick) + cy))
        cv2.line(res, pt1=(cx, cy), pt2=(px, py), color=(255, 255, 255), thickness=radius // 50)

    ## INNER CIRCLE
    inner_radius = int(radius * 0.8)
    cv2.circle(res, center=(cx, cy), radius=inner_radius, color=(142, 70, 57), thickness=-1)
    cv2.ellipse(res, center=(cx, cy), axes=(inner_radius, inner_radius), angle=-90, startAngle=0, endAngle=theta, color=(116, 223, 250), thickness=-1)

    thick = 1
    size = scale * 0.6 * H / 25
    thickness = thick * H / 25
    txt = f"{beat:.1f}"
    txt = "1.0" if txt == "9.0" else txt
    cv2.putText(res, txt, (W * 1 // 25, H * 9 // 10), cv2.FONT_HERSHEY_DUPLEX, size, color=(0, 0, 0), thickness=int(thickness * 2))
    cv2.putText(res, txt, (W * 1 // 25, H * 9 // 10), cv2.FONT_HERSHEY_DUPLEX, size, color=(255, 255, 255), thickness=int(thickness))
    return res


class TimeTaster:
    def __init__(self):
        self.winname = "Time Taster"
        win: gbn.Guibbon = gbn.create_window(self.winname)
        rng = [f"{v:.4f}" for v in np.linspace(1, 9, 8 * 30 + 1)[:-1]]
        slider = win.create_slider("time", rng, self.onchange_time_slider, initial_index=10)

        self.time: float = float(slider.get_values()[slider.get_index()])
        self.result: Image_t
        self.is_update_needed: bool = True

    def onchange_time_slider(self, index, value):
        self.time = float(value)
        self.is_update_needed = True

    def update(self):
        self.result = generate_time_tasting_image(self.time, 140)
        self.is_update_needed = False

    def show(self):
        while True:
            self.time = (time.time() / 60 * 140) % 8 + 1
            self.is_update_needed = True
            if self.is_update_needed:
                self.update()
                gbn.imshow(self.winname, self.result, mode="fit")
                self.is_update_needed = False
            gbn.waitKeyEx(1)


def start_app():
    mc = TimeTaster()
    mc.show()


def generate_video(filename: str, fps: float, bpm: float, beat_count: int):
    fpb = fps / (bpm / 60)
    frame_count = int(round(beat_count * fpb))
    ts = np.linspace(0, beat_count, frame_count)

    frame = generate_time_tasting_image(0, bpm)
    h, w = frame.shape[:2]

    # Define the codec and create VideoWriter object

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    video = imageio.get_writer(filename, fps=fps)

    for time in ts:
        # video.write(generate_time_tasting_image(time, bpm))
        beat = time % 8 + 1
        video.append_data(cv2.cvtColor(generate_time_tasting_image(beat, bpm), cv2.COLOR_BGR2RGB))

    # video.release()
    video.close()

def generate_video_musique_saut():
    sr = 48000
    t1 = 1664
    t209 = 5990784
    bpm = 208*60/((t209-t1)/sr)
    beat_counts = 208+8
    generate_video(f"out/musique1_bpm={bpm:.4f}_bc={beat_counts}.mp4", 60, bpm, beat_counts)

    t1 = 6306054
    t200 = 11105573
    bpm = 200*60/((t200-t1)/sr)
    beat_counts = 200+8
    # generate_video(f"out/musique2_bpm={bpm:.4f}_bc={beat_counts}.mp4", 60, bpm, beat_counts)

if __name__ == "__main__":
    generate_video_musique_saut()


