import utils as ut
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.spatial as sp
import imageio


def generate_points_xy0(N, pdf, contrast=0.0):
    H, W = pdf.shape
    pdf = pdf/np.sum(pdf)
    pdf_y = np.sum(pdf, axis=1)
    cdf_y = np.cumsum(pdf_y)
    cdf_y_inv = ut.invert_cdf(cdf_y)

    pdf_y.shape += (1,)
    cdf_xs_inv = np.cumsum(pdf/pdf_y, axis=1)
    for k in range(cdf_xs_inv.shape[1]):
        cdf_xs_inv[:, k] = ut.invert_cdf(cdf_xs_inv[:, k])

    pts_list = np.random.rand(N, 2) * [W, H]
    xr = pts_list[:, 0]
    yr = pts_list[:, 1]
    ys = cdf_y_inv[yr.astype(int)]
    xs = cdf_xs_inv[ys, xr.astype(int)]
    out = np.concatenate((xs, ys))
    return out


def generate_points_xy(N, pdf, contrast=0.0):
    tic = time.time()

    def algo1():
        out = []
        n = 0
        while n < N:

            pts_list = np.random.rand(N, 2) * [W, H]
            zs = np.random.rand(N)
            xs = pts_list[:, 0]
            ys = pts_list[:, 1]

            pdfs = pdf[ys.astype(int), xs.astype(int)]
            selected = zs < pdfs
            pts_list = pts_list[selected]

            out.append(pts_list)
            n += pts_list.shape[0]

        out = np.concatenate(out)[:N]
        return out

    H, W = pdf.shape
    pdf = pdf/pdf.max()
    gamma = 2**contrast
    pdf = pdf**gamma
    # pdf = pdf/np.sum(pdf[:]) * H*W

    out = algo1()

    borders = np.array([[0, 0], [0, H-1], [W-1, 0], [W-1, H-1]])
    out = np.concatenate((out, borders))
    print("generate_points_xy", time.time()-tic)
    return out


def paint_delaunay(img, points_xy, show=False):
    tic = time.time()

    tri = sp.Delaunay(points_xy)
    frame = np.zeros_like(img)
    for s in tri.simplices:
        tri_vertices = points_xy[s].astype(int)
        p_xy = np.mean(tri_vertices, axis=0).astype(int)
        color = np.squeeze(img[p_xy[1], p_xy[0]]).tolist()
        cv.fillPoly(frame, [tri_vertices], color)

    print("delaunay:", time.time()-tic)

    if show:
        cv.imshow("paint_delaunay", frame)
        cv.waitKey(0)
    return frame


def compute_magic(img3):
    frame_mono = []
    Ns = [20, 5000, 200] # hls
    ctr = [0.2, 1, -10]
    for channel in range(3):
        img = img3[:, :, channel]
        pdf = ut.compute_gradient(img)

        pts_xy = generate_points_xy(Ns[channel], pdf, contrast=ctr[channel])
        frame_ch = paint_delaunay(img, pts_xy)
        frame_ch.shape += (1,)
        frame_mono.append(frame_ch)

    frame = np.concatenate((frame_mono), axis=2)
    return frame


def main():
    # img = cv.imread("../images/baboon_512x512.png")
    img_bgr = cv.imread("../images/vermeer_758x640.jpg")
    img_bgr = ut.resize(img_bgr, new_height=512)
    H, W = img_bgr.shape[:2]

    img_g = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    img_hls = ut.cvtBRG_to_HLScube(img_bgr)

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Be sure to use lower case
    # video = cv.VideoWriter('out/video_mp4v.avi', fourcc, 6, (W, H))
    img_list = []


    for k in range(10):
        frame = compute_magic(img_hls)
        img_list.append(frame)
        # video.write(frame)
        cv.imshow("frame", frame)
        cv.waitKey(1)

    img_list = [ut.cvtHLScube_to_BGR(hls) for hls in img_list]
    img_list = [cv.cvtColor(bgr, code=cv.COLOR_BGR2RGB) for bgr in img_list]

    imageio.mimsave("out/vermeer_758x640.gif", img_list, fps=10)
    # N = len(pts_xy)
    # print(N)
    # xs, ys = list(zip(*pts_xy))

    # plt.scatter(xs, -ys, marker=".")
    # plt.show()


if __name__ == '__main__':
    main()