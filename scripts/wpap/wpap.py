import photolab.utils
from photolab import utils_old as ut
from photolab import color_spaces as cs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.spatial as sp
import imageio
import os

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


def paint_delaunay(img, points_xy, height=None, show=False):
    tic = time.time()
    H, W = img.shape[:2]
    if height is None:
        height = H
    width = int(round(W*height/H))

    points_xy_scaled = points_xy * [width/W, height/H]

    tri = sp.Delaunay(points_xy_scaled)
    frame = np.zeros(shape=(height, width), dtype=img.dtype)
    for s in tri.simplices:
        tri_vertices_scaled = points_xy_scaled[s].astype(int)
        tri_vertices = points_xy[s].astype(int)
        p_xy = np.mean(tri_vertices, axis=0).astype(int)
        color = np.squeeze(img[p_xy[1], p_xy[0]]).tolist()
        cv2.fillPoly(frame, [tri_vertices_scaled], color)

    print("delaunay:", time.time()-tic)

    if show:
        frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        cv2.imshow("paint_delaunay", frame_u8)
        cv2.waitKeyEx(0)
    return frame


def compute_magic(img3, height=None):
    frame_mono = []
    Ns = [20, 5000, 200] # hls
    ctr = [0.2, 1, -10]
    for channel in range(3):
        img = img3[:, :, channel]
        pdf = ut.compute_gradient(img)

        pts_xy = generate_points_xy(Ns[channel], pdf, contrast=ctr[channel])
        frame_ch = paint_delaunay(img, pts_xy, height)
        frame_ch.shape += (1,)
        frame_mono.append(frame_ch)

    frame = np.concatenate((frame_mono), axis=2)
    return frame

def main_superflux():
    import superflux as sf

    def cvtBRG_to_HLScube(img_bgr):
        res = cvtBRG_to_HLScube(img_bgr)
        return {"res": res}


    H_dst = 3840
    folder = "../images"
    filename = "vermeer_758x640.jpg"
    # filename = "baboon_512x512.png"
    filepath = os.path.join(folder, filename)
    img_bgr = cv2.imread(filepath)
    img_bgr = photolab.utils.resize(img_bgr, new_height=512)
    H, W = img_bgr.shape[:2]
    img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hls = ut.cvtBRG_to_HLScube(img_bgr)

    GRAYSCALE = sf.Node(cvtBRG_to_HLScube, "res")
    BGR2HLS = sf.Node(cvtBRG_to_HLScube, "res")


    V2 = sf.Value(v2)
    V5 = sf.Value(v5)
    Mul = sf.Node(multiply, ["res"])
    Sub = sf.Node(subtract, ["res"])

    V2("res") >> Mul["lhs"]
    V5("res") >> Mul["rhs"]

    Mul("res") >> Sub["lhs"]
    V2("res") >> Sub["rhs"]

def main():
    H_dst = 3840
    folder = "../../images"
    filename = "vermeer_758x640.jpg"
    # filename = "baboon_512x512.png"
    filepath = os.path.join(folder, filename)
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        raise FileNotFoundError(filepath)
    img_bgr = photolab.utils.resize(img_bgr, new_height=512)
    H, W = img_bgr.shape[:2]
    img_g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hls = cs.BGR_to_HLS_u8(img_bgr).astype(float) / 255

    # Define the codec and create VideoWriter object
    # fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    # fourcc = cv.VideoWriter_fourcc(*'MJPG')  # Be sure to use lower case
    # video = cv.VideoWriter('out/video_mp4v.avi', fourcc, 6, (W, H))
    img_list = []
    for k in range(16):
        frame = compute_magic(img_hls, H_dst)
        frame = cs.HLS_to_BGR_u8((frame * 255).astype(np.uint8))
        frame = cv2.GaussianBlur(frame, (5, 5), 2, 2)
        img_list.append(frame)
        # video.write(frame)
        # cv.imshow("frame", frame)
        # cv.waitKey(0)

    dst_folder = "out"
    try:
        os.mkdir(dst_folder)
    except:
        pass

    print(f"Save all images in {os.path.abspath(dst_folder)}")
    for i in range(len(img_list)):
        dst_filename = ut.insert_text_before_file_extension(os.path.join(dst_folder, filename), f"_{i}")
        dst_filename = '.'.join(dst_filename.split('.')[:-1]) + ".png"
        cv2.imwrite(dst_filename, img_list[i])

    img_list = [cv2.cvtColor(bgr, code=cv2.COLOR_BGR2RGB) for bgr in img_list]
    dst_filename_gif = filename[:filename.rfind(".")]
    dst_filename_gif = f"{dst_folder}/{dst_filename_gif}.gif"
    print(f"Create a gif with all images: {os.path.abspath(dst_filename_gif)}")
    imageio.mimsave(dst_filename_gif, img_list, fps=10)



if __name__ == '__main__':
    main()