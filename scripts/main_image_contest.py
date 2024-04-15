import cv2
import photolab.utils as ut

KEY_1 = 49
KEY_2 = 50
WINNAME = "IMAGE CONTEST"


def run_round(img_list, contenders):
    """
    only support contenders with length being a power of 2 (2, 4, 8, 16, etc...)
    """

    print("Contenders:", contenders)
    winners = []
    for k in range(len(contenders)//2):
        i = contenders[2*k]
        j = contenders[2*k+1]

        frame = cv2.hconcat((img_list[i], img_list[j]))
        cv2.imshow(WINNAME, frame)
        while True:
            key = cv2.waitKey(0)
            if key == ord("1"):
                winners.append(i)
                print("\t", i, ">", j)
                break
            elif key == ord("2"):
                winners.append(j)
                print("\t", i, "<", j)
                break
    return winners


def main():
    filename_list = [f"wpap/out/vermeer_758x640_{i}.png" for i in range(16)]
    img_list = [cv2.imread(filename) for filename in filename_list]
    img_list = [ut.resize(img, 720) for img in img_list]

    cv2.namedWindow(WINNAME)
    contenders = list(range(len(img_list)))

    while len(contenders) > 1:
        winners = run_round(img_list, contenders)
        contenders = winners

    print("WINNER IS", filename_list[contenders[0]])



if __name__ == '__main__':
    main()