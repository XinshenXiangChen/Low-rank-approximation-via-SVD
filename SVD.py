from PIL import Image
import numpy as np

SAMPLES = 10

def extract_RGB(matrix):
    R = matrix[:, :, 0]
    G = matrix[:, :, 1]
    B = matrix[:, :, 2]
    return R, G, B


def left(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix @ matrix.T)
    return eigenvalues, eigenvectors

def right(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
    return eigenvalues, eigenvectors

def SVD(matrix):
    R, G, B = extract_RGB(matrix)
    le_R, lev_R = left(R)
    le_G, lev_G = left(G)
    le_B, lev_B = left(B)

    re_R, lev_R = right(R)
    re_G, lev_G = right(G)
    re_B, lev_B = right(B)



def main():

    img = Image.open('img.png')
    matrix = np.array(img)




if __name__ == '__main__':

    main()