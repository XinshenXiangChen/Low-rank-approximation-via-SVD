from PIL import Image
import numpy as np

SAMPLES = 100


def extract_RGB(matrix):
    R = matrix[:, :, 0]
    G = matrix[:, :, 1]
    B = matrix[:, :, 2]
    return R, G, B


def _descending(eigenvalues, eigenvectors):
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def left(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix @ matrix.T)
    return _descending(eigenvalues, eigenvectors)


def right(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix.T @ matrix)
    return _descending(eigenvalues, eigenvectors)


def low_rank(channel, k):
    """Rank-k approximation of one channel from its left and right singular vectors."""
    channel = channel.astype(np.float64)
    left_eigenvalues, left_vectors = left(channel)
    right_eigenvalues, right_vectors = right(channel)

    rank = min(k, *channel.shape)
    # The smaller Gram matrix is the better-conditioned source of singular values.
    if channel.shape[0] >= channel.shape[1]:
        eigenvalues = right_eigenvalues
    else:
        eigenvalues = left_eigenvalues
    singular_values = np.sqrt(np.clip(eigenvalues[:rank], 0, None))

    U = left_vectors[:, :rank].copy()
    V = right_vectors[:, :rank].copy()
    # eigh fixes each vector only up to sign. Flip so A v = σ u.
    signs = np.sign(np.sum(U * (channel @ V), axis=0))
    signs[signs == 0] = 1
    U *= signs

    return (U * singular_values) @ V.T


def SVD(matrix, k=SAMPLES):
    R, G, B = extract_RGB(matrix)
    approximated = np.stack([low_rank(channel, k) for channel in (R, G, B)], axis=2)
    return np.clip(np.rint(approximated), 0, 255).astype(np.uint8)


def main():
    img = Image.open('img.png').convert('RGB')
    matrix = np.array(img)
    approximated = SVD(matrix)

    Image.fromarray(approximated, mode='RGB').save('img_approx.png')
    original = matrix.astype(np.float64)
    error = np.linalg.norm(original - approximated) / np.linalg.norm(original)
    print(f'Saved img_approx.png with rank {SAMPLES}. Relative error: {error:.4f}')


if __name__ == '__main__':
    main()
