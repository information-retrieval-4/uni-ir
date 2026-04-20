"""
WEMIR - Weighted Edge Matching Information Retrieval
====================================================
Implementation of the CBIR method from:
    Tamilkodi & Nesakumari (2021)
    "A novel framework for retrieval of image using weighted edge matching algorithm"
    Multimedia Tools and Applications

Pipeline:
    1. Preprocessing: median filter -> K-means (k=3) -> histogram equalization -> DWT (LL)
    2. Feature extraction: SVD reduction -> 5×5 block Hungarian assignment
    3. Retrieval: Euclidean / Manhattan distance
"""

import numpy as np
import cv2
import pywt
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import pickle
import time


# Prepro stuff


def median_filter(image, ksize=3):
    """Remove noise using median filter (paper Section 2)."""
    return cv2.medianBlur(image, ksize)


def kmeans_cluster(image, k=3, max_iter=100):
    """
    K-means clustering on RGB pixel values (paper Section 2.1).
    Groups pixel values into k clusters using Euclidean distance.
    Returns flattened labels and cluster centers.
    """
    pixels = image.reshape(-1, 3).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        max_iter,
        0.2
    )
    # Deterministic init: assign initial labels based on pixel luminance
    luminance = (0.299 * pixels[:, 2] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 0])
    init_labels = np.digitize(
        luminance,
        bins=np.linspace(luminance.min(), luminance.max() + 1e-6, k + 1)[1:-1]
    ).astype(np.int32).reshape(-1, 1)

    _, labels, centers = cv2.kmeans(
        pixels, k, init_labels, criteria, 1, cv2.KMEANS_USE_INITIAL_LABELS
    )
    return labels.flatten(), centers


def select_largest_cluster(image, labels, k=3):
    """
    Select the cluster with the most pixels (paper: "spot any one group").
    Returns a masked image containing only the selected cluster's pixels
    and the boolean mask.
    """
    counts = np.bincount(labels, minlength=k)
    largest = np.argmax(counts)
    mask = (labels == largest).reshape(image.shape[:2])

    result = np.zeros_like(image)
    result[mask] = image[mask]
    return result, mask


def histogram_equalization(image):
    """
    Histogram equalization for contrast enhancement (paper Section 2.2).

    The paper describes the "confined mean computation" which maps pixel
    intensities via the CDF to achieve uniform distribution:
        E[i,j] = floor(N * sum(H[m], m=0..I[i,j]))

    We use OpenCV's equalizeHist on the luminance channel.
    """
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)


def dwt_ll(image):
    """
    Apply 1-level Discrete Wavelet Transform and return the LL subband
    (paper Section 2.3).

    Uses Haar wavelet. The LL subband is the low-frequency approximation
    at half the original resolution.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    coeffs = pywt.dwt2(gray.astype(np.float64), "haar")
    ll = coeffs[0]
    return ll


# Feature Extraction


def svd_reduce(matrix, rank=None, energy_ratio=0.9):
    """
    SVD for dimensionality reduction (paper Step 2).

    Factorizes I = U S V^T, then reconstructs with only the top-r
    singular values to reduce the representation.

    Args:
        matrix: 2D array (the LL subband)
        rank: number of singular values to keep. If None, auto-select
              based on energy_ratio.
        energy_ratio: fraction of total energy to preserve (default 0.9)

    Returns:
        Reduced matrix (rank-r approximation)
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)

    if rank is None:
        total_energy = np.sum(S**2)
        cumulative = np.cumsum(S**2)
        rank = np.searchsorted(cumulative, energy_ratio * total_energy) + 1
        rank = max(rank, 5)  # at least 5 to form valid 5×5 blocks

    rank = min(rank, len(S))
    reduced = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    return reduced


def hungarian_block(block):
    """
    Apply the Hungarian algorithm to a 5×5 block (paper Steps 3-10).

    The paper describes:
        Step 3: Ensure balanced (square) matrix
        Step 4: Row subtraction - subtract row minima
        Step 5: Column subtraction - subtract col minima
        Step 6: Draw minimum lines to cover all zeros
        Step 7: If lines == size → done; else subtract min uncovered
                from uncovered, add to intersections
        Step 8: Repeat until lines == size
        Step 9: Select assignments (single zeros)
        Step 10: Store the assigned intensity values

    We use scipy's linear_sum_assignment which implements the full
    Hungarian algorithm correctly and efficiently.

    Args:
        block: 5×5 numpy array of pixel intensities

    Returns:
        Array of 5 assigned intensity values (the minimum edge values)
    """
    # Ensure non-negative (the algorithm works on cost matrices)
    cost = block.copy()
    if cost.min() < 0:
        cost = cost - cost.min()

    row_ind, col_ind = linear_sum_assignment(cost)
    return block[row_ind, col_ind]


def extract_block_features(matrix, block_size=5):
    """
    Split matrix into non-overlapping blocks and extract features via
    Hungarian assignment (paper Steps 3-10).

    Each 5×5 block yields 5 feature values (the minimum cost assignment).
    The matrix is zero-padded if not evenly divisible by block_size.

    Args:
        matrix: 2D array (SVD-reduced LL subband)
        block_size: size of each square block (default 5)

    Returns:
        1D feature vector
    """
    h, w = matrix.shape

    # Pad to make dimensions divisible by block_size (paper Step 3)
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        matrix = np.pad(
            matrix, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
        )

    h, w = matrix.shape
    features = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = matrix[i : i + block_size, j : j + block_size]
            assigned = hungarian_block(block)
            features.extend(assigned)

    return np.array(features, dtype=np.float64)


# Full Pipeline


def preprocess(image):
    """
    Full preprocessing pipeline (paper Section 2).

    1. Median filter for noise removal
    2. K-means clustering (k=3) on RGB pixels
    3. Select largest cluster
    4. Histogram equalization (confined mean)
    5. 1-level DWT → LL subband

    Args:
        image: BGR image (numpy array)

    Returns:
        LL subband as 2D float array
    """
    # 1. Median filter
    filtered = median_filter(image)

    # 2. K-means clustering (k=3)
    labels, centers = kmeans_cluster(filtered, k=3)

    # 3. Select largest cluster
    cluster_img, mask = select_largest_cluster(filtered, labels)

    # 4. Histogram equalization
    enhanced = histogram_equalization(cluster_img)

    # 5. DWT - extract LL subband
    ll = dwt_ll(enhanced)

    return ll


def extract_features(image, svd_rank=None):
    """
    Full WEMIR feature extraction pipeline.

    Preprocessing → SVD reduction → 5×5 block Hungarian assignment

    Args:
        image: BGR image (numpy array)
        svd_rank: number of singular values to keep (None = auto)

    Returns:
        1D feature vector
    """
    ll = preprocess(image)
    reduced = svd_reduce(ll, rank=svd_rank)
    features = extract_block_features(reduced)
    return features


# Distance / Similarity


def compute_distance(feat_a, feat_b, metric="euclidean"):
    """
    Compute distance between two feature vectors (paper Section 3, Step 12).

    Handles different-length vectors by zero-padding the shorter one.

    Args:
        feat_a, feat_b: 1D feature vectors
        metric: 'euclidean' or 'manhattan'

    Returns:
        Scalar distance value
    """
    max_len = max(len(feat_a), len(feat_b))
    a = np.zeros(max_len)
    b = np.zeros(max_len)
    a[: len(feat_a)] = feat_a
    b[: len(feat_b)] = feat_b

    if metric == "euclidean":
        return np.sqrt(np.sum((a - b) ** 2))
    elif metric == "manhattan":
        return np.sum(np.abs(a - b))
    else:
        raise ValueError(f"Unknown metric: {metric}")


# Index
class WEMIRIndex:
    """
    WEMIR feature index for content-based image retrieval.

    Builds a database of feature vectors from a directory of images,
    and supports querying with a new image to find the most similar ones.
    """

    def __init__(self, svd_rank=None):
        self.svd_rank = svd_rank
        self.features = {}  # path (str) -> feature vector (np.ndarray)
        self.labels = {}  # path (str) -> category label (str)

    def build(self, image_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
        """
        Build the feature index from all images in a directory.

        Expects images organized in category subfolders:
            image_dir/
                category1/
                    img1.jpg
                    img2.jpg
                category2/
                    ...

        The parent folder name is used as the category label.

        Args:
            image_dir: path to the root image directory
            extensions: file extensions to include
        """
        image_dir = Path(image_dir)
        image_paths = sorted(
            [p for p in image_dir.rglob("*") if p.suffix.lower() in extensions]
        )

        total = len(image_paths)
        print(f"Building WEMIR index for {total} images...")
        start = time.time()

        for idx, img_path in enumerate(image_paths, 1):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  [{idx}/{total}] skipped (unreadable): {img_path.name}")
                continue

            try:
                feat = extract_features(image, self.svd_rank)
                self.features[str(img_path)] = feat
                self.labels[str(img_path)] = img_path.parent.name

                if idx % 50 == 0 or idx == total:
                    elapsed = time.time() - start
                    print(f"  [{idx}/{total}] {elapsed:.1f}s elapsed")

            except Exception as e:
                print(f"  [{idx}/{total}] failed: {img_path.name} — {e}")

        elapsed = time.time() - start
        print(f"Done! Indexed {len(self.features)} images in {elapsed:.1f}s")

    def query(self, query_image_or_path, top_k=10, metric="euclidean"):
        """
        Retrieve the top-k most similar images to a query.

        Args:
            query_image_or_path: path string or BGR numpy array
            top_k: number of results to return
            metric: 'euclidean' or 'manhattan'

        Returns:
            List of (path, distance, label) tuples, sorted ascending by distance
        """
        if isinstance(query_image_or_path, (str, Path)):
            image = cv2.imread(str(query_image_or_path))
            if image is None:
                raise ValueError(f"Cannot read image: {query_image_or_path}")
        else:
            image = query_image_or_path

        query_feat = extract_features(image, self.svd_rank)

        distances = []
        for path, feat in self.features.items():
            dist = compute_distance(query_feat, feat, metric)
            label = self.labels.get(path, "unknown")
            distances.append((path, dist, label))

        distances.sort(key=lambda x: x[1])
        return distances[:top_k]

    def evaluate(self, query_image_path, top_k=10, metric="euclidean"):
        """
        Query and compute precision/recall.

        The query image's category (parent folder name) is used as ground truth.
        Precision = relevant retrieved / total retrieved
        Recall = relevant retrieved / total relevant in database

        Args:
            query_image_path: path to the query image
            top_k: number of results
            metric: distance metric

        Returns:
            dict with 'results', 'precision', 'recall', 'query_label'
        """
        query_path = Path(query_image_path)
        query_label = query_path.parent.name
        results = self.query(query_image_path, top_k, metric)

        # Count relevant images in the full database
        total_relevant = sum(1 for lbl in self.labels.values() if lbl == query_label)

        # Count relevant in retrieved results
        relevant_retrieved = sum(1 for _, _, lbl in results if lbl == query_label)

        precision = relevant_retrieved / len(results) if results else 0.0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

        return {
            "results": results,
            "precision": precision,
            "recall": recall,
            "query_label": query_label,
            "total_relevant": total_relevant,
            "relevant_retrieved": relevant_retrieved,
        }

    def save(self, path):
        """Save the index to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "features": self.features,
                    "labels": self.labels,
                    "svd_rank": self.svd_rank,
                },
                f,
            )
        print(f"Index saved to {path}")

    @classmethod
    def load(cls, path):
        """Load an index from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls(svd_rank=data.get("svd_rank"))
        idx.features = data["features"]
        idx.labels = data["labels"]
        print(f"Index loaded: {len(idx.features)} images")
        return idx
