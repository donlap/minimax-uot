"""Shape completion on Completion3D with the plan-based estimators."""

import glob
import os
import urllib.request
import zipfile

import h5py
import numpy as np
from sklearn.decomposition import PCA

from uot import discrete

URL = "http://download.cs.stanford.edu/downloads/completion3d/shapenet16K2019.zip"
CHAIR = "03001627"
CAR = "02958343"


def download(directory, synsets=(CHAIR, CAR), url=URL):
    """Downloads Completion3D and extracts the given classes.

    Args:
        directory: Directory of the dataset.
        synsets: ShapeNet identifiers of the classes to extract.
        url: Address of the archive.

    Returns:
        The root directory of the extracted dataset.
    """
    root = os.path.join(directory, "shapenet16384")
    if os.path.isdir(root):
        return root
    os.makedirs(directory, exist_ok=True)
    archive = os.path.join(directory, os.path.basename(url))
    urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(directory, [m for m in z.namelist() if set(m.split("/")) & set(synsets)])
    os.remove(archive)
    return root


def voxelize(partial, complete, grid_size=16):
    """Voxelizes a partial point cloud and its complete version.

    Both clouds are rotated by 90 degrees about the x axis and rescaled by
    the bounding box of the complete cloud.

    Args:
        partial: Partial point cloud of shape (p, 3).
        complete: Complete point cloud of shape (q, 3).
        grid_size: Number of voxels per axis.

    Returns:
        A tuple of the flattened occupancy grids of the two clouds.
    """
    c, s = np.cos(np.radians(90.0)), np.sin(np.radians(90.0))
    rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    partial, complete = partial @ rotation.T, complete @ rotation.T
    low = np.min(complete, axis=0)
    scale = np.max(complete, axis=0) - low + 1e-7
    grids = []
    for cloud in (partial, complete):
        index = np.floor(np.clip((cloud - low) / scale, 0.0, 0.9999) * grid_size).astype(int)
        grid = np.zeros((grid_size,) * 3, dtype=np.float32)
        grid[index[:, 0], index[:, 1], index[:, 2]] = 1.0
        grids.append(grid.flatten())
    return grids[0], grids[1]


def load(root, split, synset, grid_size=16, limit=None):
    """Loads and voxelizes the shapes of one class.

    Args:
        root: Root directory of the dataset.
        split: Name of the split.
        synset: ShapeNet identifier of the class.
        grid_size: Number of voxels per axis.
        limit: Optional maximum number of shapes.

    Returns:
        A tuple of the partial and complete occupancy grids, each of shape
        (k, grid_size**3).

    Raises:
        FileNotFoundError: If no shape is found.
    """
    partials, completes = [], []
    for path in sorted(glob.glob(os.path.join(root, split, "partial", synset, "*.h5"))):
        target = os.path.join(root, split, "gt", synset, os.path.basename(path))
        if not os.path.exists(target):
            continue
        with h5py.File(path, "r") as fp, h5py.File(target, "r") as fg:
            p, g = voxelize(np.array(fp["data"][:]), np.array(fg["data"][:]), grid_size)
        partials.append(p)
        completes.append(g)
        if limit is not None and len(partials) >= limit:
            break
    if not partials:
        raise FileNotFoundError(f"no shapes of class {synset} in {os.path.join(root, split)}")
    return np.array(partials), np.array(completes)


def dataset(root, grid_size=16, outliers=30, tests=4, dim=1024, seed=0):
    """Builds the latent samples of the shape completion experiment.

    The source sample holds the partial training chairs and the first
    outliers partial training cars. The target sample holds the complete
    training chairs. A PCA fitted on the complete training chairs and cars
    maps the occupancy grids to dim dimensions.

    Args:
        root: Root directory of the dataset.
        grid_size: Number of voxels per axis.
        outliers: Number of cars in the source sample.
        tests: Number of test chairs and of test cars.
        dim: Latent dimension.
        seed: Seed of the randomized PCA.

    Returns:
        A tuple of the source sample, the target sample, the latent test
        shapes, the fitted PCA, and the occupancy grids of the test shapes.
        The test shapes are the last validation chairs followed by the first
        validation cars.
    """
    chairs, chairs_gt = load(root, "train", CHAIR, grid_size)
    test_chairs = load(root, "val", CHAIR, grid_size)[0][-tests:]
    cars, cars_gt = load(root, "train", CAR, grid_size, outliers)
    test_cars = load(root, "val", CAR, grid_size, tests)[0]
    pca = PCA(n_components=dim, random_state=seed).fit(np.vstack((chairs_gt, cars_gt)))
    x = np.vstack((pca.transform(chairs), pca.transform(cars)))
    y = pca.transform(chairs_gt)
    test = np.vstack((pca.transform(test_chairs), pca.transform(test_cars)))
    return x, y, test, pca, np.vstack((test_chairs, test_cars))


def complete(x, y, test, pca, rho=0.5, bandwidth=1.93):
    """Completes the test shapes with PB-1NN and PB-Kernel.

    The discrete problem is solved in single precision with unit masses and
    the cost divided by its median.

    Args:
        x: Source sample of shape (n, dim).
        y: Target sample of shape (m, dim).
        test: Latent test shapes of shape (k, dim).
        pca: The fitted PCA.
        rho: Weight of each marginal KL penalty.
        bandwidth: Bandwidth of PB-Kernel.

    Returns:
        A dict mapping each estimator name to the completed occupancy values
        in [0, 1], of shape (k, grid_size**3), and the growth values, of
        shape (k,).
    """
    a = (np.ones(len(x)) / len(x)).astype(np.float32)
    b = (np.ones(len(y)) / len(y)).astype(np.float32)
    t, lam = discrete.plan_maps(x.astype(np.float32), y.astype(np.float32), a, b, rho=rho, normalize=True)
    fits = {
        "PB-1NN": discrete.extend_1nn(x, test, t, lam),
        "PB-Kernel": discrete.extend_kernel(x, test, t, lam, bandwidth=bandwidth),
    }
    return {name: (np.clip(pca.inverse_transform(m), 0.0, 1.0), g) for name, (m, g) in fits.items()}
