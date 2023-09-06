from typing import Callable, List

import numpy as np
import gc
import scipy.signal
import torch
import tifffile as tif
from torch.nn import functional as func
from .utils import to_RGB_Mask, to_RGB_sample, remap_severity_labels
from typing import Any, Optional, Dict, List, Tuple

WINDOW_CACHE = dict()


def _spline_window(window_size: int, power: int = 2) -> np.ndarray:

    """Generates a 1-dimensional spline of order 'power' (typically 2), in the designated
    window.
    https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    Args:
        window_size (int): size of the interested window
        power (int, optional): Order of the spline. Defaults to 2.
    Returns:
        np.ndarray: 1D spline
    """
    intersection = int(window_size / 4)
    wind_outer = (abs(2 * (scipy.signal.triang(window_size)))**power) / 2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2 * (scipy.signal.triang(window_size) - 1))**power) / 2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


def _spline_2d(window_size: int, power: int = 2) -> torch.Tensor:

    """Makes a 1D window spline function, then combines it to return a 2D window function.
    The 2D window is useful to smoothly interpolate between patches.
    Args:
        window_size (int): size of the window (patch)
        power (int, optional): Which order for the spline. Defaults to 2.
    Returns:
        np.ndarray: numpy array containing a 2D spline function
    """
    # Memorization to avoid remaking it for every call
    # since the same window is needed multiple times
    global WINDOW_CACHE
    key = f"{window_size}_{power}"
    if key in WINDOW_CACHE:
        wind = WINDOW_CACHE[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)  # SREENI: Changed from 3, 3, to 1, 1
        wind = torch.from_numpy(wind * wind.transpose(1, 0, 2))
        WINDOW_CACHE[key] = wind
    return wind


def compute_pad(image: torch.Tensor, tile_size: int, subdivisions: int):

    """Compute padding

    Args:
        image (np.ndarray): input image
        tile_size (int): tile size
        subdivisions (int): subdivisions

    Returns:
        type_: description_
    """
    # compute the pad as (window - window/subdivisions)
    pad = int(round(tile_size * (1 - 1.0 / subdivisions)))

    image = image.detach().numpy()
    width, height, _ = image.shape

    width_start_pad = pad
    height_start_pad = pad
    width_end_pad = pad
    height_end_pad = pad

    if width % pad != 0:
        width_end_pad = 2 * pad
    if height % pad != 0:
        height_end_pad = 2 * pad
    return (width_start_pad, width_end_pad, height_start_pad, height_end_pad)


def pad_image(image: np.ndarray, tile_size: int, subdivisions: int) -> np.ndarray:

    """Add borders to the given image for a "valid" border pattern according to "window_size" and "subdivisions".
    Image is expected as a numpy array with shape (width, height, channels).

    Args:
        image (torch.Tensor): input image, 3D channels-last tensor
        tile_size (int): size of a single patch, useful to compute padding
        subdivisions (int): amount of overlap, useful for padding

    Returns:
        torch.Tensor: same image, padded specularly by a certain amount in every direction
    """
    # compute the pad as (window - window/subdivisions)
    # pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    width_start_pad, width_end_pad, height_start_pad, height_end_pad = compute_pad(image, tile_size, subdivisions)
    # add pad pixels in height and width, nothing channel-wise
    image = np.pad(image, ((width_start_pad, width_end_pad), (height_start_pad, height_end_pad), (0, 0)),
                   mode="reflect")
    # return (width_start_pad, width_end_pad, height_start_pad, height_end_pad), image
    return torch.from_numpy(image)


def unpad_image(padded_image: torch.Tensor, tile_size: int, subdivisions: int) -> torch.Tensor:

    """Reverts changes made by 'pad_image'. The same padding is removed, so tile_size and subdivisions
    must be coherent.
    Args:
        padded_image (torch.Tensor): image with padding still applied
        tile_size (int): size of a single patch
        subdivisions (int): subdivisions to compute overlap
    Returns:
        torch.Tensor: image without padding, 2D channels-last tensor
    """
    # compute the same amount as before, window - window/subdivisions
    # pad = int(round(tile_size * (1 - 1.0 / subdivisions)))
    # # crop the image left, right, top and bottom
    # result = padded_image[pad:-pad, pad:-pad]
    # return result
    padded_image = padded_image.detach().numpy()
    aug = int(round(tile_size * (1 - 1.0/subdivisions)))
    ret = padded_image[
        aug:-aug,
        aug:-aug
    ]
    return torch.from_numpy(ret)


def rotate_and_mirror(image: torch.Tensor) -> List[torch.Tensor]:

    """Duplicates an image with shape (h, w, channels) 8 times, in order
    to have all the possible rotations and mirrors of that image that fits the
    possible 90 degrees rotations. https://en.wikipedia.org/wiki/Dihedral_group
    Args:
        image (torch.Tensor): input image, already padded.
    Returns:
        List[torch.Tensor]: list of images, rotated and mirrored.
    """
    variants = []
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    image = torch.flip(image, dims=(0, 1))
    variants.append(image)
    variants.append(torch.rot90(image, k=1, dims=(0, 1)))
    variants.append(torch.rot90(image, k=2, dims=(0, 1)))
    variants.append(torch.rot90(image, k=3, dims=(0, 1)))
    return variants


def undo_rotate_and_mirror(variants: List[torch.Tensor]) -> torch.Tensor:

    """Reverts the 8 duplications provided by rotate and mirror.
    This restores the transformed inputs to the original position, then averages them.
    Args:
        variants (List[torch.Tensor]): D4 dihedral group of the same image
    Returns:
        torch.Tensor: averaged result over the given input.
    """
    origs = []
    origs.append(variants[0])
    origs.append(torch.rot90(variants[1], k=3, dims=(0, 1)))
    origs.append(torch.rot90(variants[2], k=2, dims=(0, 1)))
    origs.append(torch.rot90(variants[3], k=1, dims=(0, 1)))

    origs.append(torch.flip(variants[4], dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[5], k=3, dims=(0, 1)), dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[6], k=2, dims=(0, 1)), dims=(0, 1)))
    origs.append(torch.flip(torch.rot90(variants[7], k=1, dims=(0, 1)), dims=(0, 1)))
    return torch.mean(torch.stack(origs), axis=0)


def windowed_generator(padded_image: torch.Tensor, window_size: int, subdivisions: int, batch_size: int = None):

    """Generator that yield tiles grouped by batch size.
    Args:
        padded_image (np.ndarray): input image to be processed (already padded), supposed channels-first
        window_size (int): size of a single patch
        subdivisions (int): subdivision count on each patch to compute the step
        batch_size (int, optional): amount of patches in each batch. Defaults to None.
    Yields:
        Tuple[List[tuple], np.ndarray]: list of coordinates and respective patches as single batch array
    """
    step = window_size // subdivisions
    width, height, _ = padded_image.shape
    batch_size = batch_size or 1

    batch = []
    coords = []

    # step with fixed window on the image to build up the arrays
    for x in range(0, width - window_size + 1, step):
        for y in range(0, height - window_size + 1, step):
            coords.append((x, y))
            # extract the tile, place channels first for batch
            tile = padded_image[x:x + window_size, y:y + window_size]
            batch.append(tile.permute(2, 0, 1))
            # yield the batch once full and restore lists right after
            if len(batch) == batch_size:
                yield coords, torch.stack(batch)
                coords.clear()
                batch.clear()
    # handle last (possibly unfinished) batch
    if len(batch) > 0:
        yield coords, torch.stack(batch)


def reconstruct(canvas: torch.Tensor, tile_size: int, coords: List[tuple], predictions: torch.Tensor) -> torch.Tensor:

    """Helper function that iterates the result batch onto the given canvas to reconstruct
    the final result batch after batch.
    Args:
        canvas (torch.Tensor): container for the final image.
        tile_size (int): size of a single patch.
        coords (List[tuple]): list of pixel coordinates corresponding to the batch items
        predictions (torch.Tensor): array containing patch predictions, shape (batch, tile_size, tile_size, num_classes)
    Returns:
        torch.Tensor: the updated canvas, shape (padded_w, padded_h, num_classes)
    """    
    for (x, y), patch in zip(coords, predictions):
        canvas[x:x + tile_size, y:y + tile_size] += patch
    return canvas


def preprocess_onnx_data(data: Optional[Dict]) -> List[Any]:

        """
        Preprocessing step, where the generic input is transformed into tensors or other inputs.
        The result is in list form, where each element correspond to the respective input.
        :param data: generic dictionary containing input data
        :type data: Optional[Dict]
        :param files: optional list of files, already available locally
        :type files: Optional[List]
        :param params: additional key-value parameters, if needed
        :type params: dict
        :return: list of preprocessed inputs
        :rtype: list
        """
        input_dims = (12,480,480)
        mean = 0.5
        std = 0.5

        batch = []
        for image in data:
            assert image.shape == input_dims, f"Wrong input dims: {image.shape}"
            # append to list as channels-first image
            batch.append(image.numpy())#.transpose(2, 0, 1))
        # construct batch
        batch = np.stack(batch, axis=0)
        batch = (batch - mean) / std
        return [batch]


def predict_smooth_windowing(mode: str,
                            image: torch.Tensor,
                            tile_size: int,
                            subdivisions: int,
                            prediction_fn: Callable,
                            batch_size: int = None,
                            channels_first: bool = False,
                            mirrored: bool = False) -> np.ndarray:
    
    """Allows to predict a large image in one go, dividing it in squared, fixed-size tiles and smoothly
    interpolating over them to produce a single, coherent output with the same dimensions.
    Args:
        image (np.ndarray): input image, expected a 3D vector
        tile_size (int): size of each squared tile
        subdivisions (int): number of subdivisions over the single tile for overlaps
        prediction_fn (Callable): callback that takes the input batch and returns an output tensor
        batch_size (int, optional): size of each batch. Defaults to None.
        channels_first (int, optional): whether the input image is channels-first or not
        mirrored (bool, optional): whether to use dihedral predictions (every simmetry). Defaults to False.
    Returns:
        np.ndarray: numpy array with dimensions (w, h), containing smooth predictions
    """

    gc.collect()

    if channels_first:
        image = image.permute(1, 2, 0)

    width, height, _ = image.shape
    padded = pad_image(image=image, tile_size=tile_size, subdivisions=subdivisions)
    padded_width, padded_height, _ = padded.shape
    padded_variants = rotate_and_mirror(padded) if mirrored else [padded]
    spline = _spline_2d(window_size=tile_size, power=2).to(image.device).squeeze(-1)

    results_del = []
    # results_del_0 = []
    # results_del_1 = []
    results_gra = []

    # RGB part for debug purposes
    # results_sample_R = []
    # results_sample_G = []
    # results_sample_B = []

    for img in padded_variants:
        canvas_del = torch.zeros((padded_width, padded_height), device=image.device)
        # canvas_del_0 = torch.zeros((padded_width, padded_height), device=image.device)
        # canvas_del_1 = torch.zeros((padded_width, padded_height), device=image.device) 
        canvas_gra = torch.zeros((padded_width, padded_height), device=image.device)

        # RGB part for debug purposes
        # canvas_R = torch.zeros((padded_width, padded_height), device=image.device)
        # canvas_G = torch.zeros((padded_width, padded_height), device=image.device)
        # canvas_B = torch.zeros((padded_width, padded_height), device=image.device)
        # index = 0
        
        for coords, batch in windowed_generator(padded_image=img,
                                                window_size=tile_size,
                                                subdivisions=subdivisions,
                                                batch_size=batch_size):
            
            if mode == "multitask_onnx":
                batch = preprocess_onnx_data(batch)

            if mode == "DEL" or mode == "GRA":
                # returns batch of channels-first, return to channels-last
                result = prediction_fn(batch)  # .permute(0, 2, 3, 1)
                if isinstance(result, tuple) or isinstance(result, list):
                    pred_batch, *_ = result
                else:
                    pred_batch = result

                pred_batch = pred_batch.squeeze()
                pred_batch = [tile * spline for tile in pred_batch]
                canvas_del = reconstruct(canvas_del, tile_size=tile_size, coords=coords, predictions=pred_batch)
                
            elif mode == "multitask":
                # returns batch of channels-first, return to channels-last
                pred_batch_del, pred_batch_gra = prediction_fn(batch)  # .permute(0, 2, 3, 1)
                pred_batch_del = pred_batch_del.squeeze()
                pred_batch_gra = pred_batch_gra.squeeze()

                pred_batch_del = [tile * spline for tile in pred_batch_del]
                pred_batch_gra = [tile * spline for tile in pred_batch_gra]

                canvas_del = reconstruct(canvas_del, tile_size=tile_size, coords=coords, predictions=pred_batch_del)
                canvas_gra = reconstruct(canvas_gra, tile_size=tile_size, coords=coords, predictions=pred_batch_gra)

                # RGB part for debug purposes
                # batch_sample_R  = [tile * 1 for tile in batch[:,3,:,:]]
                # batch_sample_G  = [tile * 1 for tile in batch[:,2,:,:]]
                # batch_sample_B  = [tile * 1 for tile in batch[:,1,:,:]]              

                # canvas_R = reconstruct(canvas_R, tile_size=tile_size, coords=coords, predictions=batch_sample_R)
                # canvas_G = reconstruct(canvas_G, tile_size=tile_size, coords=coords, predictions=batch_sample_G)
                # canvas_B = reconstruct(canvas_B, tile_size=tile_size, coords=coords, predictions=batch_sample_B)
                # canvas_sample = torch.stack([canvas_R, canvas_G, canvas_B], dim=2)

                # path = "/home/merlo/multitask-segmentation/Semantic Segmentation/assets/smoothing/Canva" + str(index) + ".png" 
                # to_RGB_sample(canvas_sample, path=path, RGB_image=True)
                # index += 1
                
            elif mode == "multitask_onnx":
                 # returns batch of channels-first, return to channels-last
                pred_batch_del, pred_batch_gra = prediction_fn(batch)  # .permute(0, 2, 3, 1)
                
                pred_batch_del = pred_batch_del.squeeze()
                pred_batch_del = (np.argmax(pred_batch_del, axis=1)).astype(np.uint8) # postprocess DELINEATION
                # pred_batch_del_0 = pred_batch_del[:,0,:,:].squeeze()
                # pred_batch_del_1 = pred_batch_del[:,1,:,:].squeeze()
                pred_batch_gra = pred_batch_gra.squeeze()
                
                pred_batch_del = torch.from_numpy(pred_batch_del)
                # pred_batch_del_0 = torch.from_numpy(pred_batch_del_0)
                # pred_batch_del_1 = torch.from_numpy(pred_batch_del_1)
                pred_batch_gra = torch.from_numpy(pred_batch_gra)

                pred_batch_del = [tile * spline for tile in pred_batch_del] # spline.numpy()
                # pred_batch_del_0 = [tile * spline for tile in pred_batch_del_0] # spline.numpy()
                # pred_batch_del_1 = [tile * spline for tile in pred_batch_del_1] # spline.numpy()
                pred_batch_gra = [tile * spline for tile in pred_batch_gra] # spline.numpy()         
                
                canvas_del = reconstruct(canvas_del, tile_size=tile_size, coords=coords, predictions=pred_batch_del)
                # canvas_del_0 = reconstruct(canvas_del_0, tile_size=tile_size, coords=coords, predictions=pred_batch_del_0)
                # canvas_del_1 = reconstruct(canvas_del_1, tile_size=tile_size, coords=coords, predictions=pred_batch_del_1)
                canvas_gra = reconstruct(canvas_gra, tile_size=tile_size, coords=coords, predictions=pred_batch_gra)
            
        if mode == "DEL" or mode == "GRA":
            canvas_del /= (subdivisions**2)
            results_del.append(canvas_del)

        elif mode == "multitask":
            canvas_del /= (subdivisions**2)
            results_del.append(canvas_del)
            canvas_gra /= (subdivisions**2)
            results_gra.append(canvas_gra)
            
            # RGB part for debug purposes
            # canvas_R /= (subdivisions**2)
            # results_sample_R.append(canvas_R)
            # canvas_G /= (subdivisions**2)
            # results_sample_G.append(canvas_G)
            # canvas_B /= (subdivisions**2)
            # results_sample_B.append(canvas_B)

        elif mode == "multitask_onnx":
            canvas_del /= (subdivisions**2)
            results_del.append(canvas_del)
            # canvas_del_0 /= (subdivisions**2)
            # results_del_0.append(canvas_del_0)
            # canvas_del_1 /= (subdivisions**2)
            # results_del_1.append(canvas_del_1)
            canvas_gra /= (subdivisions**2)
            results_gra.append(canvas_gra)

    if mode == "DEL" or mode == "GRA":
        padded_result = undo_rotate_and_mirror(results_del) if mirrored else results_del[0]
        prediction = unpad_image(padded_result, tile_size=tile_size, subdivisions=subdivisions)

    elif mode == "multitask":
        padded_result_del = undo_rotate_and_mirror(results_del) if mirrored else results_del[0]
        prediction_del = unpad_image(padded_result_del, tile_size=tile_size, subdivisions=subdivisions)

        padded_result_gra = undo_rotate_and_mirror(results_gra) if mirrored else results_gra[0]
        prediction_gra = unpad_image(padded_result_gra, tile_size=tile_size, subdivisions=subdivisions)

        # RGB part for debug purposes
        # padded_result_sample_R = undo_rotate_and_mirror(results_sample_R) if mirrored else results_sample_R[0]
        # unpadded_sample_R = unpad_image(padded_result_sample_R, tile_size=tile_size, subdivisions=subdivisions)
        # padded_result_sample_G = undo_rotate_and_mirror(results_sample_G) if mirrored else results_sample_G[0]
        # unpadded_sample_G = unpad_image(padded_result_sample_G, tile_size=tile_size, subdivisions=subdivisions)
        # padded_result_sample_B = undo_rotate_and_mirror(results_sample_B) if mirrored else results_sample_B[0]
        # unpadded_sample_B = unpad_image(padded_result_sample_B, tile_size=tile_size, subdivisions=subdivisions)

        # sample_result = torch.stack([unpadded_sample_R, unpadded_sample_G, unpadded_sample_B], dim=2)
        # path = "/home/merlo/multitask-segmentation/Semantic Segmentation/assets/smoothing/FinalResult.png" 
        # to_RGB_sample(sample_result[:width, :height], path=path, RGB_image=True)

    elif mode == "multitask_onnx":
        padded_result_del = undo_rotate_and_mirror(results_del) if mirrored else results_del[0]
        prediction_del = unpad_image(padded_result_del, tile_size=tile_size, subdivisions=subdivisions)
        # padded_result_del_0 = undo_rotate_and_mirror(results_del_0) if mirrored else results_del_0[0]
        # prediction_del_0 = unpad_image(padded_result_del_0, tile_size=tile_size, subdivisions=subdivisions)
        # padded_result_del_1 = undo_rotate_and_mirror(results_del_1) if mirrored else results_del_1[0]
        # prediction_del_1 = unpad_image(padded_result_del_1, tile_size=tile_size, subdivisions=subdivisions)
        # prediction_del = torch.stack([prediction_del_0,prediction_del_1], dim=0)
        # prediction_del = torch.from_numpy((np.argmax(prediction_del.numpy(), axis=0)).astype(np.uint8)) # postprocess DELINEATION

        padded_result_gra = undo_rotate_and_mirror(results_gra) if mirrored else results_gra[0]
        prediction_gra = unpad_image(padded_result_gra, tile_size=tile_size, subdivisions=subdivisions)

        prediction_gra = (np.clip(prediction_gra.numpy(), 0, 4) / 1.0) # 4.0 # postprocess GRADING
        prediction_gra = torch.from_numpy(prediction_gra.astype(np.uint8))
        
    gc.collect()
    
    if mode == "DEL" or mode == "GRA":
        return prediction[:width, :height]
    elif mode == "multitask" or mode == "multitask_onnx":
        return prediction_del[:width, :height], prediction_gra[:width, :height]