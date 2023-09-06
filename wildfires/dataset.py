import glob
import os
import re
import functools
import sys
import json
import datetime
from sys import getsizeof
import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt
from typing import Any, Dict, List, cast, Optional, Sequence, Callable, Tuple
from pathlib import Path

import torch
from torch import Tensor

from torchgeo.datasets import RasterDataset, GeoDataset
from torchgeo.datasets.utils import BoundingBox, concat_samples, disambiguate_timestamp, merge_samples

import rasterio
from rasterio.crs import CRS
import rasterio.merge
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from wildfires.utils import to_RGB_Mask, to_RGB_sample


class DatasetCEMS_Simplyfied(RasterDataset):
    filename_glob = "EMSR*AOI*S2L2A.tiff"
    file_types = {
        "sentinel": "S2L2A",
        "del_mask": "DEL",
        "gra_mask": "GRA",
        "lc_esa_mask": "ESA_LC",
        "lc_annual_mask": "Annual9_LC",
        "cloud_mask": "CM"
    }
    
    rgb_bands = ["B04", "B03", "B02"]
    all_bands = ["B02", "B03", "B04", "B08"]

    def __init__(self, root: str = "data", 
                       crs: Optional[CRS] = None, 
                       res: Optional[float] = None, 
                       bands: Optional[Sequence[str]] = None, 
                       transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
                       cache: bool = True, 
                       annotation_type: List[str] = [], 
                       ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            annotation_type: list of string to specify which files to include in the batch. Available choices are: ["del_mask", "gra_mask", "lc_esa_mask", "lc_annual_mask", "cloud_mask"]
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        super().__init__(root, crs, res, bands, transforms, cache)
        self.annotation_type = annotation_type


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True) # https://rtree.readthedocs.io/en/latest/class.html#rtree.index.Index.intersection
                                                                   # create here: https://github.com/microsoft/torchgeo/blob/7e7443a00fa7c6c34bc71c11ad662d797c3fe380/torchgeo/datasets/geo.py#L370
        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes) # rasterio.merge.merge()
        
        sample = {"crs": self.crs, "bbox": query}
        
        sample["image"] = data.float()

        for annotation_cat in self.annotation_type:
             
            if annotation_cat in self.file_types.keys():
                ann_filepaths = [Path(f.replace(self.file_types["sentinel"], self.file_types[annotation_cat])).with_suffix(".tif") for f in filepaths]

                label = self._merge_files(ann_filepaths, query, self.band_indexes)
                sample[annotation_cat] = label.long()

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
    
class DatasetCEMS_Delineation(GeoDataset):
    filename_glob = "EMSR*AOI*S2L2A.tiff"
    # file_types = {
    #     "sentinel": "S2L2A",
    #     "del_mask": "DEL",
    #     "fep_mask": "FEP",
    #     "gra_mask": "GRA",
    #     "lc_esa_mask": "ESA_LC",
    #     "lc_annual_mask": "Annual9_LC",
    #     "cloud_mask": "CM"
    # }

    file_types = {
        "sentinel": "S2L2A",
        "del_mask": "DEL",
        "cloud_mask": "CM"
    }
    
    rgb_bands = ["B04", "B03", "B02"]
    all_bands = ["B02", "B03", "B04", "B08"]
    filename_regex = ".*"
    date_format = "%Y%m%d"
    separate_files = False
    cmap: Dict[int, Tuple[int, int, int, int]] = {}
    
    def __init__(self, root: str = "data",
                crs: Optional[CRS] = None, 
                res: Optional[float] = None, 
                bands: Optional[Sequence[str]] = None, 
                transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
                cache: bool = True, 
                annotation_type: List[str] = [], 
                csv_satelliteData: str = "",
                fold_test: int = None,
                test_set: bool = False) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            annotation_type: list of string to specify which files to include 
                in the batch. Available choices are: ["del_mask", "gra_mask", 
                "lc_esa_mask", "lc_annual_mask", "cloud_mask"]
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """

        super().__init__(transforms)

        self.root = root
        self.cache = cache

        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        if csv_satelliteData == "":
            
            # Populate the dataset index
            ids = []
            for ann in annotation_type:
                if ann in self.file_types.keys():
                    regex = self.filename_glob.replace("S2L2A", self.file_types[ann]).replace(".tiff", ".tif")
                    name = os.path.join(root, "**", regex)
                    
                    ids.append(set([str(Path(file).stem).replace(self.file_types[ann], "") for file in glob.iglob(name, recursive=True)]))

            # consider only the images with all images from given annotation type
            intersection_ids = set.intersection(*ids)
            filepaths = []
            for idx in intersection_ids:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)

        else:
            df = pd.read_csv(csv_satelliteData, index_col=False)
            df = df[ (df["DEL"] == 1) & (df["GRA"] == 1) & (df["folder"] == "optimal")] # remove GRA if you would include only DEL masks
            
            # ONLY for Cross Validation
            if fold_test != None:
                if test_set:
                    df = df.loc[ df["fold"] == int(fold_test) ] # test/val fold (Leave one out cross validation)
                    print(df)
                else:
                    df = df.loc[ (df["fold"] != int(fold_test)) & (df["fold"] != 0)] # training set remaining folds

            listPaths = df["folderPath"].str.split("/").str[-1]
            filepaths = []
            for idx in listPaths:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)
            
        assert len(filepaths) > 0, f"No images found in {self.root}"

        i = 0
        for filepath in filepaths:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:

                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)
                    
                    # read date directly from the json of the Sentinel request
                    elif os.path.exists(filepath.replace(".tiff",".json")):
                        with open(filepath.replace(".tiff",".json"), "r") as f:
                            json_request = json.load(f)
                            list_dates = json_request['payload']["acquisition_date"]
                            mint, maxt = disambiguate_timestamp(list_dates[0], '%Y/%m/%d_%H:%M:%S')
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )
        
        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)
        
        self.annotation_type = annotation_type


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            dict: sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        
        hits = self.index.intersection(tuple(query), objects=True)

        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)
        
        sample = {"crs": self.crs, "bbox": query}
        sample["image"] = data.float()

        for annotation_cat in self.annotation_type:
            if annotation_cat in self.file_types.keys():
                ann_filepaths = [Path(f.replace(self.file_types["sentinel"], self.file_types[annotation_cat])).with_suffix(".tif") for f in filepaths]
                label = self._merge_files(ann_filepaths, query, self.band_indexes)
                sample[annotation_cat] = label.float()
        
        # path = "assets/images/" + filepaths[0].split("/")[4] + "_" + filepaths[0].split("/")[5] + "_"
        # if torch.any(sample["cloud_mask"] == 1):
        #     to_RGB_Mask(sample['del_mask'], path + "pre.png")

        del_mask = sample['del_mask']
        cloud_mask = sample['cloud_mask']

        # boolean mask for clouds
        bool_mask = cloud_mask.eq(1)

        # set the cloud position to 255
        del_mask[bool_mask] = 255
        sample['del_mask'] = del_mask

        # if torch.any(sample["cloud_mask"] == 1):
        #     to_RGB_Mask(sample['del_mask'], path + "post.png")

        del sample["cloud_mask"]
        # del sample["lc_esa_mask"] # Da usare in un secondo momento
        # del sample["lc_annual_mask"] # Da usare in un secondo momento
   
        # Kornia transforms
        # if self.transforms is not None:
        #     image, label_delineation, label_severity = self.transforms(sample["image"].unsqueeze(dim=0), sample["del_mask"].unsqueeze(dim=0), sample["gra_mask"].unsqueeze(dim=0))
        #     sample["image"] = image.squeeze(dim = 0)
        #     sample["del_mask"] = label_delineation.squeeze(dim = 0)

        # Albumentations transforms
        if self.transforms is not None:
            transformed = self.transforms(image=np.transpose(sample["image"].numpy(), (1, 2, 0)), delineation=np.transpose(sample["del_mask"].numpy(), (1, 2, 0)))
            sample["image"] = transformed["image"]
            sample["del_mask"] = transformed["delineation"]

        return sample


    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = round((query.maxx - query.minx) / self.res)
            out_height = round((query.maxy - query.miny) / self.res)
            count = len(band_indexes) if band_indexes else src.count  # custom line for band_indexes
            out_shape = (count, out_height, out_width)
            dest = src.read(
                indexes=band_indexes,
                out_shape=out_shape,
                window=from_bounds(*bounds, src.transform),
            )
        else:
            dest, _ = rasterio.merge.merge(
                vrt_fhs, bounds, self.res, indexes=band_indexes
            )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src


class DatasetCEMS_Severity(GeoDataset):
    filename_glob = "EMSR*AOI*S2L2A.tiff"
    # file_types = {
    #     "sentinel": "S2L2A",
    #     "del_mask": "DEL",
    #     "fep_mask": "FEP",
    #     "gra_mask": "GRA",
    #     "lc_esa_mask": "ESA_LC",
    #     "lc_annual_mask": "Annual9_LC",
    #     "cloud_mask": "CM"
    # }

    file_types = {
        "sentinel": "S2L2A",
        "gra_mask": "GRA",
        "cloud_mask": "CM"
    }
    
    rgb_bands = ["B04", "B03", "B02"]
    all_bands = ["B02", "B03", "B04", "B08"]
    filename_regex = ".*"
    date_format = "%Y%m%d"
    separate_files = False
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(self, root: str = "data",
                crs: Optional[CRS] = None, 
                res: Optional[float] = None, 
                bands: Optional[Sequence[str]] = None, 
                transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
                cache: bool = True, 
                annotation_type: List[str] = [], 
                csv_satelliteData: str = "",
                fold_test: int = None,
                test_set: bool = False,
                severity_convention: int = 4) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            annotation_type: list of string to specify which files to include 
                in the batch. Available choices are: ["del_mask", "gra_mask", 
                "lc_esa_mask", "lc_annual_mask", "cloud_mask"]
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """

        super().__init__(transforms)

        self.root = root
        self.cache = cache
        self.severity_convention = severity_convention

        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        if csv_satelliteData == "":
            
            # Populate the dataset index
            ids = []
            for ann in annotation_type:
                if ann in self.file_types.keys():
                    regex = self.filename_glob.replace("S2L2A", self.file_types[ann]).replace(".tiff", ".tif")
                    name = os.path.join(root, "**", regex)
                    
                    ids.append(set([str(Path(file).stem).replace(self.file_types[ann], "") for file in glob.iglob(name, recursive=True)]))

            # consider only the images with all images from given annotation type
            intersection_ids = set.intersection(*ids)
            filepaths = []
            for idx in intersection_ids:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)

        else:
            df = pd.read_csv(csv_satelliteData, index_col=False)
            df = df[ (df["GRA"] == 1) & (df["folder"] == "optimal")]
            
            # ONLY for Cross Validation
            if fold_test != None:
                if test_set:
                    df = df.loc[ df["fold"] == int(fold_test) ] # test/val fold (Leave one out cross validation)
                else:
                    df = df.loc[ (df["fold"] != int(fold_test)) & (df["fold"] != 0)] # training set remaining folds

            listPaths = df["folderPath"].str.split("/").str[-1]
            filepaths = []
            for idx in listPaths:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)

        assert len(filepaths) > 0, f"No images found in {self.root}"

       
        i = 0
        for filepath in filepaths:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:

                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)
                    
                    # read date directly from the json of the Sentinel request
                    elif os.path.exists(filepath.replace(".tiff",".json")):
                        with open(filepath.replace(".tiff",".json"), "r") as f:
                            json_request = json.load(f)
                            list_dates = json_request['payload']["acquisition_date"]
                            mint, maxt = disambiguate_timestamp(list_dates[0], '%Y/%m/%d_%H:%M:%S')
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )
        
        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)
        
        self.annotation_type = annotation_type


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            dict: sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        
        hits = self.index.intersection(tuple(query), objects=True)

        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)
        
        sample = {"crs": self.crs, "bbox": query}
        sample["image"] = data.float()

        for annotation_cat in self.annotation_type:
             
            if annotation_cat in self.file_types.keys():
                ann_filepaths = [Path(f.replace(self.file_types["sentinel"], self.file_types[annotation_cat])).with_suffix(".tif") for f in filepaths]

                label = self._merge_files(ann_filepaths, query, self.band_indexes)
                sample[annotation_cat] = label.float() 

        # REMAPPING severity levels (NOT USED)
        if self.severity_convention == 3:
            unique_severity_levels = torch.unique(sample['gra_mask'])
            unique_severity_levels.sort()
            # unique_severity_levels = unique_severity_levels[1:] # remove zero value

            if  unique_severity_levels.tolist() == [0,1,2,3,4] or \
                unique_severity_levels.tolist() == [0,2,3,4] or \
                unique_severity_levels.tolist() == [0,1,3,4] or \
                unique_severity_levels.tolist() == [0,1,3] or \
                unique_severity_levels.tolist() == [0,1,4] or \
                unique_severity_levels.tolist() == [0,2,3] or \
                unique_severity_levels.tolist() == [0,2,4] or \
                unique_severity_levels.tolist() == [0,3,4] or \
                unique_severity_levels.tolist() == [0,1]:

                grading_lut = torch.tensor([0,1,1,2,3])
                sample["gra_mask"] = grading_lut[torch.floor(sample["gra_mask"]).long()]
                sample["gra_mask"] = sample["gra_mask"].float()
            
            elif unique_severity_levels.tolist() == [0,1,2,4] or \
                 unique_severity_levels.tolist() == [0,1,2,3] or \
                 unique_severity_levels.tolist() == [0,1,2]:
                
                grading_lut = torch.tensor([0,1,2,3,3])
                sample["gra_mask"] = grading_lut[torch.floor(sample["gra_mask"]).long()]
                sample["gra_mask"] = sample["gra_mask"].float()
        
        # NORMALIZATION severity levels
        min_value = 0
        max_value = self.severity_convention
        sample['gra_mask'] = (sample['gra_mask'] - min_value) / (max_value - min_value)
        sample[sample['gra_mask'] > 15] = 255 # restore no data pixel after normalization

        gra_mask = sample['gra_mask']
        cloud_mask = sample['cloud_mask']

        # boolean mask for clouds
        # bool_mask_cloud = cloud_mask.eq(1)                                    # Only cloud masked
        # bool_mask_cloudShadow = cloud_mask.eq(3)                              # Only cloud's shadow
        bool_mask_cloud = torch.logical_or(cloud_mask == 1, cloud_mask == 3)    # Cloud and cloud's shadow are masked

        # set the cloud labels to 255
        gra_mask[bool_mask_cloud] = 255

        sample['gra_mask'] = gra_mask
        
        del sample["cloud_mask"]
        # del sample["lc_esa_mask"]     # Da usare in un secondo momento
        # del sample["lc_annual_mask"]  # Da usare in un secondo momento
        
        # Kornia transformation
        # if self.transforms is not None:
        #     image, label_severity = self.transforms(sample["image"].unsqueeze(dim=0), sample["gra_mask"].unsqueeze(dim=0))
        #     sample["image"] = image.squeeze(dim = 0)
        #     sample["gra_mask"] = label_severity.squeeze(dim = 0)

        # Albumentations transformation
        if self.transforms is not None:
            transformed = self.transforms(image=np.transpose(sample["image"].numpy(), (1, 2, 0)), grading=np.transpose(sample["gra_mask"].numpy(), (1, 2, 0)))

            sample["image"] = transformed["image"]
            sample["gra_mask"] = transformed["grading"]
            
        return sample

    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = round((query.maxx - query.minx) / self.res)
            out_height = round((query.maxy - query.miny) / self.res)
            count = len(band_indexes) if band_indexes else src.count  # custom line for band_indexes
            out_shape = (count, out_height, out_width)
            dest = src.read(
                indexes=band_indexes,
                out_shape=out_shape,
                window=from_bounds(*bounds, src.transform),
            )
        else:
            dest, _ = rasterio.merge.merge(
                vrt_fhs, bounds, self.res, indexes=band_indexes
            )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src
        

class DatasetCEMS_Multitask(GeoDataset):
    filename_glob = "EMSR*AOI*S2L2A.tiff"
    # file_types = {
    #     "sentinel": "S2L2A",
    #     "del_mask": "DEL",
    #     "fep_mask": "FEP",
    #     "gra_mask": "GRA",
    #     "lc_esa_mask": "ESA_LC",
    #     "lc_annual_mask": "Annual9_LC",
    #     "cloud_mask": "CM"
    # }

    file_types = {
        "sentinel": "S2L2A",
        "gra_mask": "GRA",
        "del_mask": "DEL",
        "cloud_mask": "CM"
    }
    
    rgb_bands = ["B04", "B03", "B02"]
    all_bands = ["B02", "B03", "B04", "B08"]
    filename_regex = ".*"
    date_format = "%Y%m%d"
    separate_files = False
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(self, root: str = "data",
                crs: Optional[CRS] = None, 
                res: Optional[float] = None, 
                bands: Optional[Sequence[str]] = None, 
                transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None, 
                cache: bool = True, 
                annotation_type: List[str] = [], 
                csv_satelliteData: str = "",
                fold_test: int = None,
                test_set: bool = False,
                severity_convention: int = 4) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            annotation_type: list of string to specify which files to include 
                in the batch. Available choices are: ["del_mask", "gra_mask", 
                "lc_esa_mask", "lc_annual_mask", "cloud_mask"]
        Raises:
            FileNotFoundError: if no files are found in ``root``
        """

        super().__init__(transforms)

        self.root = root
        self.cache = cache
        self.severity_convention = severity_convention

        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)

        if csv_satelliteData == "":
            
            # Populate the dataset index
            ids = []
            for ann in annotation_type:
                if ann in self.file_types.keys():
                    regex = self.filename_glob.replace("S2L2A", self.file_types[ann]).replace(".tiff", ".tif")
                    name = os.path.join(root, "**", regex)
                    
                    ids.append(set([str(Path(file).stem).replace(self.file_types[ann], "") for file in glob.iglob(name, recursive=True)]))

            # consider only the images with all images from given annotation type
            intersection_ids = set.intersection(*ids)
            filepaths = []
            for idx in intersection_ids:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)

        else:
            df = pd.read_csv(csv_satelliteData, index_col=False)
            df = df[ (df["GRA"] == 1) & (df["folder"] == "optimal")]
            
            # ONLY for Cross Validation
            if fold_test != None:
                if test_set:
                    df = df.loc[ df["fold"] == int(fold_test) ] # test/val fold (Leave one out cross validation)
                    # df.to_csv("assets/testSET.csv")
                else:
                    df = df.loc[ (df["fold"] != int(fold_test)) & (df["fold"] != 0)] # training set remaining folds
                    # df.to_csv("assets/trainingSET.csv")

            listPaths = df["folderPath"].str.split("/").str[-1]
            filepaths = []
            for idx in listPaths:
                for f in glob.iglob(pathname, recursive=True):
                    if idx in f:
                        filepaths.append(f)

        assert len(filepaths) > 0, f"No images found in {self.root}"

        i = 0
        for filepath in filepaths:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:

                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    # read date directly from the json of the Sentinel request
                    elif os.path.exists(filepath.replace(".tiff",".json")):
                        with open(filepath.replace(".tiff",".json"), "r") as f:
                            json_request = json.load(f)
                            list_dates = json_request['payload']["acquisition_date"]
                            mint, maxt = disambiguate_timestamp(list_dates[0], '%Y/%m/%d_%H:%M:%S')
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )
        
        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)
        
        self.annotation_type = annotation_type


    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            dict: sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        
        hits = self.index.intersection(tuple(query), objects=True)

        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)
        
        sample = {"crs": self.crs, "bbox": query}
        sample["image"] = data.float()

        for annotation_cat in self.annotation_type:
             
            if annotation_cat in self.file_types.keys():
                ann_filepaths = [Path(f.replace(self.file_types["sentinel"], self.file_types[annotation_cat])).with_suffix(".tif") for f in filepaths]

                label = self._merge_files(ann_filepaths, query, self.band_indexes)
                sample[annotation_cat] = label.float() 

        # REMAPPING severity levels (not used)
        if self.severity_convention == 3:
            unique_severity_levels = torch.unique(sample['gra_mask'])
            unique_severity_levels.sort()
            # unique_severity_levels = unique_severity_levels[1:] # remove zero value

            if  unique_severity_levels.tolist() == [0,1,2,3,4] or \
                unique_severity_levels.tolist() == [0,2,3,4] or \
                unique_severity_levels.tolist() == [0,1,3,4] or \
                unique_severity_levels.tolist() == [0,1,3] or \
                unique_severity_levels.tolist() == [0,1,4] or \
                unique_severity_levels.tolist() == [0,2,3] or \
                unique_severity_levels.tolist() == [0,2,4] or \
                unique_severity_levels.tolist() == [0,3,4] or \
                unique_severity_levels.tolist() == [0,1]:

                grading_lut = torch.tensor([0,1,1,2,3])
                sample["gra_mask"] = grading_lut[torch.floor(sample["gra_mask"]).long()]
                sample["gra_mask"] = sample["gra_mask"].float()
            
            elif unique_severity_levels.tolist() == [0,1,2,4] or \
                 unique_severity_levels.tolist() == [0,1,2,3] or \
                 unique_severity_levels.tolist() == [0,1,2]:
                
                grading_lut = torch.tensor([0,1,2,3,3])
                sample["gra_mask"] = grading_lut[torch.floor(sample["gra_mask"]).long()]
                sample["gra_mask"] = sample["gra_mask"].float()
        
        # NORMALIZATION severity levels
        min_value = 0
        max_value = self.severity_convention
        sample['gra_mask'] = (sample['gra_mask'] - min_value) / (max_value - min_value)
        sample[sample['gra_mask'] > 15] = 255 # restore no data pixel after normalization

        # DEBUG -> comment normalization
        # path = "assets/images/" + filepaths[0].split("/")[4] + "_" + filepaths[0].split("/")[5] + "_"
        # if torch.any(sample["cloud_mask"] == 1):
        #     to_RGB_Mask(sample['gra_mask'], path + "pre.png")
        #     to_RGB_sample(sample['image'], path + "S2_L2A.png")

        gra_mask = sample['gra_mask']
        del_mask = sample['del_mask']
        cloud_mask = sample['cloud_mask']

        # boolean mask for clouds
        # bool_mask_cloud = cloud_mask.eq(1)                                    # Only cloud masked
        # bool_mask_cloudShadow = cloud_mask.eq(3)                              # Only cloud's shadow
        bool_mask_cloud = torch.logical_or(cloud_mask == 1, cloud_mask == 3)    # Cloud and cloud's shadow are masked

        # set the cloud labels to 255
        gra_mask[bool_mask_cloud] = 255
        del_mask[bool_mask_cloud] = 255

        sample['del_mask'] = del_mask
        sample['gra_mask'] = gra_mask

        # DEBUG  -> comment normalization
        # if torch.any(sample["cloud_mask"] == 1):
        #     to_RGB_Mask(sample['del_mask'], path + "postDEL.png")
        #     to_RGB_Mask(sample['gra_mask'], path + "postGRA.png")
        
        del sample["cloud_mask"]
        # del sample["lc_esa_mask"]     # Da usare in un secondo momento
        # del sample["lc_annual_mask"]  # Da usare in un secondo momento
        
        # Kornia transformation
        # if self.transforms is not None:
        #     image, label_delineation, label_severity = self.transforms(sample["image"].unsqueeze(dim=0), sample["del_mask"].unsqueeze(dim=0), sample["gra_mask"].unsqueeze(dim=0))
        #     sample["image"] = image.squeeze(dim = 0)
        #     sample["del_mask"] = label_delineation.squeeze(dim = 0)
        #     sample["gra_mask"] = label_severity.squeeze(dim = 0)

        # Albumentations transformation
        if self.transforms is not None:
            transformed = self.transforms(image=np.transpose(sample["image"].numpy(), (1, 2, 0)), delineation=np.transpose(sample["del_mask"].numpy(), (1, 2, 0)), grading=np.transpose(sample["gra_mask"].numpy(), (1, 2, 0)))

            sample["image"] = transformed["image"]
            sample["del_mask"] = transformed["delineation"]
            sample["gra_mask"] = transformed["grading"]
            
            # DEBUG -> comment normalization
            # if torch.any(sample["gra_mask"] > 1) and (int(filepaths[0].split("/")[4].replace("EMSR","")) == 300 or int(filepaths[0].split("/")[4].replace("EMSR","")) == 510):
            #     num = np.random.randint(1, 30)
            #     path = "assets/images/" + filepaths[0].split("/")[4] + "_" + filepaths[0].split("/")[5] + "_"
            #     to_RGB_sample(sample['image'], path + str(num) + "_S2L2A.png")
            #     to_RGB_Mask(sample['gra_mask'], path + str(num) + "_GRA.png")

        return sample


    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
            out_width = round((query.maxx - query.minx) / self.res)
            out_height = round((query.maxy - query.miny) / self.res)
            count = len(band_indexes) if band_indexes else src.count  # custom line for band_indexes
            out_shape = (count, out_height, out_width)
            dest = src.read(
                indexes=band_indexes,
                out_shape=out_shape,
                window=from_bounds(*bounds, src.transform),
            )
        else:
            dest, _ = rasterio.merge.merge(
                vrt_fhs, bounds, self.res, indexes=band_indexes
            )

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src