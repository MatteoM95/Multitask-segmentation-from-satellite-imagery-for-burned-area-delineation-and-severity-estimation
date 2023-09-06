# https://github.com/ai4er-cdt/WildfireDistribution/blob/799c5be199ed4c00bfa9add0574df2f81bd402a9/src/samplers/custom_samplers.py
# https://github.com/ai4er-cdt/WildfireDistribution/blob/799c5be199ed4c00bfa9add0574df2f81bd402a9/src/datamodules/landcover.py
import numpy as np
import random
from PIL import Image
import time
import os
import math
from typing import Optional, Iterator, Union, Tuple, List
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.samplers.utils import get_random_bounding_box
from torchgeo.samplers.constants import Units


class ConstrainedRandomBatchGeoSampler(RandomBatchGeoSampler):
    """Returns batches of samples that meet the constraints specified:
        1. Proportion of samples with a burn proportion > 0 : 'burn_prop'
        2. The rest of the samples will make up 1-'burn_prop' proportion and have a sample burn proportion = 0
    Args:
        dataset: the dataset to take samples from.
        size: size of patch in lat/lon.
        batch_size: the number of samples per batch.
        length: number of samples (in total) to take per epoch. Note: this means no. of batches = length/batch_size.
        burn_prop: proportion of returned samples present that are "burned". "burned" is defined as > 0 burned pixels.
        roi: region of interest to take samples from.
        units: Units.<PIXELS/CRS> depending on if patch size is in pixels or lat/lon
    Returns:
        constrained_samples: set of samples that meet the specified constraints.
    """

    # Setup the attributes required for constraint implementation
    burn_samples_required = None
    not_burned_samples_required = None
    burn_prop_batch = None

    # Define the constructor
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        burn_prop_batch: float = 0.7,
        burn_area_prop: float = 0.4,
        mask: str = "del_mask"
    ) -> None:

        # Ensures that the input dataset is of type: IntersectionDataset
        if not isinstance(dataset, GeoDataset):
            raise TypeError(
                "Input dataset to sampler must be of type: IntersectionDataset."
            )

        # Use init from RandomBatchGeoSampler parent class
        super().__init__(dataset, size, batch_size, length, roi, units)

        # Save the dataset and input constraints to the object
        self.dataset = dataset
        self.burn_prop_batch = burn_prop_batch
        self.mask_name = mask
        self.burn_area_prop = burn_area_prop

        # Set the number of samples required of not burned/burned types
        self.burn_samples_required = math.ceil(burn_prop_batch * self.batch_size)
        self.not_burned_samples_required = self.batch_size - self.burn_samples_required

        self.counter_save_image = 0

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Defines a generator function to produce batches of areas to sample next.
        Returns:
            List((minx, maxx, miny, maxy, mint, maxt)) coordinates to index a dataset
        """

        # Generate samples in len(self) = length/batch_size = number of batches required
        for _ in range(len(self)):

            # # Uncomment if you would choose a random tile from the same EMSR_AOI
            # hit = random.choice(self.hits)
            # bounds = BoundingBox(*hit.bounds)

            # Fill a new batch of samples
            batch = []
            count_attempt = 0
            max_num_attempts = 80

            while ( self.not_burned_samples_required != 0 or self.burn_samples_required != 0 ):
                
                count_attempt += 1

                # Uncomment if you would choose a random tile from different randomly EMSR_AOI
                hit = random.choice(self.hits)
                bounds = BoundingBox(*hit.bounds)

                # Choose a random sample within that tile
                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                samp_burn_prop = self.get_burn_proportion(bounding_box)

                # If we have a "not-burned" sample and we require "not-burned" samples
                if samp_burn_prop >= 0.01 and samp_burn_prop <= self.burn_area_prop and self.not_burned_samples_required != 0:
                    # print(f"POCA AREA BRUCIATA: Area Buned: {samp_burn_prop}, count: {count_attempt},   NON Burnt {self.not_burned_samples_required},     Burnt {self.burn_samples_required}")
                    self.not_burned_samples_required -= 1
                    batch.append(bounding_box)

                # If we have a "burn" sample and we require "burn" samples
                elif samp_burn_prop > self.burn_area_prop and self.burn_samples_required != 0:
                    # print(f"TANTA AREA BRUCIATA: Area Buned: {samp_burn_prop}, count: {count_attempt},   NON Burnt {self.not_burned_samples_required},    Burnt {self.burn_samples_required}")
                    self.burn_samples_required -= 1
                    batch.append(bounding_box)
                
                # to avoid infinite loop, after this all images found are part of the batch
                elif count_attempt > max_num_attempts: 
                    # print(f"IMMAGINE DI RIEMPIMENTO Area Buned: {samp_burn_prop}, count: {count_attempt},    NON Burnt {self.not_burned_samples_required},      Burnt {self.burn_samples_required}")
                    print("--- Reached the limit of attempts in geoSampler ---")

                    if self.not_burned_samples_required != 0:
                        self.not_burned_samples_required -= 1
                        batch.append(bounding_box)

                    elif self.burn_samples_required != 0:
                        self.burn_samples_required -= 1
                        batch.append(bounding_box)

            # Return the batch of balanced samples we have gathered
            yield batch

            # Reset requirements for next batch generation
            self.burn_samples_required = math.ceil( self.burn_prop_batch * self.batch_size )
            self.not_burned_samples_required = ( self.batch_size - self.burn_samples_required )

    def get_burn_proportion(self, bounding_box):

        """Returns the burn proportion found within a given bounding box.
        Returns:
            burn_prop: the burn proportion present within the bounding box.
        """

        # Obtain the burn data within the bounding box
        burn_data = self.dataset[bounding_box][self.mask_name]

        # save image given a bounding box
        # self.save_image_RGB(self.dataset[bounding_box]["image"])
        # self.counter_save_image += 1

        # Get burn proportion within the bounding box
        non_zero_count = int( ((burn_data > 0) & (burn_data < 120)).sum() )
        samp_burn_prop = non_zero_count / burn_data.numel()
        
        return samp_burn_prop


    # def save_image_RGB(self, image_tensor):
        
    #     contrast_coeff = 10000/255*3.5*5.5      # image color contrast 
    #     brightness_coeff = -10 #-20             # image brightness 
    #     image_np = image_tensor.numpy()

    #     image_np = np.transpose(image_np, (1, 2, 0))
        
    #     # imageRGB = image_np[:,:,[3,2,1]]
    #     imageRGB = np.multiply(image_np[:,:,[3,2,1]], contrast_coeff) + brightness_coeff
    #     imageRGB[ imageRGB > 255 ] = 255
    #     imageRGB[ imageRGB < 0 ] = 0        
    #     image_RGB = Image.fromarray(imageRGB.astype(np.uint8))
    #     path_folder = "assets/images_512/"
    #     image_name = f"{self.counter_save_image}.png"

    #     if not os.path.exists(os.path.dirname(path_folder)):
    #         os.makedirs(os.path.dirname(path_folder))

    #     path = path_folder + image_name
    #     image_RGB.save(path)