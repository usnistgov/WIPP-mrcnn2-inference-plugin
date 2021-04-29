from inference import stitched_inference
from inference_utils import CleanMask
from inference import generate_inference_model
from inference import preprocess
from PIL import Image
import numpy as np
from os import listdir
from skimage.util import img_as_ubyte
from skimage.transform import resize
import skimage.io

# example
# python inference_script.py /data/kimjb/Mask_RCNN/image_test/to_caltech/exp3/AUTO0218_D08_T0001F004L01A01Z01C01.tif /data/kimjb/Mask_RCNN/logs/cells20180719T1559/mask_rcnn_cells.h5. --cropsize=256 --padding=40

if __name__ == '__main__':
    import argparse
    import os 
    import sys

    parser = argparse.ArgumentParser(prog='threshold',
                                     description='Create a binary image from a grayscale image and threshold value')

    # Define arguments
    parser.add_argument('--inputImages', dest='input_images', type=str,
                        help='filepath to the directory containing the images', required=True)
    parser.add_argument('--output', dest='output_folder', type=str, required=True)

    parser.add_argument('--cropsize', required=False,
                        default='256',
                        help='Size of patches. Must be multiple of 256')
    parser.add_argument('--padding', required=False,
                        default='50',
                        help='Amount of overlapping pixels along one axis')
    parser.add_argument('--threshold', required=False,
                        default='10',
                        help='Min number of pixels belonging to a cell.')

    # Parse arguments
    args = parser.parse_args()
    input_images = args.input_images
    output_folder = args.output_folder


    args = parser.parse_args()


    padding = int(args.padding)
    cropsize = int(args.cropsize) 
    threshold = int(args.threshold)

    mrcnn_model_path = "./mrcnn_pretrained.h5"

    model = generate_inference_model(mrcnn_model_path, cropsize)
    import time

    images = listdir(input_images)

    print(input_images)

    print(images)

    img = np.zeros((len(images), 1040, 1392))



    for i in range(len(img)):

        tic = 0
        toc = 0

        print('Start processing image ' + images[i])

        tic = time.perf_counter()

        image_resized = img_as_ubyte(resize(np.array(Image.open(input_images + "/" + images[i])), (1040, 1392)))

        img[i, :, :] = image_resized

        tac = time.perf_counter()

        print(input_images)

        mija = input_images + "/" + images[i]
        print(mija)
        mija1 = preprocess(mija)
        stitched_inference_stack, num_times_visited = stitched_inference(mija1, cropsize, model, padding=padding)



        masks = CleanMask(stitched_inference_stack, threshold, )
        masks.merge_cells()
        masks.save(output_folder + "/" + images[i])

        #skimage.io.imsave(output_folder + "/" + images[i], masks[i].astype(np.uint16), 'tifffile', False,
        #                  tile=(1024, 1024))

        print('End processing image' + images[i])

        toc = time.perf_counter()

        # print(f"before mrcnn_infer {tac - tic:0.4f} seconds")
        # print(f"after mrcnn_infer {toc - tac:0.4f} seconds")

        print(f"Processing the image" + images[i] + f"took {toc - tic:0.4f} seconds")





    


