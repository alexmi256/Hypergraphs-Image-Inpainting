import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.model import Model
from options.test_options import TestOptions
from utils.util import irregular_mask, rectangle_mask, save_images

SUPPORTED_IMAGE_TYPES = ["jpg", "png", "jpeg"]


def load_image(image_file, config):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.image.resize(image, [config.image_shape[0], config.image_shape[1]])
    image = image / 255.0

    return image


def test(config):
    entries = []
    count = 0
    # Do some basic error checking on the file(s) to process
    if config.test_file_path:
        print(f"Running with single file {config.test_file_path}")
        if config.test_file_path.suffix.lower()[1:] not in SUPPORTED_IMAGE_TYPES:
            raise Exception(f"File {config.test_file_path} is not supported")
        if not config.test_file_path.exists():
            raise Exception(f"File {config.test_file_path} does not exist")
        if not config.test_file_path.is_file():
            raise Exception(f"File {config.test_file_path} is not a file")
        entries.append(config.test_file_path)
    else:
        print(f"Running with directory {config.test_dir}")
        for entry in config.test_dir.iterdir():
            if entry.is_file() and entry.suffix.lower()[1:] in SUPPORTED_IMAGE_TYPES:
                entries.append(entry)
    # Process all entries
    for entry in entries:
        if config.random_mask == 1:
            mask = irregular_mask(config.image_shape[0], config.image_shape[1], config.min_strokes, config.max_strokes)
        else:
            mask = rectangle_mask(config.image_shape[0], config.image_shape[1], shape=config.mask_shape, position=config.mask_position)

        gt_image = load_image(str(entry), config)
        gt_image = np.expand_dims(gt_image, axis=0)

        input_image = np.where(mask == 1, 1, gt_image)

        prediction_coarse, prediction_refine = generator([input_image, mask], training=False)
        prediction_refine = prediction_refine * mask + gt_image * (1 - mask)

        output_file = str(Path(config.testing_dir).joinpath(entry.name))
        save_images(input_image[0, ...], gt_image[0, ...], prediction_coarse[0, ...], prediction_refine[0, ...], output_file)

        count += 1
        # I think this is just a way to exit early ?
        if count == config.test_num:
            return
        print("-" * 20)


if __name__ == "__main__":
    # Loading the arguments
    config = TestOptions().parse()

    model = Model()
    generator = model.build_generator()

    checkpoint = tf.train.Checkpoint(generator=generator)
    checkpoint.restore(os.path.join(config.pretrained_model_dir, config.checkpoint_prefix))

    test(config)
