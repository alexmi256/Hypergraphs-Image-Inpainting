import argparse
import os
import pathlib
import time


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()
        self.opt = self.parser.parse_args()

    def initialize(self):
        self.parser.add_argument("--dataset", type=str, default="celeba-hq", help="dataset name for testing")
        input_data = self.parser.add_mutually_exclusive_group(required=True)
        input_data.add_argument("--test_dir", type=pathlib.Path, default=None, help="directory where all input images are stored")
        input_data.add_argument(
            "--test_file_path",
            type=pathlib.Path,
            default=None,
            help="single input file to test",
        )
        self.parser.add_argument("--base_dir", type=str, default="Testing", help="directory where results will be output")
        self.parser.add_argument("--pretrained_model_dir", type=str, default="pretrained_models", help="Directory where pretrained models are stored")
        self.parser.add_argument("--checkpoint_prefix", type=str, default="ckpt")

        self.parser.add_argument("--random_mask", type=int, default=0, choices=[0, 1], help="0 -> Center 128 * 128 mask, 1 -> random mask")
        self.parser.add_argument(
            "--random_mask_type", type=str, default="irregular_mask", choices=["irregular_mask", "random_rect"], help="Mask type to generate"
        )
        self.parser.add_argument("--min_strokes", type=int, default=16, help="Minimum number of strokes to generate when using random mask")
        self.parser.add_argument("--max_strokes", type=int, default=48, help="Maximum number of strokes to generate when using random mask")

        self.parser.add_argument("--image_shape", type=str, default="256,256,3", help="Image height and width, comma separated")
        self.parser.add_argument("--test_num", type=int, default=-1)
        self.parser.add_argument("--mode", type=str, default="save")

    def parse(self):
        str_image_shape = self.opt.image_shape.split(",")
        self.opt.image_shape = [int(x) for x in str_image_shape]

        self.opt.date_str = time.strftime("%Y%m%d-%H%M%S")
        self.opt.model_name = "HypergraphII"
        self.opt.model_folder = self.opt.date_str + "_" + self.opt.model_name
        self.opt.model_folder += "_" + self.opt.dataset
        self.opt.model_folder += "_shape" + str(self.opt.image_shape[0]) + "x" + str(self.opt.image_shape[1])
        self.opt.model_folder += "_center_mask" if self.opt.random_mask == 0 else "_random_mask"

        if not os.path.isdir(self.opt.base_dir):
            os.mkdir(self.opt.base_dir)

        self.opt.testing_dir = os.path.join(self.opt.base_dir, self.opt.model_folder)

        if not os.path.isdir(self.opt.testing_dir):
            os.mkdir(self.opt.testing_dir)

        args = vars(self.opt)

        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print(str(k), ":", str(v))
        print("-" * 20 + " End " + "-" * 20)

        return self.opt
