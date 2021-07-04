import argparse
import os
import pathlib
import time


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()
        self.opt = self.parser.parse_args()

    def initialize(self):
        self.parser.add_argument("--dataset", type=str, default="celeba-hq", help="Dataset name for training")
        self.parser.add_argument("--train_dir", type=pathlib.Path, help="directory where all images are stored")
        self.parser.add_argument(
            "--train_file_path",
            type=pathlib.Path,
            help="The file storing the names of the file for training (If not provided training will happen for all images in train_dir)",
        )
        self.parser.add_argument("--gpu_ids", type=str, default="0", help="GPU to be used")
        self.parser.add_argument("--base_dir", type=str, default="Training")
        self.parser.add_argument("--checkpoints_dir", type=str, default="training_checkpoints", help="here models are saved during training")
        self.parser.add_argument("--pretrained_model_dir", type=str, default="", help="pretrained model are provided here")
        self.parser.add_argument("--batch_size", type=int, default=1, help="batch size used during training")
        self.parser.add_argument("--buffer_size", type=int, default=500, help="buffer size for data")

        self.parser.add_argument("--random_mask", type=int, default=0, choices=[0, 1], help="0 -> Square of 128 * 128, 1 -> Random lines of different sizes")
        self.parser.add_argument(
            "--mask_shape",
            type=str,
            default="square_128",
            help="Mark size and shape. rect or square. random or not. width and/or height. ex. random_rect_64_32",
        )
        self.parser.add_argument(
            "--mask_position", type=str, default="uniform", choices=["uniform", "center", "random"], help="Where the mark will occur in image"
        )
        self.parser.add_argument(
            "--incremental_training", type=int, default=1, choices=[0, 1], help="1 -> using incremental training, 0 -> not using incremental training"
        )
        self.parser.add_argument(
            "--mask_json", type=pathlib.Path, help="JSON file which has {'image_path': [start_x, start_y, end_x, end_y], ...} for mask data"
        )

        self.parser.add_argument("--epochs", type=int, default=200)
        self.parser.add_argument("--valid_l1_loss", type=float, default=0.2)
        self.parser.add_argument("--hole_l1_loss", type=float, default=1)
        self.parser.add_argument("--edge_loss", type=float, default=0.05)
        self.parser.add_argument("--gan_loss", type=float, default=0.002)
        self.parser.add_argument("--pl_comp", type=float, default=0.0001)
        self.parser.add_argument("--pl_out", type=float, default=0.0001)

        self.parser.add_argument("--learning_rate", type=float, default=1e-4)
        self.parser.add_argument("--decay_rate", type=float, default=0.96)
        self.parser.add_argument("--decay_steps", type=int, default=50000)

        self.parser.add_argument("--image_shape", type=str, default="256,256,3")

    def parse(self):
        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            int_id = int(str_id)
            if int_id >= 0:
                self.opt.gpu_ids.append(str(int_id))

        str_image_shape = self.opt.image_shape.split(",")
        self.opt.image_shape = [int(x) for x in str_image_shape]

        if len(self.opt.gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.opt.gpu_ids)

        self.opt.date_str = time.strftime("%Y%m%d-%H%M%S")
        self.opt.model_name = "HypergraphII"
        self.opt.model_folder = self.opt.model_name
        self.opt.model_folder += "_" + self.opt.dataset
        self.opt.model_folder += "_shape" + str(self.opt.image_shape[0]) + "x" + str(self.opt.image_shape[1])
        self.opt.model_folder += "_center_mask" if self.opt.random_mask == 0 else "_random_mask"
        self.opt.model_folder += "_incremental" if self.opt.incremental_training == 1 else ""

        if not os.path.isdir(self.opt.base_dir):
            os.mkdir(self.opt.base_dir)

        self.opt.training_dir = os.path.join(self.opt.base_dir, self.opt.model_folder)
        self.opt.checkpoint_saving_dir = os.path.join(self.opt.base_dir, self.opt.model_folder, self.opt.checkpoints_dir)

        if not os.path.isdir(self.opt.training_dir):
            os.mkdir(self.opt.training_dir)

        if not os.path.isdir(self.opt.checkpoint_saving_dir):
            os.mkdir(self.opt.checkpoint_saving_dir)

        args = vars(self.opt)

        print("-" * 20 + " Options " + "-" * 20)
        for k, v in sorted(args.items()):
            print(str(k), ":", str(v))
        print("-" * 20 + " End " + "-" * 20)

        return self.opt


options = TrainOptions()
args = options.parse()

# python training.py --dataset custom --train_dir frames/features/cropped --mask_shape random_rect_128_64 --mask_position uniform
# python training.py --dataset custom --train_dir frames/features/cropped --mask_json frames/features/cropped_and_resized_mask_positions.json
