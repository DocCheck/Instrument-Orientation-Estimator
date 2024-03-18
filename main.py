import argparse
import os
import sys
import train, predict


def parse_opt(parser, known=False):
    parser.add_argument("--input-data", "--input-data-path", type=str,
                        default="data/dataset/Orig_OPBesteck_dataset_rot_est_all/",
                        help="Input dataset path for training the model")
    parser.add_argument("--model-path", type=str, default="data/models/model_default/model_ori_est.pt",
                        help="Model name")
    parser.add_argument("--n-class", type=int, default=357, help="Number of output classes (angles)")
    parser.add_argument("--train-val", type=float, nargs='+', default=[0.95, 0.05],
                        help="Train and Validation proportion", required=False)
    parser.add_argument("--batch-size", type=int, default=256, help="Total batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=1000, help="Total training epochs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=128, help="input image size (pixels)")
    parser.add_argument("--valid-period", type=int, default=5, help="run validation intervals (every x epochs)")
    parser.add_argument("--visualization", action="store_false", help="visualize the training/validation process")
    parser.add_argument("--diff-t", type=int, default=3,
                        help="Maximum difference threshold between the predicted and the true angle")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    return parser.parse_known_args()[0] if known else parser.parse_args()


def train_model(opt):
    if not os.path.exists(opt.input_data):
        print("The dataset path does not exist !!!")
        return
    else:
        train.train_model(opt)


def predict_model(opt):
    if not os.path.exists(opt.input_data):
        print("The dataset path does not exist !!!")
        return
    elif not os.path.exists(opt.model_path):
        print("The model does not exist !!!")
        return
    else:
        predict.predict_model(opt)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    action = sys.argv[-1]
    opt = parse_opt(parser)

    if action == "train-model":
        train_model(opt)
    elif action == "predict-model":
        predict_model(opt)
    else:
        print("WRONG ACTION")
        exit(1)
