import os
import sys
from os.path import join as pjoin
from datetime import datetime
import argparse
from MyVectorNet.utils.ArgoverseLoader import ArgoverseInMem, GraphData
from MyVectorNet.trainer.vectornet_trainer import VectorNetTrainer

sys.path.append("utils")

def test(args):
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = pjoin(args.save_dir, time_stamp)
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("Not empty output folder!")
    else:
        os.makedirs(output_dir)

    # data loading
    try:
        test_set = ArgoverseInMem(pjoin(args.data_root, "{}_intermediate".format(args.split)))
    except:
        raise Exception("Failed to load the data!")

    # init trainer
    trainer = VectorNetTrainer(
        trainset=test_set,
        evalset=test_set,
        testset=test_set,
        batch_size=args.batch_size,
        num_workers=0,
        aux_loss=True,
        with_cuda=args.with_cuda,
        save_folder=output_dir,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    trainer.test(convert_coordinate=True, plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--data_root", type=str, default="data/interm_data")
    parser.add_argument("-s", "--split", type=str, default="val")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-w", "--num_workers", type=int, default=0)
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True)
    parser.add_argument("-rc", "--resume_checkpoint", type=str, default=None)
    parser.add_argument("-rm", "--resume_model", type=str, default="test_model/best_VectorNetOriginal.pth")
    parser.add_argument("-d", "--save_dir", type=str, default="test_result")

    args = parser.parse_args()
    test(args)

