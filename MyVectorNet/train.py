import os
import sys
from datetime import datetime
import json
import argparse

from torch_geometric.data import DataLoader
from MyVectorNet.utils.ArgoverseLoader import ArgoverseInMem, GraphData
from MyVectorNet.trainer.vectornet_trainer import VectorNetTrainer


TEST = False
sys.path.append("utils")


def train(args):
    # data loading
    train_set = ArgoverseInMem(os.path.join(args.data_root, "train_intermediate")).shuffle()
    eval_set = ArgoverseInMem(os.path.join(args.data_root, "val_intermediate"))

    # init output dir
    time_stamp = datetime.now().strftime("%m-%d-%H-%M")
    output_dir = os.path.join(args.output_dir, time_stamp)

    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        raise Exception("The output folder does exists and is not empty! Check the folder.")
    else:
        os.makedirs(output_dir)

        with open(os.path.join(output_dir, "conf.json"), "w") as fp:
            json.dump(vars(args), fp, indent=4, separators=(",", ":"))

    # init trainer
    trainer = VectorNetTrainer(
        trainset=train_set,
        evalset=eval_set,
        testset=eval_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        warmup_epoch=args.warmup_epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_update_freq=args.lr_update_freq,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        num_global_graph_layer=args.num_glayer,
        aux_loss=args.aux_loss,
        with_cuda=args.with_cuda,
        cuda_device=0,
        save_folder=output_dir,
        log_freq=args.log_freq,
        ckpt_path=args.resume_checkpoint if hasattr(args, "resume_checkpoint") and args.resume_checkpoint else None,
        model_path=args.resume_model if hasattr(args, "resume_model") and args.resume_model else None
    )

    # resume minimum eval loss
    min_eval_loss = trainer.min_eval_loss

    # training
    for iter_epoch in range(args.n_epoch):
        _ = trainer.train(iter_epoch)

        eval_loss = trainer.eval(iter_epoch)

        if not min_eval_loss:
            min_eval_loss = eval_loss
        elif eval_loss < min_eval_loss:
            min_eval_loss = eval_loss
            trainer.save(iter_epoch, min_eval_loss)
            trainer.save_model("best")

    trainer.save_model("final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", required=False, type=str, default="data/interm_data",
                        help="root dir for datasets")
    parser.add_argument("-o", "--output_dir", required=False, type=str, default="run/vectornet",
                        help="dir to save checkpoint and model")
    parser.add_argument("-l", "--num_glayer", type=int, default=1)
    parser.add_argument("-a", "--aux_loss", action="store_true", default=True)
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-e", "--n_epoch", type=int, default=40)
    parser.add_argument("-w", "--num_workers", type=int, default=0)
    parser.add_argument("-c", "--with_cuda", action="store_true", default=True)
    parser.add_argument("--log_freq", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("-we", "--warmup_epoch", type=int, default=10)
    parser.add_argument("-luf", "--lr_update_freq", type=int, default=5)
    parser.add_argument("-ldr", "--lr_decay_rate", type=float, default=0.9)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("-rc", "--resume_checkpoint", type=str)
    parser.add_argument("-rm", "--resume_model", type=str)

    args = parser.parse_args()
    train(args)



