import os
from argparse import ArgumentParser

import numpy as np
import time
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from methods.weight_methods import WeightMethods
from experiments.celeba.cone_initial_celeba import *

def delta_fn(a):
    BASE = np.array(
    [0.6736886,  0.68121034, 0.81524944, 0.5760289,  0.7205613,  0.8555076,
 0.38203922, 0.58225113, 0.787647,   0.8321292,  0.5029583,  0.68694085,
 0.6781237,  0.5240381,  0.5161666,  0.95694304, 0.6968786,  0.67976356,
 0.8808315,  0.8582131,  0.97034,    0.93267566, 0.5057539,  0.40307626,
 0.9703734,  0.48644206, 0.60786104, 0.5261031,  0.56907415, 0.59815097,
 0.6858371,  0.924108,   0.5424991,  0.7406311,  0.71019936, 0.87365365,
 0.9305602,  0.33704284, 0.7647628,  0.91907   ])  # base results from FAMO
    SIGN = np.ones(40, dtype='int64')
    KK = np.ones(40) * -1
    
    return (KK ** SIGN * (a - BASE) / BASE).mean() * 100.0  # * 100 for percentage


class CelebaMetrics():
    """
    CelebA metric accumulator.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        # y_preds: [ y_pred (batch, 1) ] x 40
        # ys     : [ y_pred (batch, 1) ] x 40
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1,2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1,2])
        self.fn += ((1 - y_preds) * ys).sum([1,2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()


def main(path, lr, bs, device, args):
    # we only train for specific task
    model = Network().to(device)
    
    train_set = CelebaDataset(data_dir=path, split='train')
    val_set   = CelebaDataset(data_dir=path, split='val')
    test_set  = CelebaDataset(data_dir=path, split='test')

    # 初始化阶段
    if args.init_type == "rbd_multitask_curvature_fullspace":
        init_data_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2
        )
        model = rbd_init_multitask_curvature_fullspace(
            model,
            init_data_loader,
            device,
            steps=args.steps,
            lr=args.lr_init,
            beta=args.beta_init,
            init_epochs=args.init_epochs,
        )

    elif args.init_type == "rbd_multitask_curvature_spectral_once":
        init_data_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2
        )
        model = rbd_init_multitask_curvature_spectral_once(
            model,
            init_data_loader,
            device,
            steps=args.steps,
            lr=args.lr_init,
            beta=args.beta_init,
            init_epochs=args.init_epochs,
        )
    elif args.init_type is None:
        pass  # No initialization
    else:
        raise ValueError(f"Invalid initialization type: {args.init_type}")

    train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=bs, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
            dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs    = args.n_epochs

    metrics   = np.zeros([epochs, 40], dtype=np.float32) # test_f1
    metric    = CelebaMetrics()
    loss_fn   = torch.nn.BCELoss()

    # weight method
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=40, device=device, **weight_methods_parameters[args.method]
    )

    best_val_f1 = 0.0
    best_epoch = None

    for epoch in range(epochs):
        # training
        model.train()
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
            optimizer.zero_grad()
            loss, extra_outputs = weight_method.backward(
                losses=losses,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
            )
            optimizer.step()
            if "famo" in args.method:
                with torch.no_grad():
                    y_ = model(x)
                    new_losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                    weight_method.method.update(new_losses.detach())
        t1 = time.time()

        model.eval()
        # validation
        metric.reset()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        val_f1 = metric.result()
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_epoch = epoch

        # testing
        metric.reset()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = [y_.to(device) for y_ in y]
                y_ = model(x)
                losses = torch.stack([loss_fn(y_task_pred, y_task) for (y_task_pred, y_task) in zip(y_, y)])
                metric.incr(y_, y)
        test_f1 = metric.result()
        metrics[epoch] = test_f1

        test_delta_m = delta_fn(test_f1)
        
        t2 = time.time()
        print(f"[info] epoch {epoch+1} | train takes {(t1-t0)/60:.1f} min | test takes {(t2-t1)/60:.1f} min | delta m {test_delta_m:2f}")
        
        if "famo" in args.method:
            name = f"{args.method}_gamma{args.gamma}_sd{args.seed}"
        elif "fairgrad" in args.method:
            name = f"{args.method}_alpha{args.alpha}_sd{args.seed}"
        else:
            name = f"{args.method}_sd{args.seed}"

        torch.save({"metric": metrics, "best_epoch": best_epoch}, os.path.join(args.save_dir, f"{name}.stats"))


if __name__ == "__main__":
    parser = ArgumentParser("Celeba", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default=None,
        choices=["rbd_multitask_curvature_fullspace", "rbd_multitask_curvature_spectral_once", None],
        help="initialization type (if not specified, skip initialization)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="number of RBD steps (iterations)",
    )
    parser.add_argument(
        "--init_epochs",
        type=int,
        default=1,
        help="number of passes over init data",
    )
    parser.add_argument(
        "--lr_init",
        type=float,
        default=3e-4,
        help="learning rate for initialization",
    )
    parser.add_argument(
        "--beta_init",
        type=float,
        default=0.9,
        help="momentum beta used during initialization",
    )
   
  
    parser.add_argument("--save-dir", type=str, default="./save", help="Directory to save the results.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # 打印 GPU 信息到日志
    print(f"=== Experiment Setup ===")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    print(f"Selected Local GPU ID: {args.gpu}")
    print(f"========================")

    device = get_device(gpus=args.gpu)
    main(path=args.data_path,
         lr=args.lr,
         bs=args.batch_size,
         device=device,
         args=args)
