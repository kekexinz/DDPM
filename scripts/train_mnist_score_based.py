import argparse
import datetime
import torch
import functools
from tqdm import trange, tqdm

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torch.optim.lr_scheduler import LambdaLR

from ddpm import script_utils


def main():
    args = create_argparser().parse_args()
    device = args.device
    
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        total_params = sum(p.numel() for p in diffusion.model.parameters())
        trainable_params = sum(p.numel() for p in diffusion.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
        print("-----------------------------")
        

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size
        dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(diffusion.model.parameters(), lr=args.learning_rate)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
        tqdm_epoch = trange(args.iterations)

        print("Starting training")
        
        for epoch in tqdm(tqdm_epoch):
            avg_loss = 0.
            num_items = 0
            for x, y in tqdm(data_loader):
                x = x.to(device)
                y = y.to(device)

                if args.use_labels:
                    loss = diffusion(x, y)
                else:
                    loss = diffusion(x)

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]

                if epoch % 10 == 0:
                    print(f"Epoch: {epoch}, Loss: {loss:5f}")
            
            scheduler.step()
            lr_current = scheduler.get_last_lr()[0]
            print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
            # Print the averaged training loss so far.
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            # Update the checkpoint after each epoch of training.
            torch.save(diffusion.state_dict(), f'ckpt_score_based_model.pth')
            
    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    defaults = dict(
        learning_rate=10e-4,
        batch_size=1024,
        iterations=100,
        log_rate=100,
        time_emb_dim=128,
        checkpoint_rate=1000,
        log_dir="./ddpm_logs",
        project_name=None,
        run_name=run_name,
        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()