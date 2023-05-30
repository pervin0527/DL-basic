import os, shutil, math, torch
from torch.utils.tensorboard import SummaryWriter

log_dir = "/home/pervinco/tensorboard_test"
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)

writer = SummaryWriter(log_dir=log_dir)

for step in range(-360, 360):
    angle_rad = step * math.pi / 180
    cos = torch.tensor(math.cos(angle_rad))
    sin = torch.tensor(math.sin(angle_rad))
    writer.add_scalar("add_scalar/cosine", cos, step)
    writer.add_scalar("add_scalar/sine", sin, step)

    writer.add_scalars("add_sclaras/scalars", {"cos" : cos, "sin" : sin}, global_step=step)

writer.close()