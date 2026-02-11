import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir, experiment_name, **kwargs):
        path = os.path.join(log_dir, experiment_name)
        os.makedirs(path, exist_ok=True)
        self.writer = SummaryWriter(log_dir=path, **kwargs)

    def log_scalar(self, tag, value, step, **kwargs):
        self.writer.add_scalar(tag, value, step, **kwargs)

    def log_scalars(self, main_tag, tag_scalar_dict, step, **kwargs):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step, **kwargs)

    def log_image(self, tag, img_tensor, step, **kwargs):
        self.writer.add_image(tag, img_tensor, step, **kwargs)

    def log_images(self, tag, img_tensor, step, **kwargs):
        self.writer.add_images(tag, img_tensor, step, **kwargs)

    def log_figure(self, tag, figure, step, **kwargs):
        self.writer.add_figure(tag, figure, step, **kwargs)

    def log_histogram(self, tag, values, step, **kwargs):
        self.writer.add_histogram(tag, values, step, **kwargs)

    def log_graph(self, model, input_to_model=None, **kwargs):
        self.writer.add_graph(model, input_to_model, **kwargs)

    def close(self):
        self.writer.close()