import time
import torch
import torch.distributed as dist
from lightning.pytorch import callbacks


class BatchDurationCallback(callbacks.Callback):

    def __init__(self):
        super().__init__()
        self.train_batch_start_time = 0
        self.total_train_batch_duration = 0.
        self.n_train_batch = 0
        self.val_batch_start_time = 0
        self.total_val_batch_duration = 0.
        self.n_val_batch = 0

    def on_train_batch_start(self, *args, **kwargs):
        self.train_batch_start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        train_batch_duration = time.time() - self.train_batch_start_time
        self.total_train_batch_duration += train_batch_duration
        self.n_train_batch += 1

    def on_train_epoch_end(self, trainer, pl_module):
        duration = self.total_train_batch_duration / self.n_train_batch
        e = pl_module.current_epoch
        te = trainer.max_epochs
        print(
            f"Epoch {e}/{te}: mean_train_batch_duration={duration*1000:.2f} ms"
        )

    def on_validation_batch_start(self, *args, **kwargs):
        self.val_batch_start_time = time.time()

    def on_validation_batch_end(self, *args, **kwargs):
        val_batch_duration = time.time() - self.val_batch_start_time
        self.total_val_batch_duration += val_batch_duration
        self.n_val_batch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        duration = self.total_val_batch_duration / self.n_val_batch
        e = pl_module.current_epoch
        te = trainer.max_epochs
        print(f"Epoch {e}/{te}: mean_val_batch_duration={duration*1000:.2f} ms")

    def load_state_dict(self, sd):
        self.total_train_batch_duration = sd["total_train_batch_duration"]
        self.n_train_batch = sd["n_train_batch"]
        self.total_val_batch_duration = sd["total_val_batch_duration"]
        self.n_val_batch = sd["n_val_batch"]

    def state_dict(self):
        return {
            "total_val_batch_duration": self.total_val_batch_duration,
            "n_val_batch": self.n_val_batch,
            "total_train_batch_duration": self.total_train_batch_duration,
            "n_train_batch": self.n_train_batch,
        }


class PeakMemoryTillNowCallback(callbacks.Callback):

    def on_fit_start(self, *args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        mem_stats = torch.cuda.memory_stats()
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        print(
            f"Before training: "
            f"peak_allocated={peak_allocated} MB, "
            f"peak_reserved={peak_reserved} MB"
        )

    def on_train_epoch_end(self, trainer, pl_module):
        mem_stats = torch.cuda.memory_stats()
        peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
        peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
        e = pl_module.current_epoch
        te = trainer.max_epochs
        print(
            f"Epoch {e}/{te}: peak_allocated={peak_allocated} MB, "
            f"peak_reserved={peak_reserved} MB"
        )
