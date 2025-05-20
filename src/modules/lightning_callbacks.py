import time
import torch
import torch.distributed as dist
from lightning.pytorch import callbacks


class BatchDurationCallback(callbacks.Callback):

    def __init__(self, avg_per_epoch: bool = False):
        super().__init__()
        # all local
        self.train_batch_start_time = 0
        self.total_train_batch_duration = 0.
        self.n_train_batch = 0
        self.val_batch_start_time = 0
        self.total_val_batch_duration = 0.
        self.n_val_batch = 0
        self.avg_per_epoch = avg_per_epoch

    def on_train_batch_start(self, *args, **kwargs):
        self.train_batch_start_time = time.time()

    def on_train_batch_end(self, *args, **kwargs):
        train_batch_duration = time.time() - self.train_batch_start_time
        self.total_train_batch_duration += train_batch_duration
        self.n_train_batch += 1

    def on_train_epoch_end(self, trainer, pl_module):
        total_duration = self.total_train_batch_duration
        n_batch = self.n_train_batch
        if dist.is_available() and dist.is_initialized():
            # synchronize across all processes
            total_duration_tensor = torch.tensor(
                [self.total_train_batch_duration, self.n_train_batch],
                dtype=torch.float,
                device=trainer.device,
            )
            dist.all_reduce(total_duration_tensor, op=dist.ReduceOp.SUM)
            total_duration, n_batch = total_duration_tensor.tolist()

        # compute global average duration
        avg_duration = (total_duration / n_batch) if n_batch > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(
                f"Epoch {e}/{te}: "
                f"mean_train_batch_duration={avg_duration*1000:.2f} ms"
            )

        if self.avg_per_epoch:
            self.total_train_batch_duration = 0.0
            self.n_train_batch = 0

    def on_validation_batch_start(self, *args, **kwargs):
        self.val_batch_start_time = time.time()

    def on_validation_batch_end(self, *args, **kwargs):
        val_batch_duration = time.time() - self.val_batch_start_time
        self.total_val_batch_duration += val_batch_duration
        self.n_val_batch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        # synchronize across all processes
        total_duration = self.total_val_batch_duration
        n_batch = self.n_val_batch
        if dist.is_available() and dist.is_initialized():
            total_duration_tensor = torch.tensor(
                [self.total_val_batch_duration, self.n_val_batch],
                dtype=torch.float32,
                device=trainer.device,
            )
            dist.all_reduce(total_duration_tensor, op=dist.ReduceOp.SUM)
            total_duration, n_batch = total_duration_tensor.tolist()

        # compute global average duration
        avg_duration = (total_duration / n_batch) if n_batch > 0 else 0.0

        if trainer.is_global_zero:
            e = pl_module.current_epoch
            te = trainer.max_epochs
            print(
                f"Epoch {e}/{te}: "
                f"mean_val_batch_duration={avg_duration*1000:.2f} ms"
            )

        if self.avg_per_epoch:
            self.total_val_batch_duration = 0.0
            self.n_val_batch = 0

    def load_state_dict(self, sd):
        self.total_train_batch_duration = sd["total_train_batch_duration"]
        self.n_train_batch = sd["n_train_batch"]
        self.total_val_batch_duration = sd["total_val_batch_duration"]
        self.n_val_batch = sd["n_val_batch"]
        self.avg_per_epoch = sd["avg_per_epoch"]

    def state_dict(self):
        return {
            "total_val_batch_duration": self.total_val_batch_duration,
            "n_val_batch": self.n_val_batch,
            "total_train_batch_duration": self.total_train_batch_duration,
            "n_train_batch": self.n_train_batch,
            "avg_per_epoch": self.avg_per_epoch,
        }


class PeakMemoryTillNowCallback(callbacks.Callback):

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_stats = torch.cuda.memory_stats()
            peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
            peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
            if trainer.is_global_zero:
                print(
                    f"Before training: "
                    f"peak_allocated={peak_allocated} MB, "
                    f"peak_reserved={peak_reserved} MB",
                    flush=True,
                )

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            mem_stats = torch.cuda.memory_stats()
            peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
            peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)

            if dist.is_available() and dist.is_initialized():
                peak_tensor = torch.tensor(
                    [peak_allocated, peak_reserved],
                    dtype=torch.float32,
                    device=trainer.device,
                )
                dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
                peak_allocated, peak_reserved = peak_tensor.tolist()

            if trainer.is_global_zero:
                e = pl_module.current_epoch
                te = trainer.max_epochs
                print(
                    f"Epoch {e}/{te}: "
                    f"peak_allocated={peak_allocated} MB, "
                    f"peak_reserved={peak_reserved} MB",
                    flush=True
                )
