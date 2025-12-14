import os

import torch

from config.config import config as service_config
from self_forcing.wan.modules import inference_utils

inference_utils.COMPILE = service_config.lip_sync.compile
inference_utils.NO_REFRESH_INFERENCE = service_config.lip_sync.no_refresh_inference

import logging

from core.app_interface import main as interface_main
from core.distributed import launch_distributed_job
from core.dit_service import main as dit_main
from self_forcing.utils import parallel_state as mpu

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.root.setLevel(service_config.log_level)

if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    sp_size = max(0, world_size - 1)
    logger = logging.getLogger(__name__)

    # Initialize distributed inference
    launch_distributed_job()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    torch.set_grad_enabled(False)

    logger.info(f"Rank {local_rank}: Starting service with sp_size {sp_size}...")

    if local_rank == 0:
        interface_main()
    else:
        # Only run dit_main if we have multiple GPUs
        if sp_size > 0:
            dit_main()
        else:
            logger.info(f"Rank {local_rank}: Skipping dit_main (single GPU mode)")

    torch.distributed.barrier()
    mpu.destroy_parallel_groups()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
