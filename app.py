import logging
import os

import torch

from config.config import config as service_config

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.root.setLevel(service_config.log_level)
logger = logging.getLogger(__name__)


if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]


def run_wanmuse_interface() -> None:
    """Run the decoupled TI2V-5B + MuseTalk backend as a single process."""

    from core.app_interface import main as interface_main

    torch.set_grad_enabled(False)
    logger.info(
        "Starting single-process interface with video backend %s",
        service_config.video.backend,
    )
    interface_main()


def run_self_forcing_distributed() -> None:
    """Preserve the original rank-0 interface and rank-1+ DiT topology."""

    from core.app_interface import main as interface_main
    from core.distributed import launch_distributed_job
    from core.dit_service import main as dit_main
    from self_forcing.utils import parallel_state as mpu
    from self_forcing.wan.modules import inference_utils

    inference_utils.COMPILE = service_config.lip_sync.compile
    inference_utils.NO_REFRESH_INFERENCE = service_config.lip_sync.no_refresh_inference

    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    sp_size = max(0, world_size - 1)

    launch_distributed_job()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    logger.info("Rank %d: Starting service with sp_size %d", local_rank, sp_size)
    if local_rank == 0:
        interface_main()
    elif sp_size > 0:
        dit_main()
    else:
        logger.info("Rank %d: Skipping dit_main (single GPU mode)", local_rank)

    torch.distributed.barrier()
    mpu.destroy_parallel_groups()
    torch.distributed.destroy_process_group()


def main() -> None:
    if service_config.video.backend == "ti2v5b_musetalk":
        run_wanmuse_interface()
    else:
        run_self_forcing_distributed()


if __name__ == "__main__":
    main()
