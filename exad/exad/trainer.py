from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .datamodule import AttrDataModule
from .models import ExAD_AE_Module, ExAD_CNN_Module

from jsonargparse import CLI


def runner(
    mode: str,
    model_name: str,
    dataset_name: str,
    adv_name: str,
    attr_name: str,
    target: int,
    data_dir: str,
    project_name: str = "exad",
    exad_outdir: str = "exad_out",
):

    run_name = f"{model_name}_{dataset_name}_{adv_name}_{attr_name}_{target}"
    if mode == "AE":
        model = ExAD_AE_Module(
            setting_name=run_name, hidden_sizes=[400, 20, 400], outdir=exad_outdir
        )
    else:
        model = ExAD_CNN_Module(setting_name=run_name, outdir=exad_outdir)

    datamodule = AttrDataModule(
        mode=mode,
        model_name=model_name,
        dataset_name=dataset_name,
        adv_name=adv_name,
        attr_name=attr_name,
        target=target,
        data_dir=data_dir,
    )

    logger = TensorBoardLogger(f"{project_name}_logs", run_name)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor="val_auroc"))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        max_epochs=200,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    CLI(runner, as_positional=False)
