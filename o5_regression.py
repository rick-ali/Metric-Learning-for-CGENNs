import engineer


def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    model_config = config["model"]
    model_module = engineer.load_module(model_config.pop("module"))
    model = model_module(dataset.ymean, dataset.ystd, **model_config)

    model = model.cuda()

    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    scheduler = None
    trainer_module = engineer.load_module(config["trainer"].pop("module"))

    trainer_config = config["trainer"]
    trainer_config["scheduler"] = scheduler
    trainer = trainer_module(
        **trainer_config,
    )

    trainer.fit(model, optimizer, train_loader, config["save_name"], val_loader, test_loader)


if __name__ == "__main__":
    engineer.fire(main)
