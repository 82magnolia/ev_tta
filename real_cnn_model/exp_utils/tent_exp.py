from real_cnn_model.data.data_container import EvTTAImageNetContainer
from real_cnn_model.models.model_container import EvTTACNNContainer
from real_cnn_model.train.trainer import EvTTACNNTrainer
import configparser


def run_exp(cfg):
    """
    Run normal train / test
    
    Args:
        cfg: Config file containing configs related to experiment
    """

    # Make instance of data container
    data_container = EvTTAImageNetContainer(cfg)

    # Make instance of model container
    model_container = EvTTACNNContainer(cfg)

    # Make instance of trainer
    trainer = EvTTACNNTrainer(cfg, model_container, data_container)

    config = configparser.ConfigParser()
    config.add_section('Default')

    cfg_dict = cfg._asdict()

    for key in cfg_dict:
        if key != 'name':
            config['Default'][key] = str(cfg_dict[key]).replace('[', '').replace(']', '')
        else:
            config['Default'][key] = str(cfg_dict[key])

    with open(trainer.exp_save_dir / 'config.ini', 'w') as configfile:
        config.write(configfile)

    # Display model
    print(model_container.models['model'])

    trainer.run()
