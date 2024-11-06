import core.logger as Logger
import torch.utils.data as data


#Create chaneg detection dataset
def create_cd_dataloader(dataset, dataset_opt, phase):
    if phase == 'train' or 'val':
        return data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            num_workers=dataset_opt['num_workers'],
            shuffle=True,
            pin_memory=True
        )
    elif phase == 'test':
        return data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            num_workers=dataset_opt['num_workers'],
            shuffle=False,
            pin_memory=True
        )
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found'.format(phase)
        )

def create_cd_dataset(dataset_opt, phase):
    from .CDDatasets import CDDataset
    dataset = CDDataset(root_dir=dataset_opt["datasetroot"],
                        resolution=dataset_opt["resolution"],
                        data_len=dataset_opt["data_len"],
                        split=phase)
    Logger.log_message('Dataset [{:s} - {:s} - {:s}] is created'.\
                       format(dataset.__class__.__name__,dataset_opt['name'],phase))
    return dataset
