# flower classifier
This project uses convolutional neural network to train an image classifier that is able to identify 102 different flower species with 92% testing accuracy. This image classifier can be used to identify flower species from new images.

##Data used
[Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). This dataset contains images of 102 different flower species with lables. These images have different sizes.

## how to run the command line application
- #### Train the image classifier

    [`train.py`](train.py): Train the image classifier, report validation accuracy along training, and save the trained model as a checkpoint.

    - Basic usage:
        - Specify directory of image data: `python train.py flowers`

    - Options:
        - Set directory to save checkpoints: `python train.py flowers --save_dir assets`
        - Choose architecture: `python train.py flowers --arch "vgg16"`
        - Set hyperparameters: `python train.py flowers --lr 0.001 --hidden_units 512 --epochs 20`
        - Use GPU for training: `python train.py flowers --gpu`
        
        - #### Identify flower name from a new image

    [`predict.py`](predict.py): Use the trained image classifier to predict flower name along with the probability of that name

    - Basic usage: 
        - Specify file path of the image and directory name of saved checkpoint: `python predict.py flowers/test/1/image_06743.jpg assets`

    - Options:
        - Return top K most likely classes: `python predict.py flowers/test/1/image_06743.jpg assets --top_k 3`
        - Use a mapping of categories to real names: `python predict.py flowers/test/1/image_06743.jpg assets --category_names cat_to_name.json`
        - Use GPU for inference: `python predict.py flowers/test/1/image_06743.jpg assets --gpu`
